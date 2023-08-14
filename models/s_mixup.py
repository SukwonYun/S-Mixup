from embedder import embedder, Result_handler
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from timeit import default_timer as timer
import copy
from layers import get_model
import torch_geometric.utils as tg_utils
from tqdm import trange

class s_mixup():
    def __init__(self, args):
        self.args = args
    

    def train(self):
        start_trial = timer()
        self = embedder(self.args)
        result_handler = Result_handler(self.args.n_runs)

        for run in range(self.args.n_runs):
            data = copy.deepcopy(self.data_obj.data).to(self.device)
            data.train_id_new = []

            model = modeler(self.args, self.device)
            
            result_handler.start_epoch()
            max_pseudo_size = data.train_mask.sum()*2
            og_y = copy.deepcopy(data.y)

            _pseudo_idx = None
            _pseudo_y = None
            
            for epoch in trange(self.args.n_epochs):
                mixup = True if epoch > self.args.mixup_start else False

                loss, pseudo_idx, pseudo_y = model(train=True, mixup=mixup, data=data, pseudo_idx=_pseudo_idx, pseudo_y=_pseudo_y, og_edge=data.edge_index, og_y=og_y)
    
                if mixup and (data.train_mask.sum() < max_pseudo_size):
                    if len(pseudo_idx) > 0:
                        # self-train
                        data.train_mask[pseudo_idx] = True
                        _pseudo_idx = pseudo_idx
                        _pseudo_y = pseudo_y

                loss_val, accs = model(train=False, data=data, og_edge=data.edge_index, og_y=og_y)
                result_handler.update_result(accs)

            best_acc_train, best_acc_test = result_handler.end_epoch()

            print(f'Seed:{run+1}, Train Acc: {best_acc_train[-1]:.4f}, best acc: {best_acc_test:.4f}')
            
        rst = result_handler.results()
        
        return rst


class modeler(nn.Module):
    def __init__(self, args, device):
        super(modeler, self).__init__()
        self.args = args
        self.device = device
        
        self.model = get_model('gcn', self.args.n_features, self.args.n_classes, self.args.hidden_dim, self.args.dropout).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

    def forward(self, train=False, mixup=True, data=None, pseudo_idx=None, pseudo_y=None, og_edge=None, og_y=None):
        if train: # train
            self.model.train()

            data = copy.deepcopy(data).to(self.device)

            _pseudo_idx = None
            _pseudo_y = None
            _pseudo_conf = None
            conf_ = None

            if not mixup: # w/o mixup
                self.optimizer.zero_grad()
                out = self.model(data.x, data.edge_index)
                loss = F.nll_loss(out[data.train_id], data.y[data.train_id])

                loss.backward()

                self.optimizer.step()

            else : # w/ mixup
                lam = np.random.beta(self.args.alpha, self.args.alpha)
                lam = np.maximum(lam, 1-lam)
                self.optimizer.zero_grad()

                data.x.requires_grad = True
                data.edge_weight = torch.ones((data.edge_index.size(1), ), dtype=None, device=data.edge_index.device)
                data.edge_weight.requires_grad = True

                out_4grad = self.model(data.x, data.edge_index, data.edge_weight)
                loss = F.nll_loss(out_4grad[data.train_id], data.y[data.train_id])
                
                if pseudo_idx != None:
                    _loss_pseudo = F.nll_loss(out_4grad[pseudo_idx], pseudo_y, reduction='none')
                    loss += torch.mean(_loss_pseudo * 1)
                
                loss.backward()

                self.optimizer.step()

                self.optimizer.zero_grad()
                grads_e = F.normalize(torch.sqrt(data.edge_weight.grad**2), dim = 0)

                conf_ = torch.softmax(out_4grad, dim=1)
                pred_conf, pred = out_4grad.max(dim=1)
                min_conf, _ = out_4grad.min(dim=1)
                binary_conf = torch.softmax(torch.stack([pred_conf.T, min_conf.T]), dim=0).T
                entropy_max_min = -(binary_conf.T[0]*torch.log(binary_conf.T[0]) + binary_conf.T[1]*torch.log(binary_conf.T[1]))

                conf_top2, _ = torch.topk(out_4grad, 2, dim=1)
                binary_conf = torch.softmax(conf_top2, dim=1)

                num_node_inclass = torch.tensor([(pred==i).count_nonzero().item() for i in range(data.num_classes)], dtype=int)
                grad_pool_size = torch.round(num_node_inclass*self.args.target_ratio).to(int)
                grad_pool_size = torch.where(grad_pool_size.nonzero().size() != 0 and grad_pool_size < num_node_inclass, grad_pool_size, num_node_inclass)

                pred_top_pool = list(range(data.num_classes))
                grad_top = list(range(data.num_classes))
                ent_pool = list(range(data.num_classes))
                
                grads_sorted = torch.argsort(pred_conf, descending=True)
                ent_sorted = torch.argsort(entropy_max_min, descending=True)

                for i in range(data.num_classes):
                    if grad_pool_size[i] != 0:
                        pred_top_pool[i] = grads_sorted[pred[grads_sorted] == i][:grad_pool_size[i]]
                        grad_top[i] = torch.randint(0, grad_pool_size[i], (1, grad_pool_size[i]))[0]
                        ent_pool[i] = ent_sorted[pred[ent_sorted] == i][:grad_pool_size[i]]
                    else:
                        pred_top_pool[i] = torch.tensor([], dtype=int).to(self.device)
                        grad_top[i] = torch.tensor([], dtype=int).to(self.device)
                        ent_pool[i] = torch.tensor([], dtype=int).to(self.device)

                MASKING_INT = -1
                
                # intra-class mixup
                mixup_source = torch.cat([pred_top_pool[i] for i in range(data.num_classes)]).to(self.device)
                mixup_target = torch.cat([ent_pool[i] for i in range(data.num_classes)]).to(self.device)

                target_newid = torch.full((data.num_nodes, ), MASKING_INT, dtype=int).to(self.device)
                source_newid = torch.full((data.num_nodes, ), MASKING_INT, dtype=int).to(self.device)
                target_newid[mixup_target] = torch.tensor(np.arange(data.num_nodes, data.num_nodes+mixup_target.size()[0]), dtype=int).to(self.device)
                source_newid[mixup_source] = torch.tensor(np.arange(data.num_nodes, data.num_nodes+mixup_source.size()[0]), dtype=int).to(self.device)
                
                # inter class mixup (leverage entropy)
                enth = ent_sorted[round(data.num_nodes//2-data.num_nodes*self.args.target_ratio):round(data.num_nodes//2+data.num_nodes*self.args.target_ratio)]
                enth_newid = torch.full((data.num_nodes, ), MASKING_INT, dtype=int).to(self.device)
                enth_newid[enth] = torch.tensor(np.arange(data.num_nodes+mixup_target.size()[0], data.num_nodes+mixup_target.size()[0]+enth.size()[0]), dtype=int).to(self.device)
                
                enth_shff = copy.deepcopy(enth).detach().cpu().numpy()
                np.random.shuffle(enth_shff)
                enth_shff = torch.Tensor(enth_shff).to(int)
                enth_shff_newid = torch.full((data.num_nodes, ), MASKING_INT, dtype=int).to(self.device)
                enth_shff_newid[enth] = torch.tensor(np.arange(data.num_nodes+mixup_target.size()[0], data.num_nodes+mixup_target.size()[0]+enth_shff.size()[0]), dtype=int).to(self.device)

                self.optimizer.zero_grad()
                data.x.requires_grad = False
                data.edge_weight.requires_grad = False
                
                intra_mixed = data.x[mixup_target]*lam + data.x[mixup_source]*(1-lam)
                inter_mixed = data.x[enth]*lam + data.x[enth_shff]*(1-lam)
                
                # gradient-based edge selection
                th_index = round(grads_e.size()[0]*self.args.edge_ratio)
                grad_th = torch.sort(grads_e, descending=True)[0][th_index]
                grad_conn_mask = grads_e > grad_th
            
                edge_conn_target = copy.deepcopy(data.edge_index.T[grad_conn_mask*(target_newid[data.edge_index[0]] != MASKING_INT)].T)
                edge_conn_target[0] = target_newid[edge_conn_target[0]]
                edge_conn_source = copy.deepcopy(data.edge_index.T[grad_conn_mask*(source_newid[data.edge_index[0]] != MASKING_INT)].T)
                edge_conn_source[0] = source_newid[edge_conn_source[0]]

                edge_conn_enth = copy.deepcopy(data.edge_index.T[grad_conn_mask*(enth_newid[data.edge_index[0]] != MASKING_INT)].T)
                edge_conn_enth[0] = enth_newid[edge_conn_enth[0]]
                edge_conn_enth_shff = copy.deepcopy(data.edge_index.T[grad_conn_mask*(enth_shff_newid[data.edge_index[0]] != MASKING_INT)].T)
                edge_conn_enth_shff[0] = enth_shff_newid[edge_conn_enth_shff[0]]

                edge_self_source = torch.cat([target_newid[mixup_target], source_newid[mixup_source], enth_newid[enth], enth_shff_newid[enth]]).to(self.device)
                edge_self_target = torch.cat([mixup_target, mixup_source, enth, enth])
                edge_self = torch.stack((edge_self_source, edge_self_target), dim=0)

                new_edge = torch.cat([data.edge_index, edge_conn_target, edge_conn_source, edge_conn_enth, edge_conn_enth_shff, edge_self], dim=1)
                new_edge = tg_utils.to_undirected(new_edge)
                new_x = torch.cat([data.x, intra_mixed, inter_mixed], dim=0)
                
                out = self.model(new_x, new_edge)

                # Final loss
                loss = self.args.eta * F.nll_loss(out[target_newid[mixup_target]], pred[mixup_source]) # intra-class
                loss += (1-self.args.eta) * (F.nll_loss(out[enth_newid[enth]], pred[enth])*lam + F.nll_loss(out[enth_shff_newid[enth_shff]], pred[enth_shff])*(1-lam)) # inter-class
                
                loss.backward()
                self.optimizer.step()

            if conf_ != None:
                topk = 1
                pred_conf_val, pred_label = conf_.max(dim=1)
                class_list = list(pred_label.unique().cpu().numpy())
                pred_label[data.train_mask] = -1
                topk_list = [min(topk, (pred_label == i).sum().item()) for i in class_list]

                pseudo_idx_list = []
                pseudo_conf_list = []

                [pseudo_idx_list.extend(list((pred_label == class_list[i]).nonzero().reshape(-1)[pred_conf_val[pred_label == class_list[i]].topk(topk_list[i]).indices].cpu().numpy())) for i in range(len(class_list))]
                [pseudo_conf_list.extend(list(pred_conf_val[pred_label == class_list[i]].topk(topk_list[i]).values.cpu().detach().numpy())) for i in range(len(class_list))]
                
                _pseudo_idx = torch.tensor(pseudo_idx_list).to(conf_.device)
                _pseudo_conf = torch.tensor(pseudo_conf_list).to(conf_.device)
                
                # Self-training
                remain_idx = _pseudo_conf > 0.8
                if len(remain_idx) > 0:
                    _pseudo_idx = _pseudo_idx[remain_idx]
                    _pseudo_y = pred_label[_pseudo_idx]

            return loss.item(), _pseudo_idx, _pseudo_y


        else: # test
            self.model.eval()
            data = copy.deepcopy(data).to(self.device)

            out = self.model(data.x, og_edge)
            loss = F.nll_loss(out[data.val_id], og_y[data.val_id])

            pred = out.argmax(dim=1)
            correct = pred.eq(og_y)

            accs = []
            for _, id_ in data('train_id', 'val_id', 'test_id'):
                accs.append(correct[id_].sum().item() / id_.shape[0])
            accs.append(correct)

            return loss, accs
