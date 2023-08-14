import scipy.sparse as sp
import numpy as np
import torch
import misc.utils
from misc.data import dataSplit, dataloader, datasetName
import os

class embedder:
    def __init__(self, args):
        self.args = args
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        # Data
        if 'cora' in self.args.dataset.lower():
            self.dataset = datasetName.Cora
        if 'citeseer' in self.args.dataset.lower():
            self.dataset = datasetName.CiteSeer
        if 'pubmed' in self.args.dataset.lower():
            self.dataset = datasetName.PubMed
        if 'cs' in self.args.dataset.lower():
            self.dataset = datasetName.CS
        if 'physics' in self.args.dataset.lower():
            self.dataset = datasetName.Physics

        self._load_data()
        self.args.n_features = self.data_obj.data.num_node_features
        self.args.n_classes =  self.data_obj.data.num_classes
         
    def _load_data(self):
        self.data_obj = dataloader(dataset_name=self.dataset)
        self.data_obj.set_random_seed(seed=self.args.seed)
        self.data_obj.split_data(dataSplit.STANDARD)


class Result_handler():
    class train_results():
        def __init__(self, best_acc, acc_train_max):
            self.best_acc = best_acc
            self.acc_train_max = acc_train_max

    def __init__(self, num_epoch):
        self.acc_train_4plot_mean = torch.zeros(1, num_epoch)
        self.acc_test_4plot_mean = torch.zeros(1, num_epoch)
        
        self.best_acc = []
        self.acc_train_max = []

    def start_epoch(self):
        self.acc_train_4plot_temp = []
        self.acc_test_4plot_temp = []
        self.best_acc_train, self.best_acc_val, self.best_acc_test = 0.0, 0.0, 0.0
        
        self.best_correct = []

    def update_result(self, accs):
        if accs[1] >= self.best_acc_val:
            self.best_acc_train = accs[0]
            self.best_acc_val, self.best_acc_test = accs[1], accs[2]
            self.best_correct = accs[3].int()
    
    def end_epoch(self):
        self.best_acc.append(self.best_acc_test)
        self.acc_train_max.append(self.best_acc_train)

        return self.acc_train_max, self.best_acc_test

    def results(self):
        self.train_results = Result_handler.train_results(self.best_acc, self.acc_train_max)
        
        return self.train_results