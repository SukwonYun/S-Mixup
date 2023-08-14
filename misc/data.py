import torch
import numpy as np
import random
import copy
import os
from enum import Enum, auto

from torch_geometric.datasets import Planetoid, Coauthor
import torch_geometric.transforms as T

from .utils import get_dir, DIR_TYPE

class datasetName(Enum):
    Cora = auto()
    CiteSeer = auto()
    PubMed = auto()
    CS = auto()
    Physics = auto()


class dataSplit(Enum):
    STANDARD = auto()

class dataloader:
    def __init__(self, dataset_name:datasetName):
        self.data = self._get_data(dataset_name)

    def _get_data(self, dataset_name:datasetName):
        # load data
        try:
            path = get_dir(DIR_TYPE.data_dir, dataset_name.name)
            dataset = None
            if datasetName.Cora == dataset_name \
                or datasetName.CiteSeer == dataset_name \
                or datasetName.PubMed == dataset_name:
                dataset = Planetoid(path, dataset_name.name, transform=T.NormalizeFeatures())
            elif datasetName.CS == dataset_name \
                or dataset_name.Physics == dataset_name:
                dataset = Coauthor(path, dataset_name.name, transform=T.NormalizeFeatures())

            data = dataset[0]
            data.num_classes = dataset.num_classes
        except Exception as e:
            print(e)
    
        return data

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def split_data(self, split:dataSplit):
        if dataSplit.STANDARD == split:
            self._use_standard_split()

    def _use_standard_split(self):
        if hasattr(self.data, 'train_mask') and hasattr(self.data, 'val_mask') and hasattr(self.data, 'test_mask'):
            node_id_2split_data = np.arange(self.data.num_nodes)
            self.data.train_id = node_id_2split_data[self.data.train_mask]
            self.data.val_id = node_id_2split_data[self.data.val_mask]
            self.data.test_id = node_id_2split_data[self.data.test_mask]
            labeled = ~torch.logical_or(self.data.train_mask, torch.logical_or(self.data.val_mask, self.data.test_mask))
            self.data.unlabeled_id = node_id_2split_data[labeled]
        else:
            print('No standard split setting, make standard split')
            self._make_standard_split()
            
    def _make_standard_split(self) :
        node_id_2split_data = np.arange(self.data.num_nodes)
        np.random.shuffle(node_id_2split_data)
        self.data.train_id = np.array([], dtype=int)
        self.data.val_id = np.array([], dtype=int)
        self.data.test_id = np.array([], dtype=int)
        self.data.unlabeled_id = []
        for class_label in range(self.data.num_classes):
            num_sample = self.data.y[self.data.y == class_label].size()[0]
            
            if num_sample >= 100:
                self.data.train_id = np.concatenate((self.data.train_id, node_id_2split_data[self.data.y[node_id_2split_data] == class_label][:20]))
                self.data.val_id = np.concatenate((self.data.val_id, node_id_2split_data[self.data.y[node_id_2split_data] == class_label][20:50]))
                self.data.test_id = np.concatenate((self.data.test_id, node_id_2split_data[self.data.y[node_id_2split_data] == class_label][50:]))
            else:
                self.data.train_id = np.concatenate((self.data.train_id, node_id_2split_data[self.data.y[node_id_2split_data] == class_label][:int(num_sample * 0.2)]))
                self.data.val_id = np.concatenate((self.data.val_id, node_id_2split_data[self.data.y[node_id_2split_data] == class_label][int(num_sample * 0.2):int(num_sample * 0.5)]))
                self.data.test_id = np.concatenate((self.data.test_id, node_id_2split_data[self.data.y[node_id_2split_data] == class_label][int(num_sample * 0.5):]))

        data_index = np.arange(self.data.y.shape[0])
        self.data.train_mask = torch.tensor(np.in1d(data_index, self.data.train_id), dtype=torch.bool)    