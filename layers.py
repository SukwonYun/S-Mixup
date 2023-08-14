from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from enum import Enum, auto


class GNN(Enum):
    GCN = auto()

def get_model(layer:str, _in_channels:int, _out_channels:int, _hidden_channels:int, dropout:float):
    if layer.lower() == 'gcn':
        model = GCN(hidden_channels=_hidden_channels, in_channels=_in_channels, out_channels=_out_channels, dropout=dropout)
    
    return model

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)