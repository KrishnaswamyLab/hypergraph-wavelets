import torch
from torch_geometric.utils import scatter

def compute_mean(x, edge_index):
    row, col = edge_index
    mean_features = scatter(x[row], col, dim=0, reduce='mean')
    return mean_features

def compute_var(x, edge_index):
    row, col = edge_index
    mean_features = scatter(x[row], col, dim=0, reduce='mean')
    mean_sq = scatter((x*x)[row], col, dim=0, reduce='mean')
    var_features = mean_sq - mean_features * mean_features
    return var_features

def compute_cov(x, edge_index):
    row, col = edge_index
    mean_features = scatter(x[row], col, dim=0, reduce='mean')
    xxT = torch.einsum('bi,bj->bij', x, x)
    xxT = xxT[row,...]
    shapes = xxT.size()
    xxT = xxT.reshape(-1, x.size(1) * x.size(1))
    mean_xxT = scatter(xxT, col, dim=0, reduce='mean').reshape(-1, x.size(1), x.size(1))
    cov = mean_xxT - torch.einsum('bi,bj->bij', mean_features, mean_features)
    return cov