import torch
from torch_scatter import scatter_mean

def compute_mean(x, edge_index):
    row, col = edge_index
    mean_features = scatter_mean(x[row], col, dim=0)
    return mean_features

def compute_var(x, edge_index):
    row, col = edge_index
    mean_features = scatter_mean(x[row], col, dim=0)
    mean_sq = scatter_mean((x*x)[row], col, dim=0)
    var_features = mean_sq - mean_features * mean_features
    return var_features

def compute_cov(x, edge_index):
    row, col = edge_index
    mean_features = scatter_mean(x[row], col, dim=0)
    xxT = torch.einsum('bi,bj->bij', x, x)
    xxT = xxT[row,...]
    shapes = xxT.size()
    xxT = xxT.reshape(-1, x.size(1) * x.size(1))
    mean_xxT = scatter_mean(xxT, col, dim=0).reshape(-1, x.size(1), x.size(1))
    cov = mean_xxT - torch.einsum('bi,bj->bij', mean_features, mean_features)
    return cov

"""
(manual, for testing, slow)
"""
def mean_manual(x, edge_index):
    means = torch.zeros(edge_index[1].max()+1, x.size(1))
    for i in range(edge_index[1].max()+1):
        ids = edge_index[0][edge_index[1] == i]
        means[i,:] = x[ids,:].mean(dim=0)
    return means