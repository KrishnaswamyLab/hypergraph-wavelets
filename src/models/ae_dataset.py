"""
Adapted From https://github.com/professorwug/autometric
and https://github.com/xingzhis/dmae/blob/main/src/data.py
"""
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.preprocessing import PowerTransformer as PT


class PointCloudDataset(torch.utils.data.Dataset):
    """
    Point Cloud Dataset
    """
    def __init__(self, pointcloud, distances, xor_dists, batch_size = 64, shuffle=True):
        self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        self.distances = torch.tensor(distances, dtype=torch.float32) # shape (n_channels, n_samples, n_samples)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.xor_dists = xor_dists

    def __len__(self):
        return len(self.pointcloud)
    
    def __getitem__(self, idx):
        if self.shuffle:
            batch_idxs = torch.randperm(len(self.pointcloud))[:self.batch_size]
        else:
            batch_idxs = torch.arange(idx, idx+self.batch_size) % len(self.pointcloud)
        batch = {}
        batch['x'] = self.pointcloud[batch_idxs]
        dist_mat = self.distances[batch_idxs, :][:, batch_idxs]
        triu_ind = np.triu_indices(dist_mat.size(0), k=1)
        batch['d'] = dist_mat[triu_ind[0], triu_ind[1]]   
        if len(self.xor_dists.shape) == 3:
            xor_dist_mat = self.xor_dists[:, batch_idxs, :][:, :, batch_idxs]
            batch['m'] = xor_dist_mat[:, triu_ind[0], triu_ind[1]]
        else:
            xor_dist_mat = self.xor_dists[batch_idxs, :][:, batch_idxs]
            batch['m'] = xor_dist_mat[triu_ind[0], triu_ind[1]]
        return batch

def dataloader_from_pc(pointcloud, distances, xor_dists, batch_size = 64, shuffle=True):
    dataset = PointCloudDataset(pointcloud, distances, xor_dists, batch_size, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=shuffle)
    return dataloader

# def train_and_testloader_from_pc(
def train_valid_loader_from_pc(
    pointcloud, distances, xor_dists, batch_size = 64, train_valid_split = 0.8, shuffle=True, seed=42
):
    X = pointcloud
    D = distances
    np.random.seed(seed)
    if shuffle:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        D = D[idxs][:,idxs]
    xD = xor_dists
    split_idx = int(len(X)*train_valid_split)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    D_train = D[:split_idx,:split_idx]
    D_test = D[split_idx:,split_idx:]
    if len(xor_dists.shape) == 3:
        xD_train = xD[:,:split_idx,:split_idx]
        xD_test = xD[:,split_idx:,split_idx:]
    else:
        xD_train = xD[:split_idx,:split_idx]
        xD_test = xD[split_idx:,split_idx:]
    trainloader = dataloader_from_pc(X_train, D_train, xD_train, batch_size)
    testloader = dataloader_from_pc(X_test, D_test, xD_test, batch_size)
    return trainloader, testloader




# TODO: make this more standard? Can (or should) I do batching in the dataloader instead?
# class PointCloudDataset(torch.utils.data.Dataset):
#     """
#     Point Cloud Dataset
#     """
#     def __init__(self, pointcloud, distances, xor_dists, batch_size = 64, shuffle=True):
#         self.pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
#         self.distances = torch.tensor(distances, dtype=torch.float32) # shape (n_channels, n_samples, n_samples)
#         self.shuffle = shuffle
#         self.batch_size = batch_size
#         self.xor_dists = xor_dists

#     def __len__(self):
#         return len(self.pointcloud)
    
#     def __getitem__(self, idx):
#         if self.shuffle:
#             batch_idxs = torch.randperm(len(self.pointcloud))[:self.batch_size]
#         else:
#             batch_idxs = torch.arange(idx, idx+self.batch_size) % len(self.pointcloud)
#         batch = {}
#         batch['x'] = self.pointcloud[batch_idxs]
#         dist_mat = self.distances[:, batch_idxs, :][:, :, batch_idxs]
#         triu_ind = np.triu_indices(dist_mat.size(0), k=1)
#         batch['d'] = dist_mat[:, triu_ind[0], triu_ind[1]]   
#         xor_dist_mat = self.xor_dists[:, batch_idxs, :][:, :, batch_idxs]

#         return batch

# def dataloader_from_pc(pointcloud, distances, xor_dists, batch_size = 64, shuffle=True):
#     dataset = PointCloudDataset(pointcloud, distances, xor_dists, batch_size, shuffle=shuffle)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=shuffle)
#     return dataloader

# def train_and_testloader_from_pc(
#     pointcloud, distances, xor_dists, batch_size = 64, train_test_split = 0.8
# ):
#     X = pointcloud
#     D = distances
#     xD = xor_dists
#     split_idx = int(len(X)*train_test_split)
#     X_train = X[:split_idx]
#     X_test = X[split_idx:]
#     D_train = D[:,:split_idx,:split_idx]
#     D_test = D[:,split_idx:,split_idx:]
#     xD_train = xD[:,:split_idx,:split_idx]
#     xD_test = xD[:,split_idx:,split_idx:]
#     trainloader = dataloader_from_pc(X_train, D_train, xD_train, batch_size)
#     testloader = dataloader_from_pc(X_test, D_test, xD_test, batch_size)
#     return trainloader, testloader

def train_valid_testloader_from_pc(
    pointcloud, distances, batch_size = 64, train_test_split = 0.8, train_valid_split = 0.8, shuffle=True, seed=42
):
    X = pointcloud
    D = distances
    np.random.seed(seed)
    if shuffle:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        D = D[:,idxs,:][:,:,idxs]
    split_idx = int(len(X)*train_test_split)
    split_val_idx = int(split_idx*train_valid_split)
    X_train = X[:split_val_idx]
    X_valid = X[split_val_idx:split_idx]
    X_test = X[split_idx:]
    D_train = D[:,:split_val_idx,:split_val_idx]
    D_valid = D[:,split_val_idx:split_idx,split_val_idx:split_idx]
    D_test = D[:,split_idx:,split_idx:]
    trainloader = dataloader_from_pc(X_train, D_train, batch_size)
    validloader = dataloader_from_pc(X_valid, D_valid, batch_size)
    testloader = dataloader_from_pc(X_test, D_test, batch_size)
    return trainloader, validloader, testloader

class LogTransform():
    def __init__(self, eps=1e-10, device=None):
        self.eps = eps
    def transform(self, X):
        return torch.log(X+self.eps)
    def fit_transform(self, X):
        return self.transform_cpu(X)
    def transform_cpu(self, X):
        return np.log(X+self.eps)
    
class NonTransform():
    def __init__(self, device=None):
        pass
    def transform(self, X):
        return X
    def fit_transform(self, X):
        return X
    
class StandardScaler():
    def __init__(self):
        self.ss = SS()
        self.mean_ = None
        self.std_ = None
    def fit_transform(self, X):
        res = self.ss.fit_transform(X)
        self.mean_ = torch.tensor(self.ss.mean_)
        self.std_ = torch.tensor(self.ss.scale_)
        return res
    def transform(self, X):
        return standard_scale_transform_torch(X, self.mean_, self.std_)

class MinMaxScaler():
    def __init__(self):
        self.mms = MMS()
        self.min_ = None
        self.scale_ = None
    def fit_transform(self, X):
        res = self.mms.fit_transform(X)
        self.min_ = torch.tensor(self.mms.min_)
        self.scale_ = torch.tensor(self.mms.scale_)
        return res
    def transform(self, X):
        return minmax_scale_transform_torch(X, self.min_, self.scale_)

class PowerTransformer():
    def __init__(self):
        self.pt = PT()
        self.lambdas_ = None
    def fit_transform(self, X):
        res = self.pt.fit_transform(X)
        self.lambdas_ = torch.tensor(self.pt.lambdas_)
        return res
    def transform(self, X):
        return standard_scale_transform_torch(
            yeo_johnson_transform_torch(X, self.lambdas_),
            torch.tensor(self.pt._scaler.mean_),
            torch.tensor(self.pt._scaler.scale_)
        )

def standard_scale_transform_torch(X, mean_, std_):
    return (X - mean_.to(device=X.device, dtype=X.dtype)) / std_.to(device=X.device, dtype=X.dtype)

def minmax_scale_transform_torch(X, min_, scale_):
    return (X - min_.to(device=X.device, dtype=X.dtype)) * scale_.to(device=X.device, dtype=X.dtype)


def yeo_johnson_transform_torch(X, lambdas):
    """
    Applies the Yeo-Johnson transformation to a PyTorch tensor.
    
    Parameters:
    X (torch.Tensor): The data to be transformed.
    lambdas (torch.Tensor or ndarray): The lambda parameters from the fitted sklearn PowerTransformer.
    
    Returns:
    torch.Tensor: The transformed data.
    """
    lambdas = lambdas.to(device=X.device, dtype=X.dtype)
    X_transformed = torch.zeros_like(X, device=X.device, dtype=X.dtype)
    
    # Define two masks for the conditional operation
    positive = X >= 0
    negative = X < 0

    # Applying the Yeo-Johnson transformation
    # For positive values
    pos_transform = torch.where(
        lambdas != 0,
        torch.pow(X[positive] + 1, lambdas) - 1,
        torch.log(X[positive] + 1)
    ) / lambdas

    # For negative values (only if lambda != 2)
    neg_transform = torch.where(
        lambdas != 2,
        -(torch.pow(-X[negative] + 1, 2 - lambdas) - 1) / (2 - lambdas),
        -torch.log(-X[negative] + 1)
    )

    # Assigning the transformed values back to the tensor
    X_transformed[positive] = pos_transform
    X_transformed[negative] = neg_transform

    return X_transformed
