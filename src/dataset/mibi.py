from typing import List
import logging
import os
import anndata as ad
import scanpy as sc
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.data.hypergraph_data import HyperGraphData
from sklearn.neighbors import kneighbors_graph
# from dhg import Graph, Hypergraph
from natsort import natsorted


logging.getLogger('pysmiles').setLevel(logging.CRITICAL)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.hypergraphs.hypergraph_utils import data_to_hg
from src.graphs.builder import return_graph_data


class MIBIDataset(Dataset):
    '''
    MIBI Dataset.
    Spatial RNA-seq data from MIBI.
    Data are given in matrices of matrices of [cell, protein].
    The purpose is to classify these neighborhoods into 4 classes:
        - PD
        - SD
        - PR
        - CR

    NOTE: Since we want to perform data split at the subject level,
    here we use 2 separate classes, MIBIDataset and MIBISubset.
    We will have one MIBISubset instance per split (train/val/test).
    '''

    def __init__(self,
                 data_folder: str = 'data/MIBI/patchified_all_genes',
                 k_hop: int = 3,
                 transform=None):

        self._load_data(data_folder)
        self.k_hop = k_hop
        self.transform = transform
        self.gene_list = self._read_gene_list()

    def _load_data(self, data_folder: str) -> None:
        graph_path_list = natsorted(glob(os.path.join(data_folder, '*.h5ad')))
        self.class_map = {
            0: 'PD',
            1: 'SD',
            2: 'PR',
            3: 'CR',
        }
        self.num_classes = 4

        unique_subject_ids = []
        for graph_path in graph_path_list:
            subject_id = os.path.basename(graph_path).split('-')[0]
            assert subject_id[:8] == 'patient_'
            unique_subject_ids.append(subject_id)
        unique_subject_ids = natsorted(np.unique(unique_subject_ids))
        self.graph_path_by_subject = [[] for _ in range(len(unique_subject_ids))]
        self.class_by_subject = [[] for _ in range(len(unique_subject_ids))]
        for graph_path in graph_path_list:
            subject_id = os.path.basename(graph_path).split('-')[0]
            assert subject_id[:8] == 'patient_'
            graph_str = os.path.basename(graph_path)
            if 'responseM_PD' in graph_str:
                graph_class = 0
            elif 'responseM_SD' in graph_str:
                graph_class = 1
            elif 'responseM_PR' in graph_str:
                graph_class = 2
            elif 'responseM_CR' in graph_str:
                graph_class = 3
            else:
                raise ValueError(f'`graph_str` must contain responseM_`PD`, `SD`, `PR` or `CR`, but got {graph_str}.')
            subject_id_idx = np.argwhere(np.array(unique_subject_ids) == subject_id).item()
            self.graph_path_by_subject[subject_id_idx].append(graph_path)
            self.class_by_subject[subject_id_idx].append(graph_class)

        self.graph_path_arr = np.array(graph_path_list)  # Only for `_read_gene_list` purpose.
        return

    def _read_gene_list(self) -> List:
        adata = ad.read_h5ad(self.graph_path_arr[0])
        return adata.var.to_numpy().flatten().tolist()

    def __len__(self) -> int:
        return len(self.graph_path_by_subject)

    def __getitem__(self, idx: int) -> Data:
        raise NotImplementedError()


class MIBISubset(MIBIDataset):
    '''
    MIBI SubSet.

    NOTE: Since we want to perform data split at the subject level,
    here we use 2 separate classes, MIBIDataset and MIBISubset.
    We will have one MIBISubset instance per split (train/val/test).
    '''

    def __init__(self,
                 dataset: MIBIDataset = None,
                 subset_indices: List[int] = None):

        super().__init__()
        self.dataset = dataset
        self.k_hop = dataset.k_hop
        self.transform = dataset.transform
        graph_path_by_subject = [
            dataset.graph_path_by_subject[i] for i in subset_indices
        ]
        class_by_subject = [
            dataset.class_by_subject[i] for i in subset_indices
        ]

        self.graph_path_arr = np.array([item for sublist in graph_path_by_subject for item in sublist])
        self.class_arr = np.array([item for sublist in class_by_subject for item in sublist])
        assert len(self.graph_path_arr) == len(self.class_arr)

    def __len__(self) -> int:
        return len(self.graph_path_arr)

    def __getitem__(self, idx: int) -> Data:
        raise NotImplementedError()


class MIBISubsetHypergraph(MIBISubset):
    '''
    MIBI Subset in Hypergraph format.
    Spatial RNA-seq data from MIBI.

    Returned `graph_data`: a torch_geometric.data.Data instance, where
                           graph_data.x is the node features.
    '''

    def __getitem__(self, idx: int) -> Data:
        adata = ad.read_h5ad(self.graph_path_arr[idx])
        graph_data = return_graph_data(adata)
        y_true = self.class_arr[idx]
        graph_data.y = y_true

        if self.transform:
            graph_data = self.transform(graph_data)

        hypergraph = data_to_hg(graph_data, add_k_hop=self.k_hop)
        
        return hypergraph

if __name__ == '__main__':
    dataset = MIBIDataset()
    dataset_hg = MIBISubsetHypergraph(dataset,subset_indices=[0,1])
    print(dataset_hg[0])