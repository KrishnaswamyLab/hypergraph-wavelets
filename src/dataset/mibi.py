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
from dhg import Graph, Hypergraph
from natsort import natsorted

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


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
                 data_folder: str = '../../data/MIBI/all_genes',
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

        edge_list = graph_data.edge_index.t() if 'edge_index' in graph_data.keys() else None
        num_vertices = graph_data.num_nodes
        node_features = graph_data.x
        labels = graph_data.y
        graph = Graph(num_vertices, edge_list)
        hypergraph = Hypergraph.from_graph_kHop(graph, k=self.k_hop)

        other_keys = [key for key in graph_data.keys() if key not in ['edge_index', 'num_nodes', 'x', 'y', 'edge_attr']]
        other_data = {key: graph_data[key] for key in other_keys}

        hyperedge_attr = torch.zeros(hypergraph.num_e, node_features.shape[1]) # use all zero hyperedge attributes

        hyperedge_index = get_hyperedge_index(hypergraph)
        # should be edge_attr = hyperedge_attr, but I'm setting it to none for now
        hypergraph_data = HyperGraphData(x=node_features, edge_index=hyperedge_index, edge_attr=hyperedge_attr, y=labels)
        if other_data is not None:
            for key in other_data.keys():
                hypergraph_data[key] = other_data[key]
                if key == 'graph_y' and labels is None:
                    hypergraph_data['y'] = other_data[key]

        return hypergraph_data


def get_hyperedge_index(hypergraph):
    """
    Get the hyperedge index from a hypergraph object. for the HyperGraphData class.

    Args:
        hypergraph: Hypergraph object
    """
    hyperedge_list = hypergraph.e[0]
    # Flatten the list of tuples and also create a corresponding index list
    flattened_list = []
    index_list = []
    for i, t in enumerate(hyperedge_list):
        flattened_list.extend(t)
        index_list.extend([i] * len(t))

    # Convert to 2D numpy array
    hyperedge_index = torch.tensor([flattened_list, index_list])

    return hyperedge_index

def return_graph_data(adata):
    # Normalize the gene expression for each cell.
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    # Create the graph.
    G = create_knn_graph(adata)

    # NetworkX to PyG.
    data = from_networkx(G)
    data.x = torch.tensor(adata.X, dtype=torch.float)
    return data

def create_knn_graph(adata, K: int = 10):
    sparseA = kneighbors_graph(adata.obsm['spatial'], n_neighbors=K, mode='connectivity', include_self=False)
    A = sparseA.todense()
    G = nx.from_numpy_array(A)
    return G


if __name__ == '__main__':
    dataset = MIBIDataset()
