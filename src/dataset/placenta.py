from typing import List
import logging
import os
import anndata as ad
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch_geometric.data import Data

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.hypergraphs.hypergraph_utils import spatial_graph_to_hypergraph
from src.graphs.builder import return_spatial_graph_data

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class PlacentaDataset(Dataset):
    '''
    Placenta Dataset.
    Spatial RNA-seq data on placenta.
    Data are given in matrices of matrices of [pixel coordinates, gene expression].
    We have chopped up the data into small neighborhoods.
    The purpose is to classify these neighborhoods into 3 classes:
        - Normal
        - Placenta Accreta Spectrum (PAS)
        - Placental Insufficiency

    Returned `graph_data`: a torch_geometric.data.Data instance, where
                           graph_data.x is the node features.
    '''

    def __init__(self,
                 data_folder: str = '../../data/spatial_placenta_accreta/patchified_all_genes',
                 k_hop: int = 3,
                 hyperedge_features_list: List[str] = ['gene_expression'],
                 transform=None):

        self._load_data(data_folder)
        self.k_hop = k_hop
        self.hyperedge_features_list = hyperedge_features_list
        self.transform = transform
        self.gene_list = self._read_gene_list()

    def _load_data(self, data_folder: str) -> None:
        graph_path_list = sorted(glob(os.path.join(data_folder, '*.h5ad')))
        class_list = []
        self.class_map = {
            0: 'normal',
            1: 'PAS',
            2: 'insufficient',
        }
        self.num_classes = 3

        for graph_path in graph_path_list:
            graph_str = os.path.basename(graph_path)
            if 'normal' in graph_str:
                class_list.append(0)
            elif 'PAS' in graph_str:
                class_list.append(1)
            elif 'insufficient' in graph_str:
                class_list.append(2)
            else:
                raise ValueError(f'graph_str must contain `normal`, `PAS` or `insufficient`, but got {graph_str}.')

        assert len(graph_path_list) == len(class_list)

        self.graph_path_arr = np.array(graph_path_list)
        self.class_arr = np.array(class_list)
        return

    def _read_gene_list(self) -> List:
        adata = ad.read_h5ad(self.graph_path_arr[0])
        return adata.var.to_numpy().flatten().tolist()

    def __len__(self) -> int:
        return len(self.graph_path_arr)

    def __getitem__(self, idx: int) -> Data:
        adata = ad.read_h5ad(self.graph_path_arr[idx])
        graph_data = return_spatial_graph_data(adata, mode='knn')
        y_true = self.class_arr[idx]
        graph_data.y = y_true

        if self.transform:
            graph_data = self.transform(graph_data)
        return graph_data

class PlacentaDatasetHypergraph(PlacentaDataset):
    '''
    Placenta Dataset in Hypergraph format.
    Spatial RNA-seq data on placenta.
    Data are given in matrices of matrices of [pixel coordinates, gene expression].
    We have chopped up the data into small neighborhoods.
    The purpose is to classify these neighborhoods into 3 classes:
        - Normal
        - Placenta Accreta Spectrum (PAS)
        - Placental Insufficiency

    Returned `graph_data`: a torch_geometric.data.Data instance, where
                           graph_data.x is the node features.
    '''

    def __getitem__(self, idx: int) -> Data:
        adata = ad.read_h5ad(self.graph_path_arr[idx])
        graph_data = return_spatial_graph_data(adata, mode='knn')
        y_true = self.class_arr[idx]
        graph_data.y = y_true

        '''
        edge_list = graph_data.edge_index.t() if 'edge_index' in graph_data.keys() else None
        num_vertices = graph_data.num_nodes
        node_features = graph_data.x
        labels = graph_data.y
        '''

        if self.transform:
            graph_data = self.transform(graph_data)

        hypergraph = spatial_graph_to_hypergraph(graph_data=graph_data,
                                                 adata=adata,
                                                 hyperedge_features_list=self.hyperedge_features_list,
                                                 add_k_hop=self.k_hop)

        return hypergraph


if __name__ == '__main__':
    dataset = PlacentaDataset()
    dataset = PlacentaDatasetHypergraph()
    item = dataset[0]
