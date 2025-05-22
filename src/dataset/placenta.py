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
                 transform=None):

        self._load_data(data_folder)
        self.k_hop = k_hop
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
        graph_data = return_graph_data(adata)
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
    # Normalize the gene expression for each pixel.
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    # Create the graph.
    G = create_knn_graph(adata)

    # NetworkX to PyG.
    data = from_networkx(G)
    data.x = torch.tensor(adata.X.todense(), dtype=torch.float)
    return data

def create_knn_graph(adata, K: int = 10):
    sparseA = kneighbors_graph(adata.obsm['spatial'], n_neighbors=K, mode='connectivity', include_self=False)
    A = sparseA.todense()
    G = nx.from_numpy_array(A)
    return G


if __name__ == '__main__':
    dataset = PlacentaDataset()
    dataset = PlacentaDatasetHypergraph()
    item = dataset[0]
