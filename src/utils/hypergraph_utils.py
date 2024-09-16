from torch_geometric.utils import to_undirected
import dhg 
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import BaseTransform

import torch_geometric
import tqdm

from functools import partial

from multiprocessing import Pool

import torch.multiprocessing as mp


import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch_geometric.data.hypergraph_data import HyperGraphData
from torch_geometric.data import Dataset
from dhg import Graph, Hypergraph
from tqdm import tqdm

def data_to_hg(data, add_k_hop = 0, min_k_hop_size = 0):
    edge_index_undirected = to_undirected(data.edge_index)
    unique_edges = edge_index_undirected.t().tolist()

    edges_as_tuples = [(int(edge[0]), int(edge[1])) for edge in unique_edges]
    if add_k_hop:
        g = dhg.Graph(data.num_nodes, edges_as_tuples)
        hg_k_hop = dhg.Hypergraph.from_graph_kHop(g, k = add_k_hop)
        hyperedges = hg_k_hop.e 
        if min_k_hop_size > 0:
            hyperedges_filtered = [edge for edge in hyperedges[0] if len(edge) > min_k_hop_size]
            edges_as_tuples = edges_as_tuples + hyperedges_filtered
        else:
            edges_as_tuples = edges_as_tuples + hyperedges[0]
    hg = dhg.Hypergraph(data.num_nodes, edges_as_tuples)
    return hg

def get_clique_from_node( node_idx, graph ):
    subnodes, subgraph_edges, _, _ = torch_geometric.utils.k_hop_subgraph(node_idx = node_idx, num_hops = 1, edge_index = graph.edge_index)
    periph_nodes = subnodes[subnodes != node_idx]
    periph_edge_index, _ = torch_geometric.utils.subgraph(periph_nodes, subgraph_edges)
    mask = periph_edge_index[0] < periph_edge_index[1]
    periph_edge_index = periph_edge_index[:, mask]
    
    three_cliques = torch.cat([periph_edge_index, torch.ones((1,periph_edge_index.shape[1]), dtype = torch.long) * node_idx], dim = 0)

    four_cliques = []
    #finding 4 cliques - just triangles in the peripherical nodes
    if len(periph_nodes) >= 3:
        for edge_ in periph_edge_index.T:
            for node in periph_nodes:
                if node in edge_:
                    continue # the node is already in the edge, not a triangle.
                mask_1 = (periph_edge_index==node) + (periph_edge_index == edge_[0])
                mask_2 = (periph_edge_index==node) + (periph_edge_index == edge_[1]) 
                hyperedge_found = mask_1.all(0).any() * mask_2.all(0).any()
                if hyperedge_found:
                    four_cliques.append(torch.cat([edge_, torch.tensor([node]), torch.tensor([node_idx])])[None,:])
    
    if len(four_cliques)>0:
        four_cliques = torch.cat(four_cliques).T
    
    return (three_cliques, four_cliques)

def get_cliques_planar_(graph, njobs = 5):
    """
    Computes the 3 and 4-cliques from a torch geometric graph.

    Input: 
        - torch geometric graph
    Output:
        Same object with attributes "three_cliques" and "four_cliques"
        - torch tensor of shape (3, num_3_cliques) containing the 3-cliques
        - torch tensor of shape (4, num_4_cliques) containing the 4-cliques
    """
    four_cliques_list = []
    three_cliques_list = []

    f = partial(get_clique_from_node, graph = graph)

    with Pool(njobs) as p:
        res = p.map(f,[i for i in range(graph.x.shape[0])])

    three_cliques_comb = torch.cat([r[0] for r in res], dim = 1)
    four_cliques_comb = torch.cat([r[1] for r in res if len(r[1])>0], dim = 1)

    three_cliques_sorted = torch.sort(three_cliques_comb,0)[0]
    four_cliques_sorted = torch.sort(four_cliques_comb,0)[0]

    three_cliques = torch.unique(three_cliques_sorted, sorted = False, dim = 1)
    four_cliques = torch.unique(four_cliques_sorted, sorted = False, dim = 1)

    graph.three_cliques = three_cliques
    graph.four_cliques = four_cliques

    return graph

def get_cliques_planar(graph, njobs = 1):
    """
    Computes the 3 and 4-cliques from a torch geometric graph.

    Input: 
        - torch geometric graph
    Output:
        Same object with attributes "three_cliques" and "four_cliques"
        - torch tensor of shape (3, num_3_cliques) containing the 3-cliques
        - torch tensor of shape (4, num_4_cliques) containing the 4-cliques
    """
    four_cliques_list = []
    three_cliques_list = []


    for node_idx in tqdm.tqdm(range(graph.x.shape[0])):

        subnodes, subgraph_edges, _, _ = torch_geometric.utils.k_hop_subgraph(node_idx = node_idx, num_hops = 1, edge_index = graph.edge_index)
        periph_nodes = subnodes[subnodes != node_idx]
        periph_edge_index, _ = torch_geometric.utils.subgraph(periph_nodes, subgraph_edges)
        mask = periph_edge_index[0] < periph_edge_index[1]
        periph_edge_index = periph_edge_index[:, mask]
        
        three_cliques = torch.cat([periph_edge_index, torch.ones((1,periph_edge_index.shape[1]), dtype = torch.long) * node_idx], dim = 0)
        three_cliques_list.append(three_cliques)

        four_cliques = []
        #finding 4 cliques - just triangles in the peripherical nodes
        
        if len(periph_nodes) >= 3:
            for edge_ in periph_edge_index.T:
                for node in periph_nodes:
                    if node in edge_:
                        continue # the node is already in the edge, not a triangle.
                    mask_1 = (periph_edge_index==node) + (periph_edge_index == edge_[0])
                    mask_2 = (periph_edge_index==node) + (periph_edge_index == edge_[1]) 
                    hyperedge_found = mask_1.all(0).any() * mask_2.all(0).any()
                    if hyperedge_found:
                        #breakpoint()
                        four_cliques.append(torch.cat([edge_, torch.tensor([node]), torch.tensor([node_idx])])[None,:])
        
        if len(four_cliques)>0:
            four_cliques = torch.cat(four_cliques).T
            four_cliques_list.append(four_cliques)

    three_cliques = torch.sort(torch.cat(three_cliques_list, dim = 1),0)[0]
    four_cliques = torch.sort(torch.cat(four_cliques_list, dim = 1))[0]

    three_cliques = torch.unique(three_cliques, sorted = False, dim = 1)
    four_cliques = torch.unique(four_cliques, sorted = False, dim = 1)

    graph.three_cliques = three_cliques
    graph.four_cliques = four_cliques

    return graph

class CliqueHyperEdgeTransform(BaseTransform):
    def __init__(self):
        super().__init__()
    
    def forward(self,data):
        return get_cliques_planar(data)
    
    def __repr__(self):
        return f"CliqueHyperEdgeTransform"

if __name__ == '__main__': 
    # visualize converting an ER graph into a hypergraph with the desired features

    # Create a random graph using NetworkX
    G = nx.fast_gnp_random_graph(10, 0.3)  # Generate a random graph with 10 nodes and edge probability 0.3

    # Visualize the generated graph (optional)
    nx.draw(G, with_labels=True)
    plt.show()

    # Convert NetworkX graph to PyTorch Geometric data object
    data = from_networkx(G)

    hg = data_to_hg(data, add_k_hop=1, min_k_hop_size = 3)
    hg.draw()


def get_hyperedge_index(HG):
    """
    Get the hyperedge index from a hypergraph object. for the HyperGraphData class.
    
    Args:
        HG: Hypergraph object
    """
    hyperedge_list = HG.e[0]
    # Flatten the list of tuples and also create a corresponding index list
    flattened_list = []
    index_list = []
    for i, t in enumerate(hyperedge_list):
        flattened_list.extend(t)
        index_list.extend([i] * len(t))

    # Convert to 2D numpy array
    hyperedge_index = torch.tensor([flattened_list, index_list])

    return hyperedge_index

def get_HyperGraphData(HG, node_features, hyperedge_attr, labels, other_data=None):
    """
    Get the HyperGraphData class from a hypergraph object and the corresponding node features, hyperedge attributes and labels.
    
    Args:
        HG: Hypergraph object
        node_features (torch.Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        hyperedge_attr (torch.Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`.
            (default: :obj:`None`)
        labels (torch.Tensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
        other_data (dict, optional): Dictionary of additional data. (default: :obj:`None`)
    
    Returns:
        a HyperGraphData object
    """
    hyperedge_index = get_hyperedge_index(HG)
    # disregard edge_attr for the time being
    # should be edge_attr = hyperedge_attr, but I'm setting it to none for now
    data = HyperGraphData(x=node_features, edge_index=hyperedge_index, edge_attr=hyperedge_attr, y=labels)
    if other_data is not None:
        for key in other_data.keys():
            data[key] = other_data[key]
            if key == 'graph_y' and labels is None:
                data['y'] = other_data[key]
    return data

def get_HG_data_list(original_dataset, to_hg_func=lambda g: Hypergraph.from_graph_kHop(g, k=1)):
    hgdataset = []
    for graph_dat in tqdm(original_dataset, desc='Converting to hypergraph data'):
        edge_list = graph_dat.edge_index.t() if 'edge_index' in graph_dat.keys() else None
        num_vertices = graph_dat.num_nodes # if 'num_nodes' in graph_dat.keys() else None
        node_features = graph_dat.x if 'x' in graph_dat.keys() else None
        labels = graph_dat.y if 'y' in graph_dat.keys() else None

        G = Graph(num_vertices, edge_list)
        HG1 = to_hg_func(G)

        # Extract all keys other than 'edge_index', 'num_nodes', 'x', 'y'
        other_keys = [key for key in graph_dat.keys() if key not in ['edge_index', 'num_nodes', 'x', 'y', 'edge_attr']]
        other_data = {key: graph_dat[key] for key in other_keys}
        
        X, lbl = node_features, labels
        Y = torch.zeros(HG1.num_e, X.shape[1]) # use all zero hyperedge attributes
        hgdataset.append(get_HyperGraphData(HG1, X, Y, lbl, other_data))
        #import pdb; pdb.set_trace()
    return hgdataset

class HGDatasetFromHGList(Dataset):
    def __init__(self, HG_list, node_features, hyperedge_attrs, labels, other_data=None, transform=None, pre_transform=None):
        super(HGDatasetFromHGList, self).__init__('.', transform, pre_transform)
        self.data_list = []
        for HG, node_feature, hyperedge_attr, label in zip(HG_list, node_features, hyperedge_attrs, labels):
            # subtract 1 from the label so the counts start at zero
            self.data_list.append(get_HyperGraphData(HG, node_feature, hyperedge_attr, torch.tensor(label - 1).unsqueeze(0), other_data))

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

class HGDataset(Dataset):
    def __init__(self, original_dataset, to_hg_func=lambda g: Hypergraph.from_graph_kHop(g, k=1), transform=None, pre_transform=None):
        super(HGDataset, self).__init__('.', transform, pre_transform)
        self.original_dataset = original_dataset
        self.to_hg_func = to_hg_func
        self.data_list = get_HG_data_list(original_dataset, to_hg_func)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

class HGDatasetFromDGL(Dataset):
    def __init__(self, HG, X, Y, lbl, transform=None, pre_transform=None):
        super(HGDatasetFromDGL, self).__init__('.', transform, pre_transform)
        hgdata = get_HyperGraphData(HG, X, Y, lbl)
        self.data_list = [hgdata]

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
