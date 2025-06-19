from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

import torch
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.data.hypergraph_data import HyperGraphData
from torch_geometric.data import Dataset

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_undirected, k_hop_subgraph


def data_to_hg(data, add_k_hop=1):
    edge_index_undirected = to_undirected(data.edge_index)
    hyperedges = [] 
    

    for node_idx in range(data.num_nodes):
        # first check if node_idx is in the graph. For some reason the mismatch appears to be very large!
        if node_idx not in edge_index_undirected[0]:
            print(f'Node {node_idx} not in graph, skipping.")')
            continue

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx, num_hops=add_k_hop, edge_index=edge_index_undirected, relabel_nodes=False
        )
        hyperedges.append(subset.tolist())

    hyperedge_index = get_hyperedge_index_from_edges(hyperedges)

    #TODO: Understand Hyperedge node features 
    #Currently just setting as zero and using the num_features the same of nodes.
    num_hyperedges = len(hyperedges)

    hyperedge_attr = torch.zeros(num_hyperedges, data.x.shape[1]) # use all zero hyperedge attributes
    
    return HyperGraphData(x=data.x, edge_index=hyperedge_index, edge_attr=hyperedge_attr, y=data.y)

def get_hyperedge_index_from_edges(hyperedges):
    """
    Convert a list of hyperedges to a hyperedge index tensor.
    
    Args:
        hyperedges (list of list of int): List of hyperedges where each hyperedge is a list of node indices.
    
    Returns:
        torch.Tensor: Hyperedge index tensor.
    """
    flattened_list = []
    index_list = []
    for i, hyperedge in enumerate(hyperedges):
        flattened_list.extend(hyperedge)
        index_list.extend([i] * len(hyperedge))
    
    return torch.tensor([flattened_list, index_list], dtype=torch.long)

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


    for node_idx in tqdm(range(graph.x.shape[0])):

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

# Remaining functions remain the same with minor modifications if necessary
def get_HyperGraphData(HG, node_features, hyperedge_attr, labels, other_data=None):
    """
    Modified to use the new hyperedge index tensor.
    """
    data = HyperGraphData(x=node_features, edge_index=HG.edge_index, edge_attr=hyperedge_attr, y=labels)
    if other_data is not None:
        for key in other_data.keys():
            data[key] = other_data[key]
            if key == 'graph_y' and labels is None:
                data['y'] = other_data[key]
    return data

def get_HG_data_list(original_dataset, to_hg_func=data_to_hg):
    hgdataset = []
    for graph_dat in tqdm(original_dataset, desc='Converting to hypergraph data'):
        hgdataset.append(to_hg_func(graph_dat))
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
    def __init__(self, original_dataset, to_hg_func, transform=None, pre_transform=None):
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


if __name__ == "__main__":
    def sample_data_func():
        """Fixture to provide sample graph data."""
        # this is a graph with 5 nodes and 5 edges
        # (0, 1), (0, 3), (0, 4), (1, 2), (2, 3)
        return Data(
            x=torch.rand((5, 3)),  # 5 nodes with 3 features each
            edge_index=torch.tensor([[0, 0, 0, 1, 2,], [1, 3, 4, 2, 3]], dtype=torch.long)
        )

    sample_data = sample_data_func()

    def test_data_to_hg_with_k_hop(sample_data):
        """Test data_to_hg with 1-hop addition."""
        hg = data_to_hg(sample_data, add_k_hop=1)
        assert hg.edge_index.shape[1] > sample_data.edge_index.shape[1]
        print(hg.edge_index)
        assert (hg.edge_index[1,:] == 0).sum() == 4 # 0's k hop neighborhood has 4 elements
        assert (hg.edge_index[1,:] == 1).sum() == 3
        assert (hg.edge_index[1,:] == 2).sum() == 3
        assert (hg.edge_index[1,:] == 3).sum() == 3
        assert (hg.edge_index[1,:] == 4).sum() == 2

    def test_get_hyperedge_index_from_edges():
        """Test hyperedge index creation."""
        hyperedges = [[0, 1, 2], [3, 4]]
        hyperedge_index = get_hyperedge_index_from_edges(hyperedges)
        assert hyperedge_index.shape == (2, 5)
        assert hyperedge_index.tolist() == [[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]]

    def test_create_HGDataset():
        """Test HGDataset creation."""
        dataset = [sample_data, sample_data]
        hg_dataset = HGDataset(dataset, data_to_hg)
        assert len(hg_dataset) == 2
        assert isinstance(hg_dataset.get(0), HyperGraphData)
        # print the first hypergraph's edge index
        print(hg_dataset.get(0).edge_index)

    test_data_to_hg_with_k_hop(sample_data)
    test_create_HGDataset()
    
