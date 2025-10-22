import pytest
import torch
from torch_geometric.data import Data
from src.hypergraphs.hypergraph_utils import spatial_graph_to_hypergraph, get_hyperedge_index_from_edges  # Import the actual functions

import sys
import os

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


@pytest.fixture
def sample_data():
    """Fixture to provide sample graph data."""
    # this is a graph with 5 nodes and 5 edges
    # (0, 1), (0, 3), (0, 4), (1, 2), (2, 3)
    return Data(
        x=torch.rand((5, 3)),  # 5 nodes with 3 features each
        edge_index=torch.tensor([[0, 0, 0, 1, 2,], [1, 3, 4, 2, 3]], dtype=torch.long)
    )

def test_spatial_graph_to_hypergraph_no_k_hop(sample_data):
    """Test spatial_graph_to_hypergraph without k-hop addition."""
    hg = spatial_graph_to_hypergraph(sample_data, add_k_hop=0)
    assert hg.edge_index.shape[1] == sample_data.edge_index.shape[1]

def test_spatial_graph_to_hypergraph_with_k_hop(sample_data):
    """Test spatial_graph_to_hypergraph with 1-hop addition."""
    hg = spatial_graph_to_hypergraph(sample_data, add_k_hop=1)
    assert hg.edge_index.shape[1] > sample_data.edge_index.shape[1]

def test_get_hyperedge_index_from_edges():
    """Test hyperedge index creation."""
    hyperedges = [[0, 1, 2], [3, 4]]
    hyperedge_index = get_hyperedge_index_from_edges(hyperedges)
    assert hyperedge_index.shape == (2, 5)
    assert hyperedge_index.tolist() == [[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]]
