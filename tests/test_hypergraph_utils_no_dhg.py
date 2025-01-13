import pytest
import torch
from torch_geometric.data import Data
from src.utils.hypergraph_utils_no_dhg import data_to_hg, get_hyperedge_index_from_edges  # Import the actual functions

@pytest.fixture
def sample_data():
    """Fixture to provide sample graph data."""
    return Data(
        x=torch.rand((5, 3)),  # 5 nodes with 3 features each
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]], dtype=torch.long)
    )

def test_data_to_hg_no_k_hop(sample_data):
    """Test data_to_hg without k-hop addition."""
    hg = data_to_hg(sample_data, add_k_hop=0)
    assert hg.edge_index.shape[1] == sample_data.edge_index.shape[1]

def test_data_to_hg_with_k_hop(sample_data):
    """Test data_to_hg with 1-hop addition."""
    hg = data_to_hg(sample_data, add_k_hop=1)
    assert hg.edge_index.shape[1] > sample_data.edge_index.shape[1]

def test_get_hyperedge_index_from_edges():
    """Test hyperedge index creation."""
    hyperedges = [[0, 1, 2], [3, 4]]
    hyperedge_index = get_hyperedge_index_from_edges(hyperedges)
    assert hyperedge_index.shape == (2, 5)
    assert hyperedge_index.tolist() == [[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]]
