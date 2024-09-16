from torch_geometric.transforms import KNNGraph
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

def get_graph_data(adata, k=5, loop=True): # adding self-loops by default
    coordinates = adata.obsm['spatial']
    data = Data(pos=torch.tensor(coordinates, dtype=torch.float32))
    transform = KNNGraph(k, loop=loop)
    data = transform(data)
    return data

def plot_graph(pos, edge_index, cell_type, sample, ax=None):
    pos = pos.numpy()
    edge_index = edge_index.numpy()
    cell_type = cell_type.numpy()

    # Create a graph
    G = nx.Graph()

    # Add nodes with positions
    for i in range(pos.shape[0]):
        G.add_node(i, pos=pos[i], cell_type=cell_type[i])

    # Add edges
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    # Extract positions
    node_pos = nx.get_node_attributes(G, 'pos')

    # Use provided Axes or create new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Draw nodes
    nx.draw_networkx_nodes(G, node_pos, node_color=cell_type, cmap=plt.get_cmap('viridis'), node_size=1, ax=ax)

    # Draw edges
    nx.draw_networkx_edges(G, node_pos, alpha=0.3, ax=ax)

    ax.set_xlabel('Spatial coordinate 1')
    ax.set_ylabel('Spatial coordinate 2')
    ax.set_title('Sample: ' + sample)
    # plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Cell Type')

    if ax is None:
        plt.show()
