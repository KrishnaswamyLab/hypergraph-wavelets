from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import scprep
from torch_geometric.utils.convert import from_networkx
from dhg import Hypergraph
import sys
sys.path.append('..')
from utils.hypergraph_utils import HGDataset
from models.hsn_pyg import HyperScatteringModule

def get_hyperedge_pos_df(hgdataset, coordinates):
    ei = hgdataset.edge_index
    eidf = pd.DataFrame(ei.T.numpy(), columns=['node_idx', 'he_idx'])
    eidf[['x', 'y']] = coordinates[eidf['node_idx'].values, :]
    return eidf

def enlarge_hull(points, hull, factor=0.1):
    centroid = np.mean(points[hull.vertices, :], axis=0)
    new_vertices = []
    for vertex in points[hull.vertices, :]:
        direction = vertex - centroid
        new_vertex = centroid + (1 + factor) * direction
        new_vertices.append(new_vertex)
    return np.array(new_vertices)

def compute_enlarged_hulls(df, factor=0.1):
    enlarged_hulls = []

    for set_index, group in df.groupby('he_idx'):
        points = group[['x', 'y']].values
        hull = ConvexHull(points)
        enlarged_hull = enlarge_hull(points, hull, factor)

        # Store the enlarged hull and the mean value for color coding
        enlarged_hulls.append(enlarged_hull)

    return enlarged_hulls

def plot_hulls(enlarged_hulls, color_values, title='', colormap=plt.cm.viridis, alpha=0.5, ax=None, show_cbar=True):
    if ax is None:
        fig, ax = plt.subplots()

    # Normalize color values
    norm = plt.Normalize(min(color_values), max(color_values))

    for hull, value in zip(enlarged_hulls, color_values):
        color = colormap(norm(value))
        ax.fill(hull[:, 0], hull[:, 1], color=color, alpha=alpha)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    if show_cbar:
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
    # ax.show()
    ax.set_aspect('equal')
    return ax

def plot_wavelets(node_features, edge_features, coordinates, enlarged_hulls, title='', alpha=0.5):
    fig, axes = plt.subplots(2, 6, figsize=(30, 20), dpi=100)

    for i, axs in enumerate(axes.T):

        scprep.plot.scatter2d(coordinates, c=node_features[:,i], ax=axs[0], cmap='viridis')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title(f'node wavelet {i}')
        plot_hulls(enlarged_hulls, edge_features[:,i], colormap=plt.cm.viridis, alpha=alpha, ax=axs[1])
        axs[1].set_title(f'hyperedge wavelet {i}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def get_wv_plots(G, X_data, coordinates, num_hops=1, graph_info='', device='cpu'):
    data = from_networkx(G)
    data.x = torch.tensor(X_data)
    original_dataset = [data]
    to_hg_func = lambda g: Hypergraph.from_graph_kHop(g, num_hops)
    dataset = HGDataset(original_dataset, to_hg_func)
    eidf = get_hyperedge_pos_df(dataset[0], coordinates)
    enlarged_hulls = compute_enlarged_hulls(eidf)
    model = HyperScatteringModule(in_channels=1, # doesn't matter here.
              trainable_laziness = False,
              trainable_scales = False, 
              activation = None, # just get one layer of wavelet transform 
              fixed_weights=True, 
              normalize='right', 
              reshape=False,
        ).to(device)
    init_node_sig = torch.from_numpy(X_data[:, 1].reshape(-1, 1)).to(device)
    init_he_sig = torch.zeros(dataset[0].edge_attr.shape[0], 1, device=device)
    heidx = dataset[0].edge_index.to(device)
    s_nodes, s_edges = model(init_node_sig, heidx, hyperedge_attr = init_he_sig)
    node_feats = s_nodes[0,:,:,0].detach().cpu().numpy().T
    edge_feats = s_edges[0,:,:,0].detach().cpu().numpy().T
    plot_wavelets(node_feats, edge_feats, coordinates, enlarged_hulls, title=f'Marker, {graph_info}, n_hops={num_hops}', alpha=0.2)
    plt.show()
    scprep.plot.scatter2d(coordinates, c=init_node_sig.cpu().numpy(), cmap='viridis')
    plt.show()
    np.random.seed(32)
    one_pos = np.random.choice(X_data.shape[0], 10)
    dirac_sig = torch.zeros((X_data.shape[0], 1))
    dirac_sig[one_pos, :] = 1
    init_node_sig = dirac_sig.to(device)
    init_he_sig = torch.zeros(dataset[0].edge_attr.shape[0], 1, device=device)
    heidx = dataset[0].edge_index.to(device)
    s_nodes, s_edges = model(init_node_sig, heidx, hyperedge_attr = init_he_sig)
    node_feats = s_nodes[0,:,:,0].detach().cpu().numpy().T
    edge_feats = s_edges[0,:,:,0].detach().cpu().numpy().T
    plot_wavelets(node_feats, edge_feats, coordinates, enlarged_hulls, title=f'Dirac, {graph_info}, n_hops={num_hops}', alpha=0.2)
    plt.show()
    scprep.plot.scatter2d(coordinates, c=init_node_sig.cpu().numpy(), cmap='viridis')
    plt.show()
    return dataset