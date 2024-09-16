import torch
from torch_scatter import scatter_add
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

def count_cell_types(edge_index, cell_type, num_cell_types):
    x = torch.nn.functional.one_hot(cell_type, num_classes=num_cell_types).float()
    source, target = edge_index
    counts = scatter_add(x[source], target, dim=0, dim_size=x.size(0))
    percentages = counts / counts.sum(dim=1, keepdim=True)
    return counts, percentages

def compute_mean(x, edge_index):
    row, col = edge_index
    mean_features = scatter_mean(x[row], col, dim=0)
    return mean_features

def compute_var(x, edge_index):
    row, col = edge_index
    mean_features = scatter_mean(x[row], col, dim=0)
    mean_sq = scatter_mean((x*x)[row], col, dim=0)
    var_features = mean_sq - mean_features * mean_features
    return var_features

def compute_mean_var(x, edge_index):
    mean_features = compute_mean(x, edge_index)
    var_features = compute_var(x, edge_index)
    return torch.cat([mean_features, var_features], dim=1)


class LigandReceptorLayer(MessagePassing):
    def __init__(self, n_pairs, self_loop=False, normalize=True, aggr='add'):
        super(LigandReceptorLayer, self).__init__(aggr=aggr)  # Using 'add' aggregation
        self.n_pairs = n_pairs  # Number of ligand-receptor pairs
        self.self_loop = self_loop  # Whether to add self-loops
        self.normalize = normalize  # Whether to normalize the messages

    def forward(self, x, edge_index):
        if self.self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if self.normalize:
            # Compute normalization
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = None
        # Assert the correct size for features
        assert x.size(1) == self.n_pairs * 2

        # Start propagating messages
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, x_i, norm):
        # Extract ligands and receptors
        ligands = x_j[:, :self.n_pairs]  # ligands from source nodes
        receptors = x_i[:, self.n_pairs:]  # receptors from target nodes

        # Element-wise multiplication of corresponding ligands and receptors
        interactions = ligands * receptors  # This performs pairwise multiplication directly

        # Multiply by norm, which is currently just ones and does not change the results
        if norm is not None:
            interactions = interactions * norm.view(-1, 1)

        # Sum over all interactions for each node pair to aggregate the contributions
        return interactions
    
def compute_lr_features(x_lr, edge_index, n_pairs, self_loop=False, normalize=False):
    layer = LigandReceptorLayer(n_pairs=n_pairs, self_loop=self_loop, normalize=normalize, aggr='mean')
    lr_features = layer(x_lr, edge_index)
    return lr_features

