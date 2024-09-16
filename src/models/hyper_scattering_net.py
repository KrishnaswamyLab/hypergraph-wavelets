import dhg 
import torch
import torch.nn as nn
from typing import Tuple, Optional
from einops import rearrange
from torch.nn import Linear 
from torch_geometric.nn.pool import global_mean_pool 
from torch_geometric.nn import GCNConv 
from torch_geometric.nn.norm import BatchNorm

# a good way to make hyper edges:
# dhg.Hypergraph.from_graph_kHop()

class LazyLayer(torch.nn.Module):
    
    """ Currently a single elementwise multiplication with one laziness parameter per
    channel. this is run through a softmax so that this is a real laziness parameter
    """

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
    

class HyperDiffusion(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            trainable_laziness=False,
            fixed_weights=True
    ):
        super().__init__()
        self.trainable_laziness= trainable_laziness 
        self.fixed_weights = fixed_weights 
        # in the future, we could make this time independent, but spatially dependent, as in GRAND
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        # in the future, I'd like to have different weights based on the hypergraph edge size
        if not self.fixed_weights:
            self.lin_self = torch.nn.Linear(in_channels, out_channels)
            self.lin_neigh = torch.nn.Linear(in_channels, out_channels)

    def forward(self, hg: dhg.Hypergraph, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # if not self.fixed_weights:
        #     X = self.lin(X)
        
        # X has shape num_nodes, num_features
        #import pdb; pdb.set_trace()
        # propagate from nodes to hyperedges 
        inv_deg_v = hg.D_v_neg_1.values()
        inv_deg_v = torch.nan_to_num(inv_deg_v)
        # I should degree normalize first
        X_norm = torch.einsum('ij,i->ij', X, inv_deg_v)

        edge_feat = hg.v2e(X_norm, aggr = 'sum')
        
        if not self.fixed_weights:
            edge_feat = self.lin_neigh(edge_feat)
            Y = self.lin_self(edge_feat)

        if self.trainable_laziness and Y is not None:
            edge_feat = self.lazy_layer(edge_feat, Y)

        # propagate back from hyperedges to nodes 
        inv_deg_e = hg.D_e_neg_1.values()
        inv_deg_e = torch.nan_to_num(inv_deg_e)
        edge_feat_norm = torch.einsum('ij,i->ij', edge_feat, inv_deg_e)

        node_feat = hg.e2v(edge_feat_norm, aggr = 'sum')

        if not self.fixed_weights:
            node_feat = self.lin_neigh(node_feat)
            X = self.lin_self(X)
        
        if self.trainable_laziness:
            node_feat = self.lazy_layer(node_feat, X)
        
        return node_feat, edge_feat 

class HyperScatteringModule(nn.Module):
    def __init__(self, in_channels, trainable_laziness = False, trainable_scales = False, activation = "blis", fixed_weights=True):

        super().__init__()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = HyperDiffusion(in_channels, in_channels, trainable_laziness, fixed_weights)
        # self.diffusion_layer2 = Diffuse(
        #     4 * in_channels, 4 * in_channels, trainable_laziness
        # )
        self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
            [1, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], requires_grad=trainable_scales))

        if activation == "blis":
            self.activations = [lambda x: torch.relu(x), lambda x: torch.relu(-x)]
        elif activation == None:
            self.activations = [lambda x : x]
        elif activation == "modulus":
            self.activations = [lambda x: torch.abs(x)]
        elif activation == "leaky_relu":
            m = nn.LeakyReLU()
            self.activations = [lambda x: m(x)]

    def forward(self, hg: dhg.Hypergraph, X: torch.Tensor, Y: torch.Tensor):

        """ This performs  Px with P = 1/2(I + AD^-1) (column stochastic matrix) at the different scales"""

        #x, edge_index = data.x, data.edge_index
        features = X.shape[1]
        #s0 = X[:,:,None]
        node_features = [X]

        edge_features = [Y]
        #import pdb; pdb.set_trace()
        for i in range(16):
            node_feat, edge_feat = self.diffusion_layer1(hg, node_features[-1], edge_features[-1])
            node_features.append(node_feat)
            edge_features.append(edge_feat)
        # for j in range(len(avgs)):
        #     # add an extra dimension to each tensor to avoid data loss while concatenating TODO: is there a faster way to do this?
        #     avgs[j] = avgs[j][None, :, :, :]  
        # Combine the diffusion levels into a single tensor.
        diffusion_levels = rearrange(node_features, 'i j k -> i j k')
        edge_diffusion_levels = rearrange(edge_features, 'i j k -> i j k')
        #edge_diffusion_levels = torch.cat(edge_features)
        
        # Reshape the 3d tensor into a 2d tensor and multiply with the wavelet_constructor matrix
        # This simulates the below subtraction:
        # filter0 = avgs[0] - avgs[1]
        # filter1 = avgs[1] - avgs[2] 
        # filter2 = avgs[2] - avgs[4]
        # filter3 = avgs[4] - avgs[8]
        # filter4 = avgs[8] - avgs[16] 
        # filter5 = avgs[16]

        wavelet_coeffs = torch.einsum("ij,jkl->ikl", self.wavelet_constructor, diffusion_levels) # J x num_nodes x num_features x 1
        wavelet_coeffs_edges = torch.einsum("ij,jkl->ikl", self.wavelet_constructor, edge_diffusion_levels)
        #subtracted = subtracted.view(6, x.shape[0], x.shape[1]) # reshape into given input shape
        activated = [self.activations[i](wavelet_coeffs) for i in range(len(self.activations))]
        activated_edges = [self.activations[i](wavelet_coeffs_edges) for i in range(len(self.activations))]
        s_nodes = rearrange(activated, 'a w n f -> n (w f a)')
        s_edges = rearrange(activated_edges, 'a w e f -> e (w f a)')

        return s_nodes, s_edges
    
    def out_features(self):
        return 6 * self.in_channels * len(self.activations)

class HSN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 trainable_laziness = False, 
                 trainable_scales = False, 
                 activation = "modulus", 
                 fixed_weights=True, 
                 layout = ['hsm','hsm'], 
                 **kwargs):
        
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.trainable_laziness = trainable_laziness 
        self.trainable_scales = trainable_scales 
        self.activation = activation 
        self.fixed_weights = fixed_weights

        self.layout = layout 
        self.layers = []
        self.out_dimensions = [in_channels]

        for layout_ in layout:
            if layout_ == 'hsm':
                self.layers.append(HyperScatteringModule(self.out_dimensions[-1], 
                                                         trainable_laziness = trainable_laziness,
                                                         trainable_scales = self.trainable_scales, 
                                                         activation = self.activation, 
                                                         fixed_weights=self.fixed_weights))
                self.out_dimensions.append(self.layers[-1].out_features() )
            elif layout_ == 'dim_reduction':
                input_dim = self.out_dimensions[-1]
                output_dim = input_dim//2
                self.out_dimensions.append(output_dim)
                self.layers.append(nn.Linear(input_dim, output_dim))
            else:
                raise ValueError("Not yet implemented")
        
        self.layers = nn.ModuleList(self.layers)

        # currently share backend MLPs for the node and edge features
        self.batch_norm = BatchNorm(self.out_dimensions[-1])

        self.fc1 = Linear(self.out_dimensions[-1], self.out_dimensions[-1]//2)
        self.fc2 = nn.Linear(self.out_dimensions[-1]//2, self.out_channels)
        #self.fc3 = nn.Linear(128, 64)
        #self.fc4 = nn.Linear(64, self.out_channels)
        
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(self.out_dimensions[-1]//2)
        #self.batch_norm2 = nn.BatchNorm1d(128)
        #self.batch_norm3 = nn.BatchNorm1d(64)

        self.mlp = nn.Sequential(
            self.fc1,
            self.batch_norm1,
            self.relu,
            self.fc2
        )

        # self.mlp = nn.Sequential(
        #     self.fc1,
        #     self.batch_norm1,
        #     self.relu,
        #     self.fc2,
        #     self.batch_norm2,
        #     self.relu,
        #     self.fc3,
        #     self.batch_norm3,
        #     self.relu,
        #     self.fc4
        # )

        # self.lin1 = Linear(self.out_dimensions[-1], self.out_dimensions[-1]//2 )
        # self.mean = global_mean_pool 
        # self.lin2 = Linear(self.out_dimensions[-1]//2, out_channels)
        # self.lin3 = Linear(out_channels, out_channels)

        # self.act = nn.ReLU()

    def forward(self, hg: dhg.Hypergraph,  X: torch.Tensor, Y: torch.Tensor):
        for il, layer in enumerate(self.layers):
            if self.layout[il] == 'hsm':
                X, Y = layer(hg, X, Y)
            elif self.layout[il] == 'dim_reduction':
                X = layer(X)
                Y = layer(Y) 
            else:
                X, Y = layer(hg, X, Y)
        #import pdb; pdb.set_trace()
        X = self.batch_norm(X)
        X = self.mlp(X)
        # X = self.lin1(X)
        # X = self.act(X)
        # X = self.lin2(X)
        # X = self.act(X)
        # X = self.lin3(X)

        # compute the same process on the edges:
        Y = self.batch_norm(Y)
        Y = self.mlp(Y)
        # Y = self.lin1(Y)
        # Y = self.act(Y)
        # Y = self.lin2(Y)
        # Y = self.act(Y)
        # Y = self.lin3(Y)

        return X,Y

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"training on device {device}")
    num_vertices = 15
    hg = dhg.random.uniform_hypergraph_Gnp(3,num_vertices, .4).to(device)
    signal_features = 2
    X = torch.rand(num_vertices, signal_features).to(device)
    num_edges = hg.num_e
    Y = torch.zeros(num_edges, signal_features).to(device)

    hidden_channels = 16
    out_channels = 1
    net = HSN(signal_features, hidden_channels, 1).to(device)
    #import pdb; pdb.set_trace()

    node_pred, edge_pred = net(hg ,X, Y)
    import pdb; pdb.set_trace()
    node_pred.shape





