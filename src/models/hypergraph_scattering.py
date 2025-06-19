'''
HSN rewritten with pytorch geometric, can operate on batched hypergraphs.
the data is stored in the format of pytorch geometric.
see https://github.com/pyg-team/pytorch_geometric/blob/cf24b4bcb4e825537ba08d8fc5f31073e2cd84c7/torch_geometric/data/hypergraph_data.py
for example:
    hyperedge_index = torch.tensor([
        [0, 1, 2, 1, 2, 3],
        [0, 0, 0, 1, 1, 1],
    ])
    hyperedge_weight = torch.tensor([1, 1], dtype=torch.float)

modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/hypergraph_conv.html#HypergraphConv
'''

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_scatter import scatter_softmax, scatter_sum


class LazyLayer(torch.nn.Module):

    ''' Currently a single elementwise multiplication with one laziness parameter per
    channel. this is run through a softmax so that this is a real laziness parameter
    '''

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = F.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)


class HyperDiffusion(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            trainable_laziness=False,
            fixed_weights=True,
            normalize="right",
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        assert normalize in ["right", "left", "symmetric"], f"normalize must be one of 'right', 'left', or 'symmetric', not {self.normalize}"

        self.normalize = normalize

        # in the future, we could make this time independent, but spatially dependent, as in GRAND
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        # in the future, I'd like to have different weights based on the hypergraph edge size
        if not self.fixed_weights:
            self.lin_self = torch.nn.Linear(in_channels, out_channels)
            self.lin_neigh = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None,
                hyperedge_attr: Optional[torch.Tensor] = None,
                num_edges: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        # this is the degree of the vertices (taken inverse)
        D_v_inv = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D_v_inv = 1.0 / D_v_inv
        D_v_inv[D_v_inv == float("inf")] = 0
        # this is the degree of the hyperedges (taken inverse)
        D_he_inv = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        D_he_inv = 1.0 / D_he_inv
        D_he_inv[D_he_inv == float("inf")] = 0

        if self.normalize == "left":
            out_edge = self.propagate(hyperedge_index, x=x, norm=D_he_inv,
                                size=(num_nodes, num_edges))
            out_edge = self.laziness_weight_process_edge(out_edge, hyperedge_attr)
            out_node = self.propagate(hyperedge_index.flip([0]), x=out_edge, norm=D_v_inv,
                                size=(num_edges, num_nodes))
            out_node = self.laziness_weight_process_node(out_node, x)
        elif self.normalize == "right":
            out = D_v_inv.view(-1, 1) * x
            out_edge = self.propagate(hyperedge_index, x=out, norm=D_he_inv,
                                      size=(num_nodes, num_edges))
            out_edge = self.laziness_weight_process_edge(out_edge, hyperedge_attr)
            out = D_he_inv.view(-1, 1) * out_edge
            out_node = self.propagate(hyperedge_index.flip([0]), x=out, norm=D_v_inv,
                                      size=(num_edges, num_nodes))
            out_node = self.laziness_weight_process_node(out_node, x)
        elif self.normalize == "symmetric":
            D_v_inv_sqrt = D_v_inv.sqrt()
            out = D_v_inv_sqrt.view(-1, 1) * x
            out_edge = self.propagate(hyperedge_index, x=out, norm=D_he_inv,
                                size=(num_nodes, num_edges))
            out_edge = self.laziness_weight_process_edge(out_edge, hyperedge_attr)
            out_node = self.propagate(hyperedge_index.flip([0]), x=out_edge, norm=D_v_inv_sqrt,
                                size=(num_edges, num_nodes))
            out_node = self.laziness_weight_process_node(out_node, x)
        else:
            raise ValueError(f"normalize must be one of 'right', 'left', or 'symmetric', not {self.normalize}")

        return out_node, out_edge

    def message(self, x_j: torch.Tensor, norm_i: Optional[torch.Tensor] = None) -> torch.Tensor:
        if norm_i is None:
            out = x_j
        else:
            out = norm_i.view(-1, 1) * x_j
        return out

    def laziness_weight_process_edge(self, out_edge, hyperedge_attr):
        if not self.fixed_weights:
            out_edge = self.lin_neigh(out_edge)
            hyperedge_attr = self.lin_self(out_edge)
        if self.trainable_laziness and hyperedge_attr is not None:
            out_edge = self.lazy_layer(out_edge, hyperedge_attr)
        return out_edge

    def laziness_weight_process_node(self, out_node, x):
        if not self.fixed_weights:
            out_node = self.lin_neigh(out_node)
            x = self.lin_self(x)
        if self.trainable_laziness:
            out_node = self.lazy_layer(out_node, x)
        return out_node

class HyperScatteringModule(nn.Module):
    def __init__(self,
                 in_channels,
                 num_features: int = 18085,
                 trainable_laziness: bool = False,
                 trainable_scales: bool = False,
                 fixed_weights: bool = True,
                 normalize: str = "right",
                 reshape: bool = True,
                 scale_list = [0, 1, 2, 4, 8, 16]):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer = HyperDiffusion(in_channels, in_channels, trainable_laziness, fixed_weights, normalize)

        # ensure that scale list is an increasing list of integers with 0 as the first element
        # ensure that 1 is the second element
        assert all(isinstance(x, int) for x in scale_list)
        assert all(scale_list[i] < scale_list[i+1] for i in range(len(scale_list)-1))
        assert scale_list[0] == 0
        assert scale_list[1] == 1

        self.diffusion_levels = scale_list[-1]
        wavelet_matrix = np.zeros((len(scale_list), self.diffusion_levels+1))
        for i in range(len(scale_list) - 1):
            wavelet_matrix[i, scale_list[i]] = 1
            wavelet_matrix[i, scale_list[i+1]] = -1
        wavelet_matrix[-1, -1] = 1
        self.wavelet_constructor = torch.nn.Parameter(torch.from_numpy(wavelet_matrix).float(),
                                                      requires_grad=trainable_scales)

        self.reshape = reshape

    def forward(self,
                x: torch.Tensor,
                hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None,
                hyperedge_attr: Optional[torch.Tensor] = None,
                num_edges: Optional[int] = None):

        node_features = [x]
        edge_features = [hyperedge_attr]

        for _ in range(self.diffusion_levels):
            node_feat, edge_feat = self.diffusion_layer(x=node_features[-1], hyperedge_index=hyperedge_index, hyperedge_weight=hyperedge_weight, hyperedge_attr=edge_features[-1])
            node_features.append(node_feat)
            edge_features.append(edge_feat)

        # Combine the diffusion levels into a single tensor.
        diffusion_levels = rearrange(node_features, 'i j k -> i j k').float()
        edge_diffusion_levels = rearrange(edge_features, 'i j k -> i j k').float()
        node_emb = torch.einsum("ij,jkl->ikl", self.wavelet_constructor, diffusion_levels) # J x num_nodes x num_features x 1
        edge_emb = torch.einsum("ij,jkl->ikl", self.wavelet_constructor, edge_diffusion_levels)
        # [scales, nodes, features] -> [nodes, scales * features]
        node_emb = rearrange(node_emb, 's n f -> n (s f)') if self.reshape else torch.stack(node_emb)
        # [scales, edges, features] -> [edges, scales * features]
        edge_emb = rearrange(edge_emb, 's e f -> e (s f)') if self.reshape else torch.stack(edge_emb)

        return node_emb, edge_emb

    def out_features(self):
        # NOTE: Is this correct?
        return self.num_features * len(self.wavelet_constructor)

class ScatteringActivation(nn.Module):
    def __init__(self, activation=F.silu):
        super().__init__()
        # self.norm_node = nn.BatchNorm1d(self.num_features)
        self.activation = activation

    def forward(self, node_emb: torch.Tensor, edge_emb: torch.Tensor) -> Tuple[torch.Tensor]:
        # TODO: think about normalization?
        # node_emb = self.norm_node(rearrange(node_emb, 's b l -> (w b) l'))
        # node_emb = rearrange(node_emb, '(w b) l -> s b l', s=len(self.wavelet_constructor))
        node_emb = self.activation(node_emb)
        edge_emb = self.activate(edge_emb)
        return node_emb, edge_emb


class FeatureSelfAttention(nn.Module):
    '''
    Feature-feature attention.
    The descriptor vector for each feature is constructed from the different wavelet scales.
    '''
    def __init__(self, num_scales: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(num_scales, num_heads=1, batch_first=True)
        self.proj = nn.Linear(num_scales, 1)

    def forward(self, x, return_attn: bool = False):
        '''
        The shape of x is [B, F, S]: (batch size, num features, scattering scales).
        '''
        # Apply self-attention across the feature dimension (F positions)
        x, attn_weights = self.attn(x, x, x)  # [B, F, S]
        x = self.proj(x)                      # [B, F, 1]
        x = x.squeeze(-1)                     # [B, F]
        if return_attn:
            return x, attn_weights
        return x


class NicheAttention(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.gate_nn = nn.Linear(num_features, 1)

    def forward(self, x, batch, return_attn=False):
        gate_scores = self.gate_nn(x).squeeze(-1)                        # [N]
        attn_weights = scatter_softmax(gate_scores, batch)               # [N]
        out = scatter_sum(x * attn_weights.unsqueeze(-1), batch, dim=0)  # [B, F]

        if return_attn:
            return out, attn_weights
        return out


class HypergraphScatteringNet(nn.Module):
    '''
    Hypergraph Scattering Network (HSN) module.
    Now assuming only using the node features output.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        trainable_laziness (bool): Whether the laziness parameter is trainable.
        trainable_scales (bool): Whether the scales parameter is trainable.
        fixed_weights (bool): Whether the weights are fixed.
        layout (list): List of strings specifying the layout of the network.
        normalize (str): Normalization method to use.
        pooling (str): Pooling method to use.
        classifier (str): Attention or MLP.
        **kwargs: Additional keyword arguments.
    '''

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_features: int = 18085,
                 trainable_laziness=False,
                 trainable_scales=False,
                 fixed_weights=True,
                 layout=['hsm', 'hsm'],
                 normalize="right",
                 pooling='attention',
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.trainable_laziness = trainable_laziness
        self.trainable_scales = trainable_scales
        self.fixed_weights = fixed_weights
        self.layout = layout
        self.layers = []
        self.out_dimensions = [in_channels]
        self.normalize = normalize
        self.pooling = pooling
        scale_list = kwargs.get('scale_list', None)
        if scale_list is None:
            scale_list = [0, 1, 2, 4, 8, 16]
        self.scale_list = scale_list

        for layout_name in layout:
            if layout_name == 'hsm':
                scattering_layer = HyperScatteringModule(
                    self.out_dimensions[-1],
                    num_features=self.num_features,
                    trainable_laziness=trainable_laziness,
                    trainable_scales=self.trainable_scales,
                    fixed_weights=self.fixed_weights,
                    normalize=normalize,
                    scale_list=self.scale_list)
                self.layers.append(scattering_layer)
                self.out_dimensions.append(scattering_layer.out_features())
            elif layout_name == 'act':
                self.layers.append(ScatteringActivation())
            elif layout_name == 'dim_reduction':
                input_dim = self.out_dimensions[-1]
                output_dim = input_dim // 2
                self.out_dimensions.append(output_dim)
                self.layers.append(nn.Linear(input_dim, output_dim))
            else:
                raise NotImplementedError

        self.layers = nn.ModuleList(self.layers)

        # Attention pooling over nodes to help identify niche importance.
        self.niche_attention = NicheAttention(num_features=self.out_dimensions[-1])

        # Self-attention among features to help identify feature importance.
        self.feature_attention = FeatureSelfAttention(num_scales=len(self.scale_list))

        # Final classifier.
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, self.num_features),
            torch.nn.ELU(),
            torch.nn.Linear(self.num_features, self.out_channels),
        )

    # def interpret_feature_importance(self):
    #     '''
    #     Aggregate the MLP weights to interpret feature importance.
    #     '''
    #     linear_layers = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
    #     W = linear_layers[-1].weight
    #     for layer in linear_layers[:-1][::-1]:
    #         W = W @ layer.weight  # Chain multiplication
    #     # NOTE: The order of weights are: [(scale 1, feature 1), (scale 1, feature 2), ... (scale 2, feature 1), ...].
    #     return W.squeeze()  # [wavelet scales * num features]

    def forward(self,
                x: torch.Tensor,
                hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None,
                hyperedge_attr: Optional[torch.Tensor] = None,
                num_edges: Optional[int] = None,
                batch: Optional[torch.Tensor] = None,
                return_wavelet_embeddings: bool = False,
                return_attention: bool = False):
        '''
        Forward pass of the HSN module.

        Args:
            x (torch.Tensor): Input tensor.
            hyperedge_index (torch.Tensor): Hyperedge index tensor.
            hyperedge_weight (torch.Tensor, optional): Hyperedge weight tensor.
            hyperedge_attr (torch.Tensor, optional): Hyperedge attribute tensor.
            num_edges (int, optional): Number of edges.
            batch (torch.Tensor, optional): Batch tensor.
            return_wavelet_embeddings (bool): Whether to return the wavelet embeddings.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Hyperedge attribute tensor.

        '''
        # row, col = hyperedge_index
        # edge_batch = batch[row]
        curr_value = 0
        node_in_hyperedge = []
        for edge_idx, val in enumerate(hyperedge_index[1, :]):
            if val == curr_value:
                node_in_hyperedge.append(hyperedge_index[0, edge_idx])
                curr_value += 1
        edge_batch = torch.tensor(node_in_hyperedge, device = hyperedge_index.device)
        for i, layer in enumerate(self.layers):
            if self.layout[i] == 'hsm':
                x, hyperedge_attr = layer(x, hyperedge_index, hyperedge_weight, hyperedge_attr, num_edges)
                # TODO add batch norm before non-linearity inside the hsm!
            elif self.layout[i] == 'act':
                x, hyperedge_attr = layer(x, hyperedge_attr)
                # TODO add batch norm before non-linearity inside the hsm!
            elif self.layout[i] == 'dim_reduction':
                x = layer(x) # TODO add batch norm and non-linearity!
                hyperedge_attr = layer(hyperedge_attr)
            else:
                raise ValueError

        if return_wavelet_embeddings:
            return x

        # Apply selected pooling
        if self.pooling is not None:
            assert batch is not None

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
            if edge_batch is not None:
                hyperedge_attr = global_mean_pool(hyperedge_attr, edge_batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
            if edge_batch is not None:
                hyperedge_attr = global_max_pool(hyperedge_attr, edge_batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
            if edge_batch is not None:
                hyperedge_attr = global_add_pool(hyperedge_attr, batch)
        elif self.pooling == 'attention':
            if return_attention:
                x, niche_attn = self.niche_attention(x, batch, return_attn=True)
            else:
                x = self.niche_attention(x, batch)
        else:
            raise ValueError(f'Pooling method {self.pooling} not supported.')

        # Isolate the scattering scales to a separate dimension.
        x = rearrange(x, 'b (s f) -> b f s', s=len(self.scale_list))

        if return_attention:
            x, feature_attn = self.feature_attention(x, return_attn=True)
            return niche_attn, feature_attn

        x = self.feature_attention(x)
        x = self.classifier(x)
        # NOTE: Do not add softmax here, because torch.nn.CrossEntropyLoss() internally performs softmax.
        return x


if __name__ == '__main__':
    model = HypergraphScatteringNet(
        in_channels=64,
        hidden_channels=16,
        out_channels=1,
        num_features=10,
        trainable_laziness=False,
        trainable_scales=True,
        activation=None,  # just get one layer of wavelet transform
        fixed_weights=True,
        layout=['hsm'],
        normalize='right',
        pooling='linear_combination',
        scale_list=[0,1,2,4]
    )
