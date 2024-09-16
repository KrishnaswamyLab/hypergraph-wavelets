"""
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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange
from torch.nn import Linear 
from torch_geometric.nn.pool import global_mean_pool 
from torch_geometric.nn import GCNConv 
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import MessagePassing
from .hyper_scattering_net import LazyLayer
from torch_geometric.utils import scatter, softmax
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, GlobalAttention
import pytorch_lightning as pl
import torchmetrics

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
            out_edge = self.propagate(hyperedge_index, x=out,norm=D_he_inv,
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
    def __init__(self, in_channels, trainable_laziness=False, trainable_scales=False, activation="blis", fixed_weights=True, normalize="right", reshape=True, scale_list = None):
        super().__init__()
        self.in_channels = in_channels
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = HyperDiffusion(in_channels, in_channels, trainable_laziness, fixed_weights, normalize)
        if scale_list is None:
            self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
                [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=torch.float, requires_grad=trainable_scales))
            self.diffusion_levels = 16
        else:
            # ensure that scale list is an increasing list of integers with 0 as the first element
            # ensure that 1 is the second element
            assert all(isinstance(x, int) for x in scale_list)
            assert all(scale_list[i] < scale_list[i+1] for i in range(len(scale_list)-1))
            assert scale_list[0] == 0
            assert scale_list[1] == 1

            self.diffusion_levels = scale_list[-1]
            wavelet_matrix = torch.zeros(len(scale_list), self.diffusion_levels+1, dtype = torch.float, requires_grad = trainable_scales)
            for i in range(len(scale_list) - 1):
                wavelet_matrix[i, scale_list[i]] = 1
                wavelet_matrix[i, scale_list[i+1]] = -1
            wavelet_matrix[-1, -1] = 1
            self.wavelet_constructor = torch.nn.Parameter(wavelet_matrix)

        if activation == "blis":
            self.activations = [lambda x: torch.relu(x), lambda x: torch.relu(-x)]
        elif activation == None:
            self.activations = [lambda x : x]
        elif activation == "modulus":
            self.activations = [lambda x: torch.abs(x)]
        elif activation == "leaky_relu":
            m = nn.LeakyReLU()
            self.activations = [lambda x: m(x)]
        self.reshape = reshape

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None,
                hyperedge_attr: Optional[torch.Tensor] = None,
                num_edges: Optional[int] = None):
        features = x.shape[1]
        node_features = [x]
        edge_features = [hyperedge_attr]
        for i in range(self.diffusion_levels):
            node_feat, edge_feat = self.diffusion_layer1(x=node_features[-1], hyperedge_index=hyperedge_index, hyperedge_weight=hyperedge_weight, hyperedge_attr=edge_features[-1])
            node_features.append(node_feat)
            edge_features.append(edge_feat)
        # Combine the diffusion levels into a single tensor.
        diffusion_levels = rearrange(node_features, 'i j k -> i j k').float()
        edge_diffusion_levels = rearrange(edge_features, 'i j k -> i j k').float()
        wavelet_coeffs = torch.einsum("ij,jkl->ikl", self.wavelet_constructor, diffusion_levels) # J x num_nodes x num_features x 1
        wavelet_coeffs_edges = torch.einsum("ij,jkl->ikl", self.wavelet_constructor, edge_diffusion_levels)
        # TODO add batch norm here!
        activated = [self.activations[i](wavelet_coeffs) for i in range(len(self.activations))]
        activated_edges = [self.activations[i](wavelet_coeffs_edges) for i in range(len(self.activations))]
        s_nodes = rearrange(activated, 'a w n f -> n (w f a)') if self.reshape else torch.stack(activated)
        s_edges = rearrange(activated_edges, 'a w e f -> e (w f a)') if self.reshape else torch.stack(activated_edges)
        return s_nodes, s_edges
    
    def out_features(self):
        return 6 * self.in_channels * len(self.activations)

class HSN(pl.LightningModule):
    """
    Hypergraph Scattering Network (HSN) module.
    Now assuming only using the node features output.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        trainable_laziness (bool): Whether the laziness parameter is trainable.
        trainable_scales (bool): Whether the scales parameter is trainable.
        activation (str): Activation function to use.
        fixed_weights (bool): Whether the weights are fixed.
        layout (list): List of strings specifying the layout of the network.
        normalize (str): Normalization method to use.
        pooling (str): Pooling method to use.
        task (str): Task type.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        **kwargs: Additional keyword arguments.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of hidden channels.
        trainable_laziness (bool): Whether the laziness parameter is trainable.
        trainable_scales (bool): Whether the scales parameter is trainable.
        activation (str): Activation function to use.
        fixed_weights (bool): Whether the weights are fixed.
        layout (list): List of strings specifying the layout of the network.
        layers (nn.ModuleList): List of network layers.
        out_dimensions (list): List of output dimensions.
        normalize (str): Normalization method to use.
        pooling (str): Pooling method to use.
        lr (float): Learning rate.
        batch_norm (BatchNorm): Batch normalization layer.
        fc1 (Linear): Fully connected layer 1.
        fc2 (nn.Linear): Fully connected layer 2.
        relu (nn.ReLU): ReLU activation function.
        batch_norm1 (nn.BatchNorm1d): Batch normalization layer 1.
        mlp (nn.Sequential): MLP network.

    """

    def __init__(self, 
                in_channels, 
                hidden_channels, 
                out_channels, 
                trainable_laziness=False, 
                trainable_scales=False, 
                activation="modulus", 
                fixed_weights=True, 
                layout=['hsm', 'hsm'], 
                normalize="right",
                pooling=None,
                task='classification', # assume on node only (not on hypergraphs)
                lr=1e-3,
                weight_decay=1e-5,
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
        self.normalize = normalize
        self.pooling = pooling
        self.lr = lr
        self.scale_list = kwargs.get('scale_list', None)

        assert task in ['classification', 'regression', 'node_representation']
        self.weight_decay = weight_decay
        self.task = task
        if pooling == 'attention':
            raise NotImplementedError
            # gate_nn = torch.nn.Sequential(
            #     torch.nn.Linear(hidden_channels, 1),
            #     torch.nn.Sigmoid()
            # )
            # self.attention_pool = GlobalAttention(gate_nn=gate_nn)


        for layout_ in layout:
            if layout_ == 'hsm':
                self.layers.append(HyperScatteringModule(self.out_dimensions[-1], 
                                                            trainable_laziness=trainable_laziness,
                                                            trainable_scales=self.trainable_scales, 
                                                            activation=self.activation, 
                                                            fixed_weights=self.fixed_weights,
                                                            normalize=normalize,
                                                            scale_list=self.scale_list
                                                            ))
                self.out_dimensions.append(self.layers[-1].out_features())
            elif layout_ == 'dim_reduction':
                input_dim = self.out_dimensions[-1]
                output_dim = input_dim // 2
                self.out_dimensions.append(output_dim)
                self.layers.append(nn.Linear(input_dim, output_dim))
            else:
                raise ValueError("Not yet implemented")
             
        self.layers = nn.ModuleList(self.layers)
        self.batch_norm = BatchNorm(self.out_dimensions[-1])

        self.fc1 = Linear(self.out_dimensions[-1], self.out_dimensions[-1] // 2)
        self.fc2 = nn.Linear(self.out_dimensions[-1] // 2, self.out_channels)
        
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(self.out_dimensions[-1] // 2)

        self.mlp = nn.Sequential(
            self.fc1,
            #self.batch_norm1,
            self.relu,
            self.fc2
        )

        # Metrics
        if self.task == 'classification':
            self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.out_channels)
            #self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.out_channels, pos_label=1)
            self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.out_channels)


    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_weight: Optional[torch.Tensor] = None,
                hyperedge_attr: Optional[torch.Tensor] = None,
                num_edges: Optional[int] = None,
                batch: Optional[torch.Tensor] = None):
        """
        Forward pass of the HSN module.

        Args:
            x (torch.Tensor): Input tensor.
            hyperedge_index (torch.Tensor): Hyperedge index tensor.
            hyperedge_weight (torch.Tensor, optional): Hyperedge weight tensor.
            hyperedge_attr (torch.Tensor, optional): Hyperedge attribute tensor.
            num_edges (int, optional): Number of edges.
            batch (torch.Tensor, optional): Batch tensor.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Hyperedge attribute tensor.

        """
        # row, col = hyperedge_index
        # edge_batch = batch[row]
        curr_value = 0 
        node_in_hyperedge = []
        for ind,val in enumerate(hyperedge_index[1,:]):
            if val == curr_value:
                node_in_hyperedge.append(hyperedge_index[0, ind])
                curr_value += 1
        edge_batch = torch.tensor(node_in_hyperedge, device = hyperedge_index.device)
        for il, layer in enumerate(self.layers):
            if self.layout[il] == 'hsm':
                x, hyperedge_attr = layer(x, hyperedge_index, hyperedge_weight, hyperedge_attr, num_edges)
                # TODO add batch norm before non-linearity inside the hsm!
            elif self.layout[il] == 'dim_reduction':
                x = layer(x) # TODO add batch norm and non-linearity!
                hyperedge_attr = layer(hyperedge_attr) 
            else:
                raise ValueError
            
        if self.task == 'classification':
            # Apply selected pooling
            if self.pooling is not None:
                assert batch is not None
            if self.pooling == 'mean':
                x = global_mean_pool(x, batch)
                hyperedge_attr = global_mean_pool(hyperedge_attr, batch)
            elif self.pooling == 'max':
                #import pdb; pdb.set_trace()
                x = global_max_pool(x, batch)
                if edge_batch is not None:
                    hyperedge_attr = global_max_pool(hyperedge_attr, edge_batch)
            elif self.pooling == 'sum':
                x = global_add_pool(x, batch)
                hyperedge_attr = global_add_pool(hyperedge_attr, batch)
            elif self.pooling == 'attention':
                # use a hyper GNN to learn attention weights?? I don't know...
                
                raise NotImplementedError
                # x = self.attention_pool(x, batch)
                # hyperedge_attr = self.attention_pool(hyperedge_attr, batch)
        #x = self.batch_norm(x)
        if self.task != 'node_representation':
            x = self.mlp(x)
        
        # compute the same process on the edges:
        #hyperedge_attr = self.batch_norm(hyperedge_attr)
        #hyperedge_attr = self.mlp(hyperedge_attr)

        if self.task == 'classification':
            # I don't think this is needed since F.cross_entropy assumes unnormalized logits
            #x = F.log_softmax(x, dim=1)
            pass

        return x, hyperedge_attr

    def common_step(self, batch, batch_idx, phase):
        x, edge_index, edge_attr, y, batch_id = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
        out, _ = self.forward(x, edge_index, hyperedge_attr=edge_attr, batch=batch_id)
        #import pdb; pdb.set_trace()
        #y = y[0]
        if self.task == 'classification':
            #import pdb; pdb.set_trace()
            loss = F.cross_entropy(out, y) # set to y[0] for spacegm, this may break things later
        elif self.task == 'regression':
            loss = F.mse_loss(out, y)
        elif self.task == 'node_representation':
            loss = 0
        else:
            raise ValueError(f"task must be one of 'classification' or 'regression', not {self.task}")
    
        # Logging metrics
        self.log(f'{phase}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.task == 'classification':
            acc = self.accuracy(out, y)
            auc = self.auroc(out, y)
            self.log(f'{phase}_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{phase}_auc', auc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
