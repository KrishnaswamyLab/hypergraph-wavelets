#from hypg_scattering.models.hyper_scattering_net import HSN
import sys
# fix this later!
# sys.path.insert(0, '/home/sumry2023_cqx3/hypergraph_scattering')
sys.path.append('../../')
from hypgs.models.hyper_scattering_net import HyperDiffusion
from hypgs.models.hsn_pyg import HyperDiffusion as HyperDiffusionPYG
from hypgs.models.hyper_scattering_net import HyperScatteringModule
from hypgs.models.hsn_pyg import HyperScatteringModule as HyperScatteringModulePYG
from hypgs.models.hyper_scattering_net import HSN
from hypgs.models.hsn_pyg import HSN as HSNPYG
from hypgs.utils.data import get_HyperGraphData
import dhg 
import torch
import unittest

def gen_hypg_deg_dataset(num_v, num_e, num_graphs):
    hypergraph_dataset = [dhg.random.hypergraph_Gnm(num_v, num_e, method = 'low_order_first') for _ in range(num_graphs)]
    return hypergraph_dataset

class TestHSNPYG(unittest.TestCase):
    def testHyperDiffusion(self):
        test_vs_HyperDiffusion(trainable_laziness = False, fixed_weights=True)
        test_vs_HyperDiffusion(trainable_laziness = True, fixed_weights=True)
        test_vs_HyperDiffusion(trainable_laziness = False, fixed_weights=False)
        test_vs_HyperDiffusion(trainable_laziness = True, fixed_weights=False)

    def testHyperScatteringModule(self):
        trainable_laziness_values = [False, True]
        trainable_scales_values = [False, True]
        activation_values = ["blis", "leaky_relu", 'modulus', None]
        fixed_weights_values = [True, False]

        for trainable_laziness in trainable_laziness_values:
            for trainable_scales in trainable_scales_values:
                for activation in activation_values:
                    for fixed_weights in fixed_weights_values:
                        test_vs_HyperScatteringModule(trainable_laziness, trainable_scales, activation, fixed_weights)


    # Test cases for HSN
    # def test_hsn(self):
    #     trainable_laziness_values = [False, True]
    #     trainable_scales_values = [False, True]
    #     activation_values = ["blis", "leaky_relu", 'modulus', None]
    #     fixed_weights_values = [True, False]
    #     layout_values = [['hsm', 'hsm'], ['hsm', 'dim_reduction'], ['dim_reduction', 'dim_reduction']]

    #     for trainable_laziness in trainable_laziness_values:
    #         for trainable_scales in trainable_scales_values:
    #             for activation in activation_values:
    #                 for fixed_weights in fixed_weights_values:
    #                     for layout in layout_values:
    #                         layout_str = '_'.join(layout)
    #                         test_name = f"test_hsn_{trainable_laziness}_{trainable_scales}_{activation}_{fixed_weights}_{layout_str}"
    #                         test_method = lambda self: self.test_vs_HSN(
    #                             trainable_laziness, trainable_scales, activation, fixed_weights, layout)
    #                         setattr(TestHSNPYG, test_name, test_method)
    # def testHSN(self):
    #     trainable_laziness_values = [False, True]
    #     trainable_scales_values = [False, True]
    #     activation_values = ["blis", "leaky_relu", 'modulus', None]
    #     fixed_weights_values = [True, False]
    #     layout_values = [['hsm', 'hsm'], ['hsm', 'dim_reduction'], ['dim_reduction', 'dim_reduction']]
    #     for trainable_laziness in trainable_laziness_values:
    #         for trainable_scales in trainable_scales_values:
    #             for activation in activation_values:
    #                 for fixed_weights in fixed_weights_values:
    #                     for layout in layout_values:
    #                         test_vs_HSN(trainable_laziness, trainable_scales, activation, fixed_weights, layout)


def generate_test_cases():
    trainable_laziness_values = [False, True]
    trainable_scales_values = [False, True]
    activation_values = ["blis", "leaky_relu", 'modulus', None]
    fixed_weights_values = [True, False]
    layout_values = [['hsm', 'hsm'], ['hsm', 'dim_reduction'], ['dim_reduction', 'dim_reduction']]

    for trainable_laziness in trainable_laziness_values:
        for trainable_scales in trainable_scales_values:
            for activation in activation_values:
                for fixed_weights in fixed_weights_values:
                    for layout in layout_values:
                        layout_str = '_'.join(layout)
                        test_name = f"test_hsn_{trainable_laziness}_{trainable_scales}_{activation}_{fixed_weights}_{layout_str}"
                        test_method = lambda self: test_vs_HSN(
                            trainable_laziness, trainable_scales, activation, fixed_weights, layout)
                        setattr(TestHSNPYG, test_name, test_method)

def get_test_data():
    num_graphs = 100
    num_v = 10
    num_e = 20 
    # generate graph dataset
    train_dataset = gen_hypg_deg_dataset(num_v, num_e, num_graphs)
    hg = train_dataset[0]
    starting_features = torch.randn(num_v, num_v)
    signal_features = num_v
    num_e_hg = hg.num_e 
    Y = torch.randn(num_e_hg, num_v)
    hgdata = get_HyperGraphData(hg, starting_features, Y, None)
    return signal_features, hg, starting_features, Y, hgdata

def test_vs_HyperDiffusion(trainable_laziness = False, fixed_weights=True):
    signal_features, hg, starting_features, Y, hgdata = get_test_data()
    hyd = HyperDiffusion(in_channels=signal_features, out_channels=signal_features, trainable_laziness=trainable_laziness, fixed_weights=fixed_weights)
    hyd_pyg = HyperDiffusionPYG(in_channels=signal_features, out_channels=signal_features, trainable_laziness=trainable_laziness, fixed_weights=fixed_weights, normalize='right')
    if trainable_laziness:
        hyd_pyg.lazy_layer.weights = hyd.lazy_layer.weights
    if not fixed_weights:
        hyd_pyg.lin_neigh = hyd.lin_neigh
        hyd_pyg.lin_self = hyd.lin_self
    node_feat, edge_feat = hyd(hg, starting_features, Y)
    node_feat2, edge_feat2 = hyd_pyg(x=hgdata.x, hyperedge_index=hgdata.edge_index, hyperedge_attr=hgdata.edge_attr)
    assert torch.isclose(node_feat2, node_feat).all(), "Node features are not equal"
    assert torch.isclose(edge_feat2, edge_feat).all(), "Edge features are not equal"

def test_vs_HyperScatteringModule(trainable_laziness = False, trainable_scales = False, activation = "blis", fixed_weights=True):
    signal_features, hg, starting_features, Y, hgdata = get_test_data()
    hsm = HyperScatteringModule(in_channels=signal_features, trainable_laziness = trainable_laziness, trainable_scales = trainable_scales, activation = activation, fixed_weights=fixed_weights)
    hsm_pyg = HyperScatteringModulePYG(in_channels=signal_features, trainable_laziness = trainable_laziness, trainable_scales = trainable_scales, activation = activation, fixed_weights=fixed_weights, normalize="right")
    if trainable_laziness:
        hsm_pyg.diffusion_layer1.lazy_layer.weights = hsm.diffusion_layer1.lazy_layer.weights
    if not fixed_weights:
        hsm_pyg.diffusion_layer1.lin_neigh = hsm.diffusion_layer1.lin_neigh
        hsm_pyg.diffusion_layer1.lin_self = hsm.diffusion_layer1.lin_self
    s_nodes, s_edges = hsm(hg, starting_features, Y)
    s_nodes2, s_edges2 = hsm_pyg(x=hgdata.x, hyperedge_index=hgdata.edge_index, hyperedge_attr=hgdata.edge_attr)
    assert torch.isclose(s_nodes, s_nodes2).all()
    assert torch.isclose(s_edges, s_edges2).all()

def test_vs_HSN(
        trainable_laziness = False, 
        trainable_scales = False, 
        activation = "modulus", 
        fixed_weights=True, 
        layout = ['hsm','hsm'], 
        normalize="right",
        **kwargs
    ):
    signal_features, hg, starting_features, Y, hgdata = get_test_data()
    hsn = HSN(in_channels=signal_features, 
              hidden_channels=8,
              out_channels = 5, 
              trainable_laziness = trainable_laziness,
              trainable_scales = trainable_scales, 
              activation = activation, 
              fixed_weights=fixed_weights, 
              layout=layout, 
              normalize=normalize, 
              **kwargs)
    hsn_pyg = HSNPYG(in_channels=signal_features, 
              hidden_channels=8,
              out_channels = 5, 
              trainable_laziness = trainable_laziness,
              trainable_scales = trainable_scales, 
              activation = activation, 
              fixed_weights=fixed_weights, 
              layout=layout, 
              normalize=normalize, 
              task='regression',
              **kwargs)
    for i, layout in enumerate(hsn.layout):
        if layout == 'hsm':
            if trainable_laziness:
                hsn_pyg.layers[i].diffusion_layer1.lazy_layer.weights = hsn.layers[i].diffusion_layer1.lazy_layer.weights
            if not fixed_weights:
                hsn_pyg.layers[i].diffusion_layer1.lin_neigh = hsn.layers[i].diffusion_layer1.lin_neigh
                hsn_pyg.layers[i].diffusion_layer1.lin_self = hsn.layers[i].diffusion_layer1.lin_self
        elif layout == 'dim_reduction':
            hsn_pyg.layers[i] = hsn.layers[i]
        else:
            raise ValueError(f"Unknown layout {layout}")
    hsn_pyg.batch_norm = hsn.batch_norm
    hsn_pyg.mlp = hsn.mlp

    x, hyperedge_attr = hsn(hg, starting_features, Y)
    x2, hyperedge_attr2 = hsn_pyg(x=hgdata.x, hyperedge_index=hgdata.edge_index, hyperedge_attr=hgdata.edge_attr)
    assert torch.isclose(x, x2).all()
    assert torch.isclose(hyperedge_attr, hyperedge_attr2).all()

generate_test_cases()

if __name__ == '__main__':
    unittest.main()