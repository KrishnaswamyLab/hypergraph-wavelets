#from hypg_scattering.models.hyper_scattering_net import HSN
import sys
# fix this later!
# sys.path.insert(0, '/home/sumry2023_cqx3/hypergraph_scattering')
sys.path.append('../../')
from hypgs.models.hsn_pyg import HSN
import torch
import unittest
from torch_geometric.datasets import TUDataset
from dhg import Hypergraph
from hypgs.utils.data import HGDataset
from torch_geometric.loader import DataLoader

original_dataset = TUDataset(root='../data/', name="MUTAG", use_node_attr=True)
to_hg_func = lambda g: Hypergraph.from_graph_kHop(g, 1)
hgdataset = HGDataset(original_dataset, to_hg_func)
dl = DataLoader(hgdataset, batch_size=32, shuffle=True)

class TestHSN(unittest.TestCase):
    pass

def generate_test_cases():
    trainable_laziness_values = [False, True]
    trainable_scales_values = [False, True]
    activation_values = ["blis", "leaky_relu", 'modulus', None]
    fixed_weights_values = [True, False]
    layout_values = [['hsm', 'hsm'], ['hsm', 'dim_reduction'], ['dim_reduction', 'dim_reduction']]
    pooling_values = [None, 'sum', 'mean', 'max']
    for trainable_laziness in trainable_laziness_values:
        for trainable_scales in trainable_scales_values:
            for activation in activation_values:
                for fixed_weights in fixed_weights_values:
                    for layout in layout_values:
                        for pooling in pooling_values:
                            layout_str = '_'.join(layout)
                            test_name = f"test_hsn_{trainable_laziness}_{trainable_scales}_{activation}_{fixed_weights}_{layout_str}"
                            test_method = lambda self: testHSNPooling(
                                trainable_laziness, trainable_scales, activation, fixed_weights, layout, pooling)
                            setattr(TestHSN, test_name, test_method)

def testHSNPooling(trainable_laziness, trainable_scales, activation, fixed_weights, layout, pooling):
    model = HSN(in_channels=7, 
                hidden_channels=16,
                out_channels = 2, 
                trainable_laziness = False,
                trainable_scales = False, 
                activation = 'modulus', 
                fixed_weights=True, 
                layout=['hsm'], 
                normalize='right', 
                pooling=pooling,
            )
    batch = next(iter(dl))
    x, edge_index, edge_attr, y, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
    node_feat, he_feat = model(x=x, hyperedge_index=edge_index, hyperedge_attr=edge_attr, batch=batch_idx)
    if pooling is not None:
        assert node_feat.size() == (32, 2)
    else:
        assert node_feat.size(0) == x.size(0)
        assert node_feat.size(1) == (2)

generate_test_cases()

if __name__ == '__main__':
    unittest.main()