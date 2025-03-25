
import os
import argparse
from tqdm import tqdm
import anndata as ad
import scanpy as sc

import torch
from dhg import Hypergraph

from src.hypergraphs.featurizers import get_hyperedge_features
from src.models.hsn_pyg import HSN
from src.utils.hypergraph_utils import HGDataset
from src.graphs.builder import return_graph_data


# my defaults are python main.py --data_dir data/ --output_dir wavelet_features/ --k_hop 1 
# for hyperedge averaging: main.py --data_dir data/ --output_dir hyperedge_avg/ --k_hop 3 --hyperedge_features gene_expression --wavelets 0

FEATURE_DIR = f'../data/processed/wavelet_features/{patient_name}/1_hop/'

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, default='data/interim')
    argparser.add_argument('--output_dir', type=str, default='data/processed/wavelet_features')
    argparser.add_argument('--k_hop', type=int, default=1)
    argparser.add_argument('--hyperedge_features', nargs='+', default = ['gene_expression', 'gene_correlation', 'diffused_gene_correlation'], type=str)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--wavelets', type=int, default=1)
    argparser.add_argument('--norm_target_sum', type=int, default=1e4)
    
    args = argparser.parse_args()

    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir

    k_hop = args.k_hop
    hyperedge_features_list = args.hyperedge_features
    norm_target_sum = args.norm_target_sum

    print(OUTPUT_DIR)
    
    if os.path.isdir(OUTPUT_DIR) == False:
        os.mkdir(OUTPUT_DIR)
    if os.path.isdir(DATA_DIR) == False:
        raise ValueError('Data directory does not exist')

    datasets = os.listdir(DATA_DIR)

    print(datasets)

    for dataset_name in tqdm(datasets):
        ######################################
        # LOAD IN DATA AND PREPARE MODEL
        ######################################
        dataset_path = os.path.join(DATA_DIR,dataset_name)
        print(dataset_name)
        adata = ad.read_h5ad(dataset_path)
        data = return_graph_data(adata,norm_target_sum=norm_target_sum)

        original_dataset = [data]

        to_hg_func = lambda g: Hypergraph.from_graph_kHop(g, k_hop) # what should k be? 3?

        dataset = HGDataset(original_dataset, to_hg_func)

        model = HSN(in_channels=180, 
                    hidden_channels=16,
                    out_channels = 1, 
                    trainable_laziness = False,
                    trainable_scales = False, 
                    activation = None, # just get one layer of wavelet transform 
                    fixed_weights=True, 
                    layout=['hsm'], 
                    normalize='right', 
                    pooling='max',
                    task = 'node_representation',
                    scale_list = [0,1,2,4,8] #1,2,4,8,16
        )
        model.eval()

        ######################################
        # FEATURIZATION OF HYPEREDGES 
        ######################################
        # PREPROCESSING
        # start: nodes - cells, hyperedges - neighborhoods
        # get features for hyperedges (neighborhoods) by doing one step of message passing into the hyperedges
        # get hyperedge (neighborhood) features by getting cell type histogram
        # last preprocessing step: flip the graph: nodes -> neighborhoods, hyperedges -> cells

        # P = HH^T   operators R^N -> R->N

        # get the features for the hyperedges
        #hyperedge_features_list = ['cell_type_hist', 'gene_expression', 'gene_correlation', 'diffused_gene_correlation']

        args_dict = {'num_diffusions': 1, 'correlation_pairs': [(0,1), (0,2), (1,2)]}
        # TODO
        # correlation values in hyperedge (choose some), 
        # correlation between gene and diffused gene, [DONE, please check ]
        # stdev of gene expression and cell type (low priority)
        hyperedge_features = get_hyperedge_features(adata, 
                                                    dataset, 
                                                    features = hyperedge_features_list,
                                                    **args_dict)

        hyperedge_features[torch.isnan(hyperedge_features)] = 0
        # modify the dataset to switch to dual version of hypergraph
        dual_edge_index = torch.flip(dataset[0].edge_index, [0])
        dataset[0].edge_index = dual_edge_index
        dataset[0].edge_attr = dataset[0].x 
        dataset[0].x = hyperedge_features 

        ######################################
        # RUN MODEL AND SAVE RESULTS
        ######################################
        if args.wavelets:
            node_feat, edge_feat = model(dataset[0].x, 
                                        dataset[0].edge_index, 
                                        hyperedge_attr = torch.zeros((dataset[0].edge_attr.shape[0], dataset[0].x.shape[1])))
            # save the features (node_feat are the NEIGHBORHOOD features)
        else:
            node_feat = hyperedge_features

        torch_save_dir = os.path.join(OUTPUT_DIR, dataset_name.split('.')[0] + '_neighborhood_feat.pt')
        torch.save(node_feat, torch_save_dir)