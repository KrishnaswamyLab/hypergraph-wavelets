from vendi_score import vendi
import torch
import argparse 
import numpy as np
import os 
import anndata as ad

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, default='/vast/palmer/pi/krishnaswamy_smita/jcr222/hypergraphs/data/')
    argparser.add_argument('--feature_dir', type=str, default='wavelet_features/')
    argparser.add_argument('--output_dir', type=str, default='vendi_out/')
    argparser.add_argument('--k_hop', type=int, default=1)
    argparser.add_argument('--hyperedge_features', nargs='+', default = ['cell_type_hist', 'gene_expression', 'gene_correlation', 'diffused_gene_correlation'], type=str)
    argparser.add_argument('--vendi_score_subset', type=int, default=-1)
    argparser.add_argument('--lin_prob_target', type = str, default = 'braak')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--wavelets', type=int, default=1)
    args = argparser.parse_args()

    DATA_DIR = args.data_dir
    FEATURE_DIR = args.feature_dir
    OUTPUT_DIR = args.output_dir
    k_hop = args.k_hop
    hyperedge_features_list = args.hyperedge_features
    lin_prob_target = args.lin_prob_target
    vendi_score_subset = args.vendi_score_subset

    datasets = os.listdir(DATA_DIR)
    num_datasets = len(datasets)
    print(f'Number of datasets: {num_datasets}')
    example_features = torch.load(FEATURE_DIR + datasets[0] + '_neighborhood_feat.pt', weights_only=True)

    num_features = example_features.shape[1]
    del example_features
    vendi_scores = []
    for ind, dataset_name in enumerate(datasets):
        print(DATA_DIR + dataset_name)
        adata = ad.read_h5ad(DATA_DIR + dataset_name)

        # load in the wavelet features
        hyperedge_feat = torch.load(FEATURE_DIR + dataset_name + '_neighborhood_feat.pt', weights_only=True)

        #print(f'num features: {hyperedge_feat.shape[1]}')
        
        # convert to numpy and set nans to 0
        hyperedge_feat = hyperedge_feat.detach()#.numpy()
        node_feat = hyperedge_feat
        node_feat[node_feat.isnan()] = 0

        ######################################
        # VENDI SCORE
        ######################################
        if vendi_score_subset == -1:
            # use all the data points
            node_feat = (node_feat - node_feat.mean(0)) / node_feat.std(0)
            cov = torch.matmul(node_feat.T, node_feat) / node_feat.shape[0]
            # set nan to zero
            cov[torch.isnan(cov)] = 0
            vendi_score = vendi.score_dual(cov)
            sd = 0
        else:
            # randomly subset node_feat 
            reps = []
            for _ in range(5):
                node_feat = node_feat[torch.randperm(node_feat.shape[0])[:vendi_score_subset]]
                # normalize node_feat and get the covariance matrix
                node_feat = (node_feat - node_feat.mean(0)) / node_feat.std(0)
                cov = torch.matmul(node_feat.T, node_feat) / node_feat.shape[0]
                # set nan to zero
                cov[torch.isnan(cov)] = 0
                vendi_score = vendi.score_dual(cov)
                reps.append(vendi_score)
            vendi_score = np.mean(reps)
            sd = np.std(reps)
        vendi_scores.append(vendi_score)

        # print the vendi score and sd
        print(f'Vendi Score: {vendi_score} +/- {sd}')
        # save the vendi score
        with open(OUTPUT_DIR + dataset_name + '_vendi_score.txt', 'w') as f:
            f.write(f'Vendi Score: {vendi_score} +/- {sd}')
        f.close()

    # print vendi scores average and sd
    print(f'Average Vendi Score: {np.mean(vendi_scores)} +/- {np.std(vendi_scores)}')
