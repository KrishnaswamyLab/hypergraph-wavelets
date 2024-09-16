import anndata
import torch
import pandas as pd
from torch_geometric.data.hypergraph_data import HyperGraphData
from src.models.cell_count_type_convs import CountCellTypesConv
from src.models.hsn_pyg import HyperDiffusion
from src.utils.cell_categories import retrieve_all_cell_types_categories
from tqdm import tqdm

def cell_type_feat(adata, dataset):
    #Retrieve all possible cell_types
    
    # we want to do aggregation based on the different granularities of cell types
    complete_cell_types_df = retrieve_all_cell_types_categories(adata)
    enc_df = pd.get_dummies(complete_cell_types_df)
    encoded_data = torch.tensor(enc_df.values, dtype=torch.float)

    # message pass into the hyperedges
    types_conv = CountCellTypesConv()
    data = HyperGraphData(edge_index = dataset[0].edge_index, x = encoded_data)
    node_feature_counts = types_conv(data.x, data.edge_index)
    count_df = pd.DataFrame(node_feature_counts.detach().numpy(), columns=enc_df.columns).astype(int)
    
    # intersect the columns that start with Class_
    class_cols = [col for col in count_df.columns if col.startswith('Class_')]
    class_col_df = count_df[class_cols]
    subclass_cols = [col for col in count_df.columns if col.startswith('Subclass_')]
    subclass_col_df = count_df[subclass_cols]
    supertype_cols = [col for col in count_df.columns if col.startswith('Supertype_')]
    supertype_col_df = count_df[supertype_cols]

    count_df = torch.tensor(count_df.values, dtype=torch.float)
    return count_df

def gene_expression_feat(adata, dataset):
    # GENE EXPRESSION FEATURIZATION
    data_gene = HyperGraphData(edge_index = dataset[0].edge_index, x = dataset[0].x, edge_attr = dataset[0].edge_attr)
    diffuser = HyperDiffusion(in_channels=180, out_channels=180)
    _, edge_feat = diffuser(data_gene.x, data_gene.edge_index, hyperedge_attr = data_gene.edge_attr)
    return edge_feat

def diffused_gene_correlation(adata, dataset, num_diffusions = 1):
    original_data = dataset[0].x
    data_gene = HyperGraphData(edge_index = dataset[0].edge_index, x = dataset[0].x, edge_attr = dataset[0].edge_attr)
    diffuser = HyperDiffusion(in_channels=180, out_channels=180)
    node_feat = data_gene.x
    _edge_feat = data_gene.edge_attr
    for i in range(num_diffusions):
        node_feat, _edge_feat = diffuser(data_gene.x, data_gene.edge_index, hyperedge_attr = _edge_feat)
    diffused_data = node_feat

    hyperedges = dataset[0].edge_index[1].unique() # check the convention on edge_index for hpyeredges
    hyperedge_correlations = torch.zeros((len(hyperedges), dataset[0].x.shape[1]))
    for hyperedge in tqdm(hyperedges, desc='Diffused Gene Correlation'):
        # get nodes in each hyperedge
        nodes = dataset[0].edge_index[0][dataset[0].edge_index[1] == hyperedge]
        # get the correlation between the original data and the diffused data
        original_data_hyperedge = original_data[nodes].T
        diffused_data_hyperedge = diffused_data[nodes].T
        for ind, (x,y) in enumerate(zip(original_data_hyperedge, diffused_data_hyperedge)):
            hyperedge_correlations[hyperedge, ind] = torch.corrcoef(torch.stack((x, y)))[0,1]
        
        #correlation = np.corrcoef(stacked, rowvar=False)[0,1,:]

    return hyperedge_correlations

    # within each hyperedge, get correlation original_data, diffused_data

def gene_correlation(adata, dataset, correlation_pairs = [(0,1), (0,2), (1,2)]):
    hyperedges = dataset[0].edge_index[1].unique() # check the convention on edge_index for hyperedges
    hyperedge_correlations = torch.zeros((len(hyperedges), len(correlation_pairs)))

    for hyperedge in tqdm(hyperedges, desc='Gene Correlation'):
        # get nodes in each hyperedge
        nodes = dataset[0].edge_index[0][dataset[0].edge_index[1] == hyperedge]
        # get the correlation between the original data and the diffused data
        data_hyperedge = dataset[0].x[nodes]
        for corr_ind, pair in enumerate(correlation_pairs):
            gene_a_ind, gene_b_ind = pair
            gene_a = data_hyperedge[:, gene_a_ind]
            gene_b = data_hyperedge[:, gene_b_ind]
            hyperedge_correlations[hyperedge, corr_ind] = torch.corrcoef(torch.stack((gene_a, gene_b)))[0,1]

    return hyperedge_correlations

def get_hyperedge_features(adata, 
                           dataset,
                           features = ['cell_type_hist', 'gene_expression'],
                           **kwargs):

    feat  = None
    for feature in features:
        if feature == 'cell_type_hist':
            feat_new = cell_type_feat(adata, dataset)
        elif feature == 'gene_expression':
            feat_new = gene_expression_feat(adata, dataset)
        elif feature == 'diffused_gene_correlation':
            num_diffusions = kwargs.get('num_diffusions', 1)
            feat_new = diffused_gene_correlation(adata, dataset, num_diffusions)
        elif feature == 'gene_correlation':
            # get correlation pairs from kwargs
            correlation_pairs = kwargs.get('correlation_pairs', [(0,1), (0,2), (1,2)])
            feat_new = gene_correlation(adata, dataset, correlation_pairs)
        else:
            raise ValueError('Feature not supported')
        # concatenate the features
        
        if feat is None:
            feat = feat_new
        else:
            feat = torch.cat([feat, feat_new], dim=1)

    return feat