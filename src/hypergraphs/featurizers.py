import anndata
import torch
import pandas as pd
from torch_geometric.data.hypergraph_data import HyperGraphData
from src.models.cell_count_type_convs import CountCellTypesConv
from src.models.hypergraph_scattering import HyperDiffusion
from src.utils.cell_categories import retrieve_all_cell_types_categories
from tqdm import tqdm

def cell_type_distribution_feature(adata, hyperedges):
    """
    TODO: THIS IS HARD CODED FOR BRAIN ATLAS DATASET

    Compute cell type histogram features for hyperedges by aggregating
    cell types of nodes within each hyperedge.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with cell type information
    hyperedges : list
        List of hyperedges (list of node indices for each hyperedge)
    
    Returns:
    --------
    hyperedge_features : torch.Tensor
        Cell type count features for each hyperedge (num_hyperedges, num_features)
    """
    # Retrieve all possible cell_types
    complete_cell_types_df = retrieve_all_cell_types_categories(adata)
    enc_df = pd.get_dummies(complete_cell_types_df)
    
    # Aggregate cell types for each hyperedge
    hyperedge_features = []
    for hyperedge in hyperedges:
        # Sum the one-hot encoded cell types for all nodes in this hyperedge
        hyperedge_cell_types = enc_df.iloc[hyperedge].sum(axis=0)
        hyperedge_features.append(hyperedge_cell_types.values)
    
    hyperedge_features = torch.tensor(hyperedge_features, dtype=torch.float)
    return hyperedge_features


def gene_expression_feature(hyperedges, hyperedge_index, node_features):
    """
    Compute gene expression features for hyperedges using diffusion.
    
    Parameters:
    -----------
    hyperedges : list
        List of hyperedges
    hyperedge_index : torch.Tensor
        Hyperedge index tensor [2, num_connections]
    node_features : torch.Tensor
        Node gene expression features
    
    Returns:
    --------
    edge_feat : torch.Tensor
        Diffused gene expression features for hyperedges
    """
    num_hyperedges = len(hyperedges)
          
    # Initialize hyperedge attributes (e.g., mean of nodes in each hyperedge)
    initial_edge_attr = torch.zeros(num_hyperedges, node_features.shape[1])

    for i, hyperedge in enumerate(hyperedges):
        initial_edge_attr[i] = node_features[hyperedge].mean(dim=0)
    
    data_gene = HyperGraphData(
        edge_index=hyperedge_index, 
        x=node_features, 
        edge_attr=initial_edge_attr
    )
    
    diffuser = HyperDiffusion(in_channels=180, out_channels=180)
    _, edge_feat = diffuser(data_gene.x, data_gene.edge_index, hyperedge_attr=data_gene.edge_attr)
    
    return edge_feat

def diffused_gene_correlation(hyperedges, hyperedge_index, node_features, num_diffusions=1):
    """
    Compute correlation between original and diffused gene expression within each hyperedge.
    
    Parameters:
    -----------
    hyperedges : list
        List of hyperedges
    hyperedge_index : torch.Tensor
        Hyperedge index tensor [2, num_connections]
    node_features : torch.Tensor
        Node gene expression features
    num_diffusions : int
        Number of diffusion steps to apply
    
    Returns:
    --------
    hyperedge_correlations : torch.Tensor
        Correlation features for each hyperedge (num_hyperedges, num_genes)
    """
    num_hyperedges = len(hyperedges)
    original_data = node_features
    
    # Initialize hyperedge attributes (e.g., mean of nodes in each hyperedge)
    initial_edge_attr = torch.zeros(num_hyperedges, node_features.shape[1])
    for i, hyperedge in enumerate(hyperedges):
        initial_edge_attr[i] = node_features[hyperedge].mean(dim=0)
    
    # Create hypergraph data
    data_gene = HyperGraphData(
        edge_index=hyperedge_index, 
        x=node_features, 
        edge_attr=initial_edge_attr
    )
    
    # Apply diffusion
    diffuser = HyperDiffusion(in_channels=node_features.shape[1], out_channels=node_features.shape[1])
    node_feat = data_gene.x
    edge_feat = data_gene.edge_attr
    
    for i in range(num_diffusions):
        node_feat, edge_feat = diffuser(node_feat, data_gene.edge_index, hyperedge_attr=edge_feat)
    
    diffused_data = node_feat
    
    # Compute correlations for each hyperedge
    hyperedge_correlations = []
    for hyperedge in tqdm(hyperedges, desc='Diffused Gene Correlation'):
        # Get original and diffused data for this hyperedge
        original_data_hyperedge = original_data[hyperedge].T  # (num_genes, num_nodes)
        diffused_data_hyperedge = diffused_data[hyperedge].T  # (num_genes, num_nodes)
        
        # Compute correlation for each gene
        correlations = []
        for orig_gene, diff_gene in zip(original_data_hyperedge, diffused_data_hyperedge):
            corr_matrix = torch.corrcoef(torch.stack((orig_gene, diff_gene)))
            correlations.append(corr_matrix[0, 1].item())
        
        hyperedge_correlations.append(correlations)
    
    hyperedge_correlations = torch.tensor(hyperedge_correlations, dtype=torch.float)
    return hyperedge_correlations

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

def get_hyperedge_features(graph_data,
                           adata,
                           hyperedges,
                           hyperedge_index,
                           features=['gene_expression'],
                           **kwargs):
    """
    Extract features for hyperedges.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing cell information
    hyperedges : list
        List of hyperedges, where each hyperedge is a list of node indices
    features : list
        List of feature types to extract
    **kwargs : dict
        Additional parameters for specific feature extraction methods
    
    Returns:
    --------
    feat : torch.Tensor
        Concatenated feature tensor of shape (num_hyperedges, total_feature_dim)
    """
    feat = None
    print(f'Extracting hyperedge features: {features}')
    for feature in features:
        if feature == 'gene_expression':
            feat_new = gene_expression_feature(hyperedges, hyperedge_index, node_features=graph_data.x)
        elif feature == 'diffused_gene_correlation':
            num_diffusions = kwargs.get('num_diffusions', 1)
            feat_new = diffused_gene_correlation(hyperedges, hyperedge_index, node_features=graph_data.x, num_diffusions=1)
        elif feature == 'cell_type_hist':
            feat_new = cell_type_distribution_feature(adata, hyperedges)
        elif feature == 'gene_correlation':
            # get correlation pairs from kwargs
            correlation_pairs = kwargs.get('correlation_pairs', [(0, 1), (0, 2), (1, 2)])
            feat_new = gene_correlation(adata, hyperedges, correlation_pairs)
        else:
            raise ValueError(f'Feature "{feature}" not supported')
        
        # concatenate the features
        if feat is None:
            feat = feat_new
        else:
            feat = torch.cat([feat, feat_new], dim=1)

    return feat