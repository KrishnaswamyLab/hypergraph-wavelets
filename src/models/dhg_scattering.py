import torch
import dhg 
import einops

# various toy implementations of scattering using the dhg library

def diffusion_no_lazy(hg: dhg.Hypergraph, X: torch.tensor):
    # X has shape num_nodes, num_features

    # propagate from nodes to hyperedges 
    inv_deg_v = hg.D_v_neg_1.values()
    inv_deg_v = torch.nan_to_num(inv_deg_v)
    # I should degree normalize first
    X_norm = torch.einsum('ij,i->ij', X, inv_deg_v)

    edge_feat = dhg.v2e(X_norm, aggr = 'add')
    # propagate back from hyperedges to nodes 
    inv_deg_e = hg.D_e_neg_1.values()
    inv_deg_e = torch.nan_to_num(inv_deg_e)
    edge_feat_norm = torch.einsum('ij,i->ij', edge_feat, inv_deg_e)

    node_feat = dhg.v2e(edge_feat_norm, agg = 'add')

    return node_feat, edge_feat 

