"""
GSAE, but with contrastive loss and dist matching losses

"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

import sys
# sys.path.append('../../dmae/src')
sys.path.append('/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/hypergraph_scattering/dmae/src/')
from model import AEDist
from transformations import NonTransform

# class ContrastiveLoss(torch.nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
    
#     # label is binary (shape: (batch_size, n_markers))
#     # Contrastive loss
#     def forward(self, distance, label):
#         # Contrastive loss
#         loss = (1 - label) * 0.5 * (distance ** 2) + \
#                label * 0.5 * ((torch.max(distance.new_zeros(distance.size()), self.margin - distance)) ** 2)
#         return loss.mean()


class WeightedContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super(WeightedContrastiveLoss, self).__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, distance, xor_dists):
        # they are in the shape of (*, 0.5*N*(N-1)) (after np.triu_indices, or pdist)
        # especially, distance is shape (0.5*N*(N-1)), while xor_dists is shape (n_markers, 0.5*N*(N-1))
        # where N is batch_size.
        label = ~xor_dists
        # Contrastive loss
        # Positive pairs (label == 0)
        pos_loss = self.pos_weight * (1 - label) * 0.5 * (distance ** 2)

        # Negative pairs (label == 1)
        neg_loss = self.neg_weight * label * 0.5 * (torch.clamp(self.margin - distance, min=0.0) ** 2)

        return torch.mean(pos_loss + neg_loss)

# class ContrastDistAE(AEDist):
#     # TODO make this more general, so that it can take arbitrary number of distances.
#     def __init__(
#         self,
#         dim,
#         emb_dim,
#         layer_widths=[128, 64, 32],
#         activation_fn=torch.nn.ReLU(),
#         dist_weights_within=[0.5, 0.5], # the weights within the distances
#         dist_reg_weight = 0.25,
#         reconstr_weight = 0.25,
#         contra_loss_weight = 0.25,
#         dist_reconstr_weight = 0.,
#         contra_margin=1.0,
#         pos_weight=1.0,
#         neg_weight=1.0,
#         dist_recon_topk_coords=None,
#         pp=NonTransform(),
#         eps=1e-10,
#         lr=1e-3,
#     ):
#         super().__init__(dim, emb_dim, layer_widths=layer_widths, activation_fn=activation_fn, dist_reconstr_weights=[0,0,0], dist_recon_topk_coords=dist_recon_topk_coords, pp=pp, eps=eps, lr=lr)
#         self.dist_reg_weight = dist_reg_weight
#         self.reconstr_weight = reconstr_weight
#         self.contra_loss_weight = contra_loss_weight
#         self.dist_reconstr_weight = dist_reconstr_weight
#         self.dist_weights_within = dist_weights_within
#         assert self.dist_reg_weight + self.reconstr_weight + self.dist_reconstr_weight + self.dist2_reg_weight + self.contra_loss_weight > 0.0
#         self.contrastive_loss = WeightedContrastiveLoss(margin=contra_margin, pos_weight=pos_weight, neg_weight=neg_weight)

#     def step(self, batch, batch_idx):
#         x = batch['x']
#         d = batch['d'] # shape (n_dist_channels, 0.5*B*(B-1))
#         xor_d = batch['m'] # shape (n_markers, 0.5*B*(B-1))
#         input = [x, d, xor_d]
#         loss = self.loss_function(input, self.forward(x))
#         return loss
    
#     """
#     input is list of [x, dist, xor_dist]
#     label is binary (shape: (batch_size, n_markers))
#     """
#     def loss_function(self, input, output):
#         # TODO can refactor to make arbitrary number of distances using tensor operation.
#         x, dist, xor_dists = input
#         loss = 0.0
#         x_hat, z = output
#         dist_emb = torch.nn.functional.pdist(z)
#         dist_emb = self.pp.transform(dist_emb) # assume the ground truth dist is transformed.
#         if self.dist_reg_weight > 0.0:
#             dl = self.dist_loss(dist_emb, dist)
#             self.log('dist_loss', dl, prog_bar=True, on_epoch=True)
#             loss += self.dist1_reg_weight * dl
#         if self.dist_reconstr_weight > 0.0:
#             # only use top k dimensions for distance, to save computation. 
#             # This makes sense only if the input is PCA loadings.
#             # TODO compute and transform the original distance before training, to speed up!
#             dist_orig = torch.nn.functional.pdist(x[:, :self.dist_recon_topk_coords])
#             dist_reconstr = torch.nn.functional.pdist(x_hat[:, :self.dist_recon_topk_coords])
#             dist_orig = self.pp.transform(dist_orig)
#             dist_reconstr = self.pp.transform(dist_reconstr)
#             drl = self.dist_loss(dist_reconstr, dist_orig)
#             self.log('dist_reconstr_loss', drl, prog_bar=True, on_epoch=True)
#             loss += self.dist_reconstr_weight * drl
#         if self.reconstr_weight > 0.0:
#             rl = torch.nn.functional.mse_loss(x, x_hat)
#             self.log('reconstr_loss', rl, prog_bar=True, on_epoch=True)
#             loss += self.reconstr_weight * rl
#         if self.contra_loss_weight > 0.0:
#             cl = self.contrastive_loss(dist_emb, xor_dists)
#             self.log('contra_loss', cl, prog_bar=True, on_epoch=True)
#             loss += self.contra_loss_weight * cl
#         return loss
    
#     """
#     modified to support multiple channels of dist.
#     """
#     def dist_loss(self, dist_emb, dist_gt):
#         # dist_emb = self.pp.transform(dist_emb)
#         # dist_gt = self.pp.transform(dist_gt) # it is already transformed!
#         dist_emb = dist_emb.unsqueeze(0).expand(dist_gt.shape[0], -1)
#         weights = self.dist_weights_within.unsqueeze(1)
#         loss = (weights * (dist_emb - dist_gt) ** 2).mean()
#         return loss

class ContrastDistAE(AEDist):
    # TODO make this more general, so that it can take arbitrary number of distances.
    def __init__(
        self,
        dim,
        emb_dim,
        layer_widths=[64, 64, 64],
        activation_fn='relu',
        dist_reconstr_weights=[0.9, 0.1, 0.0],
        dist_recon_topk_coords=None,
        pp=NonTransform(),
        eps=1e-10,
        lr=1e-3,
        weight_decay=0.0,
        dropout=0.0,
        batch_norm=False,
        use_dist_mse_decay=False,
        dist_mse_decay=0.1,
        cycle_weight=0.,
        cycle_dist_weight=0.,
        dist_contr_recon_weights=[0.3,0.3,0.3],
        # dist_reg_weight = 0.25,
        # reconstr_weight = 0.25,
        # contra_loss_weight = 0.25,
        dist_reconstr_weight = 0.,
        contra_margin=1.0,
        pos_weight=1.0,
        neg_weight=1.0,
    ):
        super().__init__(
            dim,
            emb_dim,
            layer_widths=layer_widths,
            activation_fn=torch.nn.ReLU(),
            dist_reconstr_weights=dist_reconstr_weights,
            dist_recon_topk_coords=dist_recon_topk_coords,
            pp=NonTransform(),
            eps=eps,
            lr=lr,
            weight_decay=weight_decay,
            batch_norm=batch_norm,
            dropout=dropout,
            use_dist_mse_decay=use_dist_mse_decay,
            dist_mse_decay=dist_mse_decay,
        )
        # super().__init__(dim, emb_dim, layer_widths=layer_widths, activation_fn=activation_fn, dist_reconstr_weights=dist_reconstr_weights, dist_recon_topk_coords=dist_recon_topk_coords, pp=pp, eps=eps, lr=lr, weight_decay=weight_decay, batch_norm=batch_norm,dropout=dropout,use_dist_mse_decay=use_dist_mse_decay) #dist_reconstr_weights is dummy
        self.dist_reg_weight = dist_contr_recon_weights[0]
        self.contra_loss_weight = dist_contr_recon_weights[1]
        self.reconstr_weight = dist_contr_recon_weights[2]
        self.dist_reconstr_weight = dist_reconstr_weight
        assert self.dist_reg_weight + self.reconstr_weight + self.dist_reconstr_weight + self.contra_loss_weight > 0.0
        self.contrastive_loss = WeightedContrastiveLoss(margin=contra_margin, pos_weight=pos_weight, neg_weight=neg_weight)

    def step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d'] # shape (n_dist_channels, 0.5*B*(B-1))
        xor_d = batch['m'] # shape (n_markers, 0.5*B*(B-1))
        input = [x, d, xor_d]
        loss = self.loss_function(input, self.forward(x), stage)
        return loss
    
    """
    input is list of [x, dist, xor_dist]
    label is binary (shape: (batch_size, n_markers))
    """
    def loss_function(self, input, output, stage):
        # TODO can refactor to make arbitrary number of distances using tensor operation.
        x, dist, xor_dists = input
        loss = 0.0
        x_hat, z = output
        dist_emb = torch.nn.functional.pdist(z)
        dist_emb = self.pp.transform(dist_emb) # assume the ground truth dist is transformed.
        if self.dist_reg_weight > 0.0:
            dl = self.dist_loss(dist_emb, dist)
            self.log(f'{stage}/dist_loss', dl, prog_bar=True, on_epoch=True)
            loss += self.dist_reg_weight * dl
        if self.dist_reconstr_weight > 0.0:
            # only use top k dimensions for distance, to save computation. 
            # This makes sense only if the input is PCA loadings.
            # TODO compute and transform the original distance before training, to speed up!
            dist_orig = torch.nn.functional.pdist(x[:, :self.dist_recon_topk_coords])
            dist_reconstr = torch.nn.functional.pdist(x_hat[:, :self.dist_recon_topk_coords])
            dist_orig = self.pp.transform(dist_orig)
            dist_reconstr = self.pp.transform(dist_reconstr)
            drl = self.dist_loss(dist_reconstr, dist_orig)
            self.log(f'{stage}/dist_reconstr_loss', drl, prog_bar=True, on_epoch=True)
            loss += self.dist_reconstr_weight * drl
        if self.reconstr_weight > 0.0:
            rl = torch.nn.functional.mse_loss(x, x_hat)
            self.log(f'{stage}/reconstr_loss', rl, prog_bar=True, on_epoch=True)
            loss += self.reconstr_weight * rl
        if self.contra_loss_weight > 0.0:
            cl = self.contrastive_loss(dist_emb, xor_dists)
            self.log(f'{stage}/contra_loss', cl, prog_bar=True, on_epoch=True)
            loss += self.contra_loss_weight * cl
        return loss
    
    # """
    # modified to support multiple channels of dist.
    # """
    # def dist_loss(self, dist_emb, dist_gt):
    #     # dist_emb = self.pp.transform(dist_emb)
    #     # dist_gt = self.pp.transform(dist_gt) # it is already transformed!
    #     dist_emb = dist_emb.unsqueeze(0).expand(dist_gt.shape[0], -1)
    #     weights = self.dist_weights_within.unsqueeze(1)
    #     loss = (weights * (dist_emb - dist_gt) ** 2).mean()
    #     return loss