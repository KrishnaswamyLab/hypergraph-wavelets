import os
import pickle
import matplotlib.pyplot as plt
import wandb
import hydra
import yaml
import numpy as np
import pandas as pd
import torch
import scipy.sparse
from scipy.spatial.distance import pdist, squareform
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
import sys
from ae import ContrastDistAE, WeightedContrastiveLoss
from ae_dataset import train_valid_loader_from_pc
# sys.path.append('../../dmae/src')
sys.path.append('/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/hypergraph_scattering/dmae/src/')
from transformations import LogTransform, NonTransform, StandardScaler, \
    MinMaxScaler, PowerTransformer, KernelTransform
# from model import AEDist, VAEDist
from metrics import distance_distortion, mAP
from procrustes import Procrustes

def to_dense_array(X):
    if scipy.sparse.issparse(X):  # Check if X is a sparse matrix
        return X.toarray()
    elif isinstance(X, np.ndarray):  # Check if X is already a numpy array
        return X
    elif isinstance(X, pd.DataFrame):  # Check if X is a pandas DataFrame
        return X.values  # or X.to_numpy()
    else:
        raise TypeError("Input is neither a sparse matrix, a numpy array, nor a pandas DataFrame")


def load_data(cfg, load_all=False):
    # if load_all:
    #     data_path = os.path.join(cfg.data.root, cfg.data.name + "_all" + cfg.data.filetype)
    # else:
    #     data_path = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    data_path = os.path.join(cfg.data.root, cfg.data.name + cfg.data.filetype)
    data = np.load(data_path, allow_pickle=True)
    # sanity check the data is not empty
    # assert 'data' in data.files and 'phate' in data.files and 'colors' in data.files \
        # and 'dist' in data.files, "Some required files are missing in the 'data' variable."
    X = data['data']
    phate_coords = data['phate']
    colors = data['colors']
    dist = data['dist']
    train_mask = data['is_train']
    xor_dist = data['xor_dist']
    if not load_all:
        X = X[train_mask,:]
        phate_coords = phate_coords[train_mask,:]
        colors = colors[train_mask]
        dist = dist[train_mask,:][:,train_mask]
        if len(xor_dist.shape) == 3:
            xor_dist[:,train_mask,:][:,:,train_mask]
        else:
            xor_dist = xor_dist[train_mask,:][:,train_mask]
    assert X.shape[0] == phate_coords.shape[0] == colors.shape[0] == dist.shape[0], \
        "The number of cells in the data, phate, and colors variables do not match."

    if cfg.training.match_potential:
        phate_D = dist
    else:
        phate_D = squareform(pdist(phate_coords))
    
    if load_all:
        NotImplemented
        # allloader = dataloader_from_pc(
        # X, # <---- Pointcloud
        # phate_D, # <---- Distance matrix to match
        # batch_size=X.shape[0],
        # shuffle=False,)
        # return allloader, None, X, phate_coords, colors, dist, pp
    else:
        trainloader, valloader = train_valid_loader_from_pc(
            X, # <---- Pointcloud
            phate_D, # <---- Distance matrix to match
            xor_dist, # <---- contrastive matrix to match
            batch_size=cfg.training.batch_size,
            train_valid_split=cfg.training.train_valid_split,
            shuffle=cfg.training.shuffle,
            seed=cfg.training.seed,)
        return trainloader, valloader, X, phate_coords, colors, dist

def make_model(cfg, dim, emb_dim, dist_std=1, from_checkpoint=False, checkpoint_path=None):
    cfg.model.dim = dim
    cfg.model.emb_dim = emb_dim
    if from_checkpoint:
        model = ContrastDistAE.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **cfg.model,
        )
    else:
        model = ContrastDistAE(
            **cfg.model,
        )
    return model

def prep_data_model(cfg):
    trainloader, valloader, X, phate_coords, colors, dist = load_data(cfg)
    model = make_model(cfg, X.shape[1], cfg.model.emb_dim)
    return model, trainloader, valloader, X, phate_coords, colors, dist

@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    if cfg.logger.use_wandb:
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run = wandb.init(
            entity=cfg.logger.entity,
            project=cfg.logger.project,
            tags=cfg.logger.tags,
            reinit=True,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )
    # print config:
    print(OmegaConf.to_yaml(cfg))

    model, trainloader, valloader, X, phate_coords, colors, dist = prep_data_model(cfg)

    early_stopping = EarlyStopping(cfg.training.monitor, patience=cfg.training.patience)
    if cfg.logger.use_wandb:
        logger = WandbLogger()
        checkpoint_callback = ModelCheckpoint(
            dirpath=wandb.run.dir,  # Save checkpoints in wandb directory
            save_top_k=1,  # Save the top 1 model
            monitor='val_loss',  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )
    else:
        logger = TensorBoardLogger(save_dir=os.path.join(cfg.path.root, cfg.path.log))
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.path.root,  # Save checkpoints in wandb directory
            filename=cfg.path.model,
            save_top_k=1,
            monitor='val_loss',  # Model selection based on validation loss
            mode='min'  # Minimize validation loss
        )
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        callbacks=[early_stopping,checkpoint_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
    )    
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
    )

    X_tensor = torch.from_numpy(X).float()
    x_hat, emb_z = model(X_tensor)
    x_hat = x_hat.cpu().detach().numpy()
    emb_z = emb_z.cpu().detach().numpy()

    procrustes = Procrustes()
    pc_s, z, disparity = procrustes.fit_transform(phate_coords, emb_z)
    if cfg.path.save:
        with open(os.path.join(cfg.path.root, f'{cfg.path.procrustes}.pkl'), 'wb') as file:
            pickle.dump(procrustes, file)

    if cfg.logger.use_wandb:
        wandb.log({'procrustes_disparity_latent': disparity})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(phate_coords[:,0], phate_coords[:,1], c=colors, s=1, cmap='Spectral')
    ax1.set_title('PHATE')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.scatter(z[:,0], z[:,1], c=colors, s=1, cmap='Spectral')
    ax2.set_title('Latent Space')
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle('Comparison of PHATE and Latent Space')
    
    if cfg.path.save:
        plotdir = os.path.join(cfg.path.root, cfg.path.plots)
        os.makedirs(plotdir, exist_ok=True)
        plt.savefig(f'{plotdir}/comparison_latent.pdf', dpi=300)

    if cfg.logger.use_wandb:
        wandb.log({'Comparison Plot Latent': plt})

    procrustes = Procrustes()
    xo, xh, disparity = procrustes.fit_transform(X, x_hat)
    if cfg.logger.use_wandb:
        wandb.log({'procrustes_disparity_reconstruction': disparity})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(xo[:,0], xo[:,1], c=colors, s=1, cmap='Spectral')
    ax1.set_title('PHATE')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.scatter(xh[:,0], xh[:,1], c=colors, s=1, cmap='Spectral')
    ax2.set_title('Latent Space')
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle('Comparison of PHATE and Latent Space')
    
    if cfg.path.save:
        plotdir = os.path.join(cfg.path.root, cfg.path.plots)
        os.makedirs(plotdir, exist_ok=True)
        plt.savefig(f'{plotdir}/comparison_reconstr.pdf', dpi=300)

    if cfg.logger.use_wandb:
        wandb.log({'Comparison Plot Reconstruction': plt})
        dist_distort = distance_distortion(dist, squareform(pdist(emb_z)))
        wandb.log({'distance_distortion': dist_distort})
        # TODO mAP score needs input graph.
        run.finish()

if __name__ == '__main__':
    main()