import argparse
import os
import sys
import math
import meld
import phate
import scprep
import numpy as np
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything

sys.path.insert(0, import_dir + '/src/')
from model.hypergraph_scattering import HypergraphScatteringNet
from train import prepare_dataloaders

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


@torch.no_grad()
def save_test_set_embeddings(model, test_loader, device, embedding_save_path):
    hyperedge_emb_arr, hyperedge_label_arr, hyperedge_gene_expression_arr = None, None, None
    hypergraph_emb_arr, hypergraph_label_arr, hypergraph_gene_expression_arr = None, None, None

    for data_item in tqdm(test_loader):
        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        embedding = model(
            x=data_item.x,
            hyperedge_index=data_item.edge_index,
            hyperedge_attr=data_item.edge_attr,
            batch=data_item.batch,
            return_wavelet_embeddings=True)

        hyperedge_emb = embedding.cpu().detach().numpy()
        # `hyperedge_emb.shape[0]` is number of hyperedges. Repeat the label for each hyperedge.
        hyperedge_label = y_true.cpu().detach().numpy().reshape(-1, 1).repeat(hyperedge_emb.shape[0], axis=0)
        hyperedge_gene_expression = data_item.x.cpu().detach().numpy()
        hypergraph_emb = embedding.cpu().detach().numpy().mean(axis=0, keepdims=True)
        hypergraph_label = y_true.cpu().detach().numpy().reshape(-1, 1)
        hypergraph_gene_expression = data_item.x.cpu().detach().numpy().mean(axis=0, keepdims=True)

        if hypergraph_emb_arr is None:
            hyperedge_emb_arr = hyperedge_emb
            hyperedge_label_arr = hyperedge_label
            hyperedge_gene_expression_arr = hyperedge_gene_expression
            hypergraph_emb_arr = hypergraph_emb
            hypergraph_label_arr = hypergraph_label
            hypergraph_gene_expression_arr = hypergraph_gene_expression
        else:
            hyperedge_emb_arr = np.concatenate((hyperedge_emb_arr, hyperedge_emb), axis=0)
            hyperedge_label_arr = np.concatenate((hyperedge_label_arr, hyperedge_label), axis=0)
            hyperedge_gene_expression_arr = np.concatenate((hyperedge_gene_expression_arr, hyperedge_gene_expression), axis=0)
            hypergraph_emb_arr = np.concatenate((hypergraph_emb_arr, hypergraph_emb), axis=0)
            hypergraph_label_arr = np.concatenate((hypergraph_label_arr, hypergraph_label), axis=0)
            hypergraph_gene_expression_arr = np.concatenate((hypergraph_gene_expression_arr, hypergraph_gene_expression), axis=0)

    with open(embedding_save_path, 'wb+') as f:
        np.savez(f,
                 hyperedge_emb_arr=hyperedge_emb_arr, hyperedge_label_arr=hyperedge_label_arr, hyperedge_gene_expression_arr=hyperedge_gene_expression_arr,
                 hypergraph_emb_arr=hypergraph_emb_arr, hypergraph_label_arr=hypergraph_label_arr, hypergraph_gene_expression_arr=hypergraph_gene_expression_arr)
    return

def visualize_test_set_embeddings(embedding_save_path, class_map, gene_list, hyperedge_sampling_rate: float = 0.1):
    with open(embedding_save_path, 'rb') as f:
        npzfile = np.load(f)
        hyperedge_emb_arr = npzfile['hyperedge_emb_arr']
        hyperedge_label_arr = npzfile['hyperedge_label_arr']
        hyperedge_gene_expression_arr = npzfile['hyperedge_gene_expression_arr']
        hypergraph_emb_arr = npzfile['hypergraph_emb_arr']
        hypergraph_label_arr = npzfile['hypergraph_label_arr']
        hypergraph_gene_expression_arr = npzfile['hypergraph_gene_expression_arr']

    # NOTE: Plot visualization where each node is a hypergraph (figure 1) or a hyperedge (figure 2).
    for entity_name, fig_title, embedding_label_expression in \
        zip(['hypergraph', 'hyperedge'],
            ['Each node is a hypergraph.', 'Each node is a hyperedge.'],
            [(hypergraph_emb_arr, hypergraph_label_arr, hypergraph_gene_expression_arr),
             (hyperedge_emb_arr, hyperedge_label_arr, hyperedge_gene_expression_arr)]):

        embedding_arr, label_arr, gene_expression_arr = embedding_label_expression

        if entity_name == 'hyperedge':
            if args.dataset == 'placenta':
                gene_name_list = [item.split('_')[1] for item in gene_list]
                plot_histogram_for_genes(gene_name_list, gene_expression_arr, ['FGF2', 'FGFR1', 'FN1', 'KRT8'])
            else:
                gene_name_list = gene_list

        # Subsample the hyperedges, otherwise it gives OOM.
        if entity_name == 'hyperedge' and len(embedding_arr) > 1e6 and hyperedge_sampling_rate is not None:
            sampled_indices = np.random.choice(len(embedding_arr), size=int(hyperedge_sampling_rate * len(embedding_arr)))
            embedding_arr = embedding_arr[sampled_indices, :]
            label_arr = label_arr[sampled_indices, :]
            gene_expression_arr = gene_expression_arr[sampled_indices, :]

        phate_op = phate.PHATE(random_state=args.random_seed, n_jobs=args.num_workers, verbose=True)
        data_phate = phate_op.fit_transform(normalize(embedding_arr, axis=1))

        fig = plt.figure(figsize=(12, 8))
        for class_idx, class_name in class_map.items():
            ax = fig.add_subplot(2, len(class_map.items()), class_idx + 1)
            color_arr = label_arr == class_idx
            scprep.plot.scatter2d(
                data_phate,
                c=color_arr,
                cmap='coolwarm',
                ax=ax,
                title=class_name,
                xticks=False,
                yticks=False,
                label_prefix='PHATE',
                fontsize=10,
                s=5,
                alpha=0.5)
        fig.suptitle(fig_title, fontsize=20)
        fig.tight_layout(pad=2)
        fig.savefig(os.path.join(os.path.dirname(embedding_save_path), entity_name + '_embeddings.png'))

        meld_op = meld.MELD(random_state=args.random_seed, n_jobs=args.num_workers, verbose=True)
        sample_densities = meld_op.fit_transform(normalize(embedding_arr, axis=1), sample_labels=label_arr)
        sample_likelihoods = meld.utils.normalize_densities(sample_densities)
        for class_idx, class_name in class_map.items():
            ax = fig.add_subplot(2, len(class_map.items()), len(class_map.items()) + class_idx + 1)
            color_arr = sample_likelihoods[class_idx]
            scprep.plot.scatter2d(
                data_phate,
                c=color_arr,
                cmap='coolwarm',
                ax=ax,
                title=class_name+' (MELD)',
                xticks=False,
                yticks=False,
                label_prefix='PHATE',
                fontsize=10,
                s=5,
                alpha=0.5)
        fig.tight_layout(pad=2)
        fig.savefig(os.path.join(os.path.dirname(embedding_save_path), entity_name + '_embeddings.png'), dpi=300)
        plt.close(fig)

        # Auto-determine layout (rows x cols) to be roughly square
        num_plots = len(gene_list)
        if args.dataset == 'placenta':
            gene_name_list = [item.split('_')[1] for item in gene_list]
        else:
            gene_name_list = gene_list
        idx_sorted = np.argsort(gene_name_list)
        gene_indices_sorted = np.arange(len(gene_name_list))[idx_sorted]
        gene_name_list_sorted = np.array(gene_name_list)[idx_sorted]

        chunk_size = 25  # At most 25 subplots per figure.
        indices_by_chunk = [np.arange(num_plots)[i:i + chunk_size] for i in range(0, num_plots, chunk_size)]

        for chunk_idx, indices in enumerate(indices_by_chunk):
            num_subplots = len(indices)
            cols = math.ceil(math.sqrt(num_subplots))
            rows = math.ceil(num_subplots / cols)
            fig = plt.figure(figsize=(cols * 5, rows * 4))

            gene_indices_chunk = gene_indices_sorted[indices]
            gene_name_list_chunk = gene_name_list_sorted[indices]

            for subplot_idx, (gene_idx, gene_name) in enumerate(zip(gene_indices_chunk, gene_name_list_chunk)):

                gene_expression = gene_expression_arr[:, gene_idx][:, None]

                ax = fig.add_subplot(rows, cols, subplot_idx + 1)

                vmax = np.percentile(gene_expression, 90)
                if vmax == 0:
                    vmax = np.max(gene_expression)

                scprep.plot.scatter2d(
                    data_phate,
                    c=gene_expression,
                    cmap='coolwarm',
                    vmin=0,
                    vmax=vmax,
                    ax=ax,
                    title=gene_name,
                    xticks=False,
                    yticks=False,
                    label_prefix='PHATE',
                    colorbar=True,
                    fontsize=16,
                    s=5,
                    alpha=0.25)
                ax.set_axis_off()

            fig.tight_layout(pad=2)
            fig.savefig(os.path.join(os.path.dirname(embedding_save_path), entity_name + f'_gene_expressions_{str(chunk_idx).zfill(3)}.png'), dpi=100)
            plt.close(fig)

    return

def plot_histogram_for_genes(gene_name_list, gene_expression_arr, gene_names_to_plot) -> None:
    num_subplots = len(gene_names_to_plot)
    cols = math.ceil(math.sqrt(num_subplots))
    rows = math.ceil(num_subplots / cols)
    fig = plt.figure(figsize=(cols * 5, rows * 4))

    for subplot_idx, gene_name in enumerate(gene_names_to_plot):
        gene_index = np.argwhere(np.array(gene_name_list) == gene_name).item()
        gene_expression = gene_expression_arr[:, gene_index]

        ax = fig.add_subplot(rows, cols, subplot_idx + 1)
        ax.hist(gene_expression, bins=256, color='#00356B')
        ax.set_yscale('log')  # Apply log scale to the y-axis
        ax.set_title(gene_name)
        ax.set_ylabel('Count')
        ax.set_xlabel('Expression')

    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(os.path.dirname(embedding_save_path), f'gene_expressions_histogram.png'), dpi=100)
    plt.close(fig)
    return


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Entry point.')
    args.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    args.add_argument('--trainable-scales', action='store_true')
    args.add_argument('--k-hop', default=3, type=int)
    args.add_argument('--num-workers', default=8, type=int)
    args.add_argument('--random-seed', default=1, type=int)
    args.add_argument('--dataset', default='placenta', type=str)
    args.add_argument('--data-folder', default='$ROOT/data/spatial_placenta_accreta/patchified_selected_genes', type=str)
    args.add_argument('--num-features', default=212, type=int)  # number of genes or features

    args = args.parse_known_args()[0]
    args.batch_size = 1
    seed_everything(args.random_seed)

    # Update paths with absolute path.
    args.data_folder = args.data_folder.replace('$ROOT', ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data.
    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(args)
    class_map = test_loader.dataset.dataset.class_map
    gene_list = test_loader.dataset.dataset.gene_list

    model = HypergraphScatteringNet(
        in_channels=64,
        hidden_channels=16,
        out_channels=num_classes,
        num_features=args.num_features,
        trainable_laziness=False,
        trainable_scales=args.trainable_scales,
        activation=None,  # just get one layer of wavelet transform
        fixed_weights=True,
        layout=['hsm'],
        normalize='right',
        pooling='attention',
        scale_list=[0,1,2,4]
    )
    model.eval()
    model.to(device)

    subset_name = os.path.basename(args.data_folder.rstrip('/'))
    current_run_identifier = f'dataset-{args.dataset}-{subset_name}_kHop-{args.k_hop}_features-{args.num_features}_trainable_scales-{args.trainable_scales}_seed-{args.random_seed}'
    model_save_path = os.path.join(ROOT_DIR, 'results', args.dataset, current_run_identifier, 'model.pt')
    embedding_save_path = os.path.join(ROOT_DIR, 'results', args.dataset, current_run_identifier, 'embeddings.npz')
    os.makedirs(os.path.dirname(embedding_save_path), exist_ok=True)

    model.eval()
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    if not os.path.isfile(embedding_save_path):
        save_test_set_embeddings(model, test_loader, device, embedding_save_path)

    visualize_test_set_embeddings(embedding_save_path, class_map, gene_list=gene_list)