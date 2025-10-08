import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

dataset_name = 'spatial_placenta_accreta_16um'


def plot_statistics(csv_path: str,
                    output_figure_path: str) -> None:
    '''
    Plot the statistics.
    1. Number of cells with gene expression out of total, per sample.
    2. Distribution of cell types per sample and per disease.
    '''
    df = pd.read_csv(csv_path)

    disease_order = ['insufficient', 'normal', 'PAS']
    # Convert to categorical with specific order
    df['disease_name'] = pd.Categorical(df['disease_name'], categories=disease_order, ordered=True)

    cell_type_cols = [col for col in df.columns if col.startswith('count_')]
    cell_type_labels = [col.replace('count_', '') for col in cell_type_cols]
    expression_cols = [col for col in df.columns if col.startswith('mean_expression_')]
    cell_type_labels_2 = [col.replace('mean_expression_', '') for col in expression_cols]
    assert cell_type_labels == cell_type_labels_2

    sample_ids = df['batch_index'].astype(str)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    fig = plt.figure(figsize=(28, 10))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(cell_type_cols))]

    # Subplot 1: Cell type distribution per sample (stacked bar)
    ax = fig.add_subplot(2, 3, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    bottoms = np.zeros(len(df))
    for i, col in enumerate(cell_type_cols):
        ax.bar(sample_ids, df[col], bottom=bottoms, color=colors[i], label=cell_type_labels[i], alpha=0.8)
        bottoms += df[col].values

    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Cell Count")
    ax.set_ylim([0, bottoms.max() * 1.05])
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)

    # Subplot 2: Cell type distribution per disease (stacked bar with total count)
    ax = fig.add_subplot(2, 3, 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    disease_grouped = df.groupby('disease_name')[cell_type_cols].mean()
    disease_names = disease_grouped.index.tolist()
    bottoms = np.zeros(len(disease_grouped))

    for i, col in enumerate(cell_type_cols):
        ax.bar(disease_names, disease_grouped[col], bottom=bottoms, color=colors[i], label=cell_type_labels[i], alpha=0.8)
        bottoms += disease_grouped[col].values

    ax.set_xlabel("Disease")
    ax.set_ylabel("Total Cell Count")
    ax.set_ylim([0, bottoms.max() * 1.05])
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)

    # Subplot 3: Same as subplot 2, but showing percentage rather than counts.
    ax = fig.add_subplot(2, 3, 3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    disease_grouped = df.groupby('disease_name')[cell_type_cols].mean()
    disease_totals = disease_grouped.sum(axis=1)  # Sum across all cell types for each disease
    disease_percentages = disease_grouped.div(disease_totals, axis=0) * 100  # Convert to percentages

    disease_names = disease_percentages.index.tolist()
    bottoms = np.zeros(len(disease_percentages))

    for i, col in enumerate(cell_type_cols):
        ax.bar(disease_names, disease_percentages[col], bottom=bottoms,
            color=colors[i], label=cell_type_labels[i], alpha=0.8)

        for j, (_, percentage) in enumerate(disease_percentages[col].items()):
            y_pos = bottoms[j] + percentage / 2
            ax.text(j + 0.05, y_pos, f'{percentage:.1f}%', ha='right', va='center', fontsize=5)

        bottoms += disease_percentages[col].values

    ax.set_xlabel("Disease")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.set_ylim([0, bottoms.max() * 1.05])
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)

    # Subplot 4: Mean expression, by batch
    ax = fig.add_subplot(2, 3, 4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    expression_matrix = np.stack([df[col].to_numpy() for col in expression_cols], axis=0)
    assert expression_matrix.shape == (len(cell_type_labels), len(sample_ids))
    im = ax.imshow(expression_matrix, cmap='viridis')
    ax.set_xticks(np.arange(len(sample_ids)))
    ax.set_xticklabels(sample_ids)
    ax.set_xlabel('Batch Index')
    ax.set_yticks(np.arange(len(cell_type_labels)))
    ax.set_yticklabels(cell_type_labels, fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean Expression of Marker Genes', fontsize=14)

    # Subplot 5: Mean expression, by disease type
    ax = fig.add_subplot(2, 3, 5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    disease_grouped = df.groupby('disease_name')[expression_cols].mean()
    disease_names = disease_grouped.index.tolist()
    expression_matrix = np.stack([disease_grouped[col].to_numpy() for col in expression_cols], axis=0)
    assert expression_matrix.shape == (len(cell_type_labels), len(disease_names))
    im = ax.imshow(expression_matrix, cmap='viridis', aspect=0.33)
    ax.set_xticks(np.arange(len(disease_names)))
    ax.set_xticklabels(disease_names)
    ax.set_xlabel('Disease')
    ax.set_yticks(np.arange(len(cell_type_labels)))
    ax.set_yticklabels(cell_type_labels, fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean Expression of Marker Genes', fontsize=14)

    plt.tight_layout(pad=2)
    plt.savefig(output_figure_path, dpi=300)
    plt.close()
    return

def plot_selected_expressions(npz_path: str,
                              output_figure_path: str) -> None:
    npz_file = np.load(npz_path)
    disease_name_arr = npz_file['disease_name']
    gene_name_arr = npz_file['gene_names']
    cell_type_name_arr = npz_file['cell_type_names']
    gene_by_celltype_matrices = npz_file['gene_by_celltype_matrices']

    disease_order = ['insufficient', 'normal', 'PAS']

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    fig = plt.figure(figsize=(18, 5))

    # Expression of some genes, by disease type
    celltypes_of_interest = ['Decidual-cells', 'Myometrial-cells', 'Extravillous-trophoblast']
    genes_of_interest = ['CXCL12', 'NRG1', 'CXCR4', 'ERBB2', 'ERBB3', 'ERBB4', 'ITGA5', 'ITGAV', 'ITGB1', 'ITGB3']
    gene_indices = np.array([np.argwhere(gene_name_arr == gene).item() for gene in genes_of_interest])
    expression_by_celltype_by_disease = []

    for disease in disease_order:
        expression_by_celltype = gene_by_celltype_matrices[np.argwhere(disease_name_arr == disease).flatten()][:, gene_indices, :].mean(axis=0)
        expression_by_celltype_by_disease.append(expression_by_celltype)
    expression_by_celltype_by_disease = np.stack(expression_by_celltype_by_disease, axis=-1)

    for fig_idx, celltype in enumerate(celltypes_of_interest):
        ax = fig.add_subplot(1, 3, fig_idx + 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        celltype_idx = np.argwhere(cell_type_name_arr == celltype).item()
        expression_matrix = expression_by_celltype_by_disease[:, celltype_idx, :]
        assert expression_matrix.shape == (len(genes_of_interest), len(disease_order))

        im = ax.imshow(expression_matrix, cmap='viridis', aspect=0.33)
        ax.set_xticks(np.arange(len(disease_order)))
        ax.set_xticklabels(disease_order)
        ax.set_xlabel('Disease')
        ax.set_yticks(np.arange(len(genes_of_interest)))
        ax.set_yticklabels(genes_of_interest, fontsize=10)
        ax.set_title(' '.join(celltype.split('-')), fontsize=16)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Mean Expression of Marker Genes', fontsize=14)

    fig.tight_layout(pad=2)
    fig.savefig(output_figure_path, dpi=300)
    plt.close()
    return


def plot_selected_top_genes(npz_path: str,
                            output_figure_path: str,
                            topk: int = 100) -> None:
    '''
    For each of the cell types,
    1. Compute the mean expression of all genes (done).
    2. Compute the variance of mean expression over disease type (variance of 3 values).
    3. Rank order them and find the 100 genes with highest variance.
    4. Plot them in heatmap.
    '''
    npz_file = np.load(npz_path)
    disease_name_arr = npz_file['disease_name']
    gene_name_arr = npz_file['gene_names']
    cell_type_name_arr = npz_file['cell_type_names']
    gene_by_celltype_matrices = npz_file['gene_by_celltype_matrices']

    disease_order = ['insufficient', 'normal', 'PAS']

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    fig = plt.figure(figsize=(30, 20))

    # Expression of some genes, by disease type
    celltypes_of_interest = ['Cytotrophoblasts', 'Syncytiotrophoblast', 'Extravillous-trophoblast', 'Decidual-cells']
    expression_by_celltype_by_disease = []

    for disease in disease_order:
        expression_by_celltype = gene_by_celltype_matrices[np.argwhere(disease_name_arr == disease).flatten()].mean(axis=0)
        expression_by_celltype_by_disease.append(expression_by_celltype)
    expression_by_celltype_by_disease = np.stack(expression_by_celltype_by_disease, axis=-1)

    for fig_idx, celltype in enumerate(celltypes_of_interest):
        ax = fig.add_subplot(1, 4, fig_idx + 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        celltype_idx = np.argwhere(cell_type_name_arr == celltype).item()
        expression_matrix = expression_by_celltype_by_disease[:, celltype_idx, :]

        # Find the top K variable genes.
        gene_variance = expression_matrix.var(axis=-1)
        topk_idx = np.argsort(gene_variance)[-topk:][::-1]

        gene_name_topk = gene_name_arr[topk_idx]
        expression_matrix_topk = expression_matrix[topk_idx, :]
        assert expression_matrix_topk.shape == (topk, len(disease_order))

        im = ax.imshow(expression_matrix_topk, cmap='viridis', aspect=0.1)
        ax.set_xticks(np.arange(len(disease_order)))
        ax.set_xticklabels(disease_order)
        ax.set_xlabel('Disease')
        ax.set_yticks(np.arange(len(gene_name_topk)))
        ax.set_yticklabels(gene_name_topk, fontsize=10)
        ax.set_title(' '.join(celltype.split('-')), fontsize=24)
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, aspect=60)
        cbar.set_label('Mean Expression of Marker Genes', fontsize=14)

    fig.tight_layout(pad=2)
    fig.savefig(output_figure_path, dpi=300)
    plt.close()
    return


if __name__ == '__main__':
    plot_statistics(csv_path=f'./{dataset_name}/dataset_statistics.csv',
                    output_figure_path=f'./{dataset_name}/dataset_statistics.png')

    plot_selected_expressions(npz_path=f'./{dataset_name}/dataset_all_expressions.npz',
                              output_figure_path=f'./{dataset_name}/dataset_selected_expressions.png')

    plot_selected_top_genes(npz_path=f'./{dataset_name}/dataset_all_expressions.npz',
                            output_figure_path=f'./{dataset_name}/dataset_selected_top_genes.png')