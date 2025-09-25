from typing import Dict, List, Tuple
import cv2
import os
import json
from glob import glob
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from tqdm import tqdm
from scipy import sparse
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import warnings
warnings.filterwarnings("ignore")

dataset_name = 'spatial_placenta_accreta_16um'
folder_in = f'../../data/spatial_placenta_accreta_16um/raw/'
folder_out = f'../../data/spatial_placenta_accreta_16um/patchified_celltype/'
NUM_BINS = 30
MIN_PIXEL_PER_GRAPH = 15

GENES_BY_CELL_TYPE = {
    'Cytotrophoblasts': ['LARGE2', 'LGR5', 'LRP2', 'SLC22A11', 'SLC13A3', 'SLC16A12', 'PEG10', 'NFE2L3'],
    'Decidual-cells': ['RBP4', 'EPYC', 'SERPINA3', 'PRL', 'CHRDL1', 'CA12', 'SCARA5', 'DKK1', 'ALDH1A2', 'NDP', 'CHI3L2'],
    'Extravillous-trohpoblast': ['DIO2', 'LAMA3', 'NOG', 'ASCL2', 'PLAC8', 'FSTL3', 'LY6D', 'COL17A1', 'NOTUM', 'PRG2'],
    'Endothelial-cells-1': ['APLN', 'AREG', 'WNT3A', 'EGFL7', 'MMRN2', 'AGTR1', 'COX4I2', 'LRRC36'],
    'Endothelial-cells-2': ['CADM3', 'RSPO2', 'CTHRC1', 'PROM1', 'WNT2', 'SLC16A10', 'MATN2', 'COL8A2',
                            'PITX2'],
    'Hofbauer-cells': ['RGS1', 'CTSW', 'DUSP2', 'CCL5', 'CD96', 'GBP5', 'CCL4', 'C1QC', 'FCGBP', 'SCN9A', 'FGL1',
                       'CD28', 'GRIN2C', 'STAB1', 'LPAR5', 'C3AR1'],
    'Mixed-immune-cells': ['IGKC', 'IGHG1', 'DES', 'CNN1', 'ACTG2', 'PAEP', 'TNC', 'MMP12', 'PCP4'],
    'Smooth-muscle-cells': ['CCL21', 'MMRN1', 'FHL5', 'LCN6', 'LCN10', 'CCL14', 'RELN', 'SULF1', 'TBX1', 'CPE',
                            'HOXD9', 'THBS2', 'IGFBP7'],
    'Syncytiotrophoblast': ['ACOXL', 'IGHA1', 'TCHH', 'GH2', 'TRIM40', 'CSH2', 'PSG7', 'PSG4', 'ALPP', 'CYP19A1',
                            'LEP', 'PSG6', 'SDC1', 'MFSD2A'],
}


def infer_cell_type(gene_matrix: sparse._csr.csr_matrix,
                    marker_gene_dict: Dict,
                    gene_to_index: Dict,
                    threshold: float = 0.0,
                    batch_name: str = '',
                    fig_pc_save_path: str = None,
                    fig_spatial_save_path: str = None,
                    spatial_location: pd.DataFrame = None,
                    overlay_image: np.ndarray = None) -> Tuple[sparse._csr.csr_matrix, List[str]]:
    '''
    Infer the cell types for each cell from `gene_matrix`, a cell-by-gene matrix.
    In this sub-cellular spatial-seq data, it's actually a pixel-by-gene matrix,
    but the principle is the same.

    We will first perform Leiden clustering on the cells, and then infer
    the cell type in each clustering using the marker genes.

    `gene_matrix` does not need to be normalized if `threshold` is 0.

    Parameters:
    -----------
    gene_matrix : sparse._csr.csr_matrix
        Cell-by-gene expression matrix (n_cells x n_genes)
    marker_gene_dict : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker gene names
    gene_to_index : Dict[str, int]
        Dictionary mapping gene names to the column indices in `gene_matrix`.
    threshold : float, optional
        Minimum score threshold for cell type assignment
    fig_save_path : str, optional
        If provided, will plot a visualization to this path.
    fig_spatial_save_path : str, optional
        If provided, will plot a visualization to this path.
    spatial_location : pd.DataFrame, optional
        DataFrame with fields 'X' and 'Y'. Required if `fig_spatial_save_path` is provided.
    overlay_image : np.ndarray, optional
        Original H&E image for overlay visualization.

    Returns:
    --------
    cell_type_matrix : sparse._csr.csr_matrix
        Binary matrix (n_cells x n_cell_types + 1) indicating cell type assignments.
        Last column represents unassigned cells (zero expression or below threshold).
    cell_type_names : List[str]
        Name of each cell type.
    '''

    # Get cell type names
    cell_types = sorted(list(marker_gene_dict.keys()))
    n_cells = gene_matrix.shape[0]
    n_cell_types = len(cell_types)

    # Construct a AnnData object, and normalize gene expressions.
    adata = ad.AnnData(X=gene_matrix, obs=pd.DataFrame({'cell_id': np.arange(n_cells)}))
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    # Run k-NN and Leiden clustering.
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10, method='umap')  # 'umap' here means UMap's fast k-NN algorithm.
    sc.tl.leiden(adata, resolution=0.1)

    # Calculate cluster-level marker scores.
    cluster_labels = adata.obs['leiden'].astype(int)
    unique_clusters = np.unique(cluster_labels)

    cluster_cell_type_assignment = {}

    for cluster_id in unique_clusters:
        cluster_mask = (cluster_labels == cluster_id).values  # Convert to numpy array
        cluster_scores = np.zeros(n_cell_types)

        for cell_type_idx, cell_type in enumerate(cell_types):
            marker_genes = marker_gene_dict[cell_type]
            marker_indices = [gene_to_index[gene] for gene in marker_genes]
            nonmarker_indices = list(set(np.arange(adata.X.shape[1])) - set(marker_indices))

            # Calculate mean expression of marker genes in this cluster.
            cluster_marker_expression = np.mean(adata.X[cluster_mask][:, marker_indices])
            cluster_nonmarker_expression = np.mean(adata.X[cluster_mask][:, nonmarker_indices])
            cluster_scores[cell_type_idx] = cluster_marker_expression - cluster_nonmarker_expression

        # Assign cluster to cell type with highest score, check threshold
        max_score = np.max(cluster_scores)
        if max_score > threshold:
            cluster_cell_type_assignment[cluster_id] = np.argmax(cluster_scores)
        else:
            cluster_cell_type_assignment[cluster_id] = -1  # Unassigned

    # Create final assignments.
    final_assignments = np.array([cluster_cell_type_assignment[cluster_id] for cluster_id in cluster_labels])

    # Create binary matrix.
    cell_type_matrix = np.zeros((n_cells, n_cell_types + 1), dtype=np.uint8)
    for i in range(n_cells):
        if final_assignments[i] >= 0:
            cell_type_matrix[i, final_assignments[i]] = 1
        else:
            cell_type_matrix[i, -1] = 1  # Unassigned category

    cell_type_names = cell_types + ['Unassigned']

    if fig_pc_save_path is not None:
        os.makedirs(os.path.dirname(fig_pc_save_path), exist_ok=True)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)

        pca_coords = adata.obsm['X_pca'][:, :2]  # First 2 PCA components
        final_assignment_labels = [cell_type_names[i] if i >= 0 else 'Unassigned' for i in final_assignments]
        unique_labels = sorted(list(set(final_assignment_labels)))

        cmap = plt.get_cmap("Paired")
        colors = [cmap(i % cmap.N) for i in range(len(unique_labels))]
        for label, color in zip(unique_labels, colors):
            mask = np.array(final_assignment_labels) == label
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                       c=[color], label=label, alpha=0.5, s=20)
        ax.set_xlabel('PC1', fontsize=18)
        ax.set_ylabel('PC2', fontsize=18)
        ax.set_title(batch_name, fontsize=20)
        ax.legend(fontsize=12, markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout(pad=2)
        fig.savefig(fig_pc_save_path, dpi=300)
        plt.close()

    if fig_spatial_save_path is not None:
        os.makedirs(os.path.dirname(fig_spatial_save_path), exist_ok=True)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)

        cmap = plt.get_cmap("Paired")
        colors = [cmap(i % cmap.N) for i in range(len(unique_labels))]
        for label, color in zip(unique_labels, colors):
            mask = np.array(final_assignment_labels) == label
            ax.scatter(spatial_location['Y'][mask],
                       spatial_location['X'][mask],
                       c=[color], label=label, alpha=0.5, s=0.1)
        ax.set_xlabel('Spatial Y', fontsize=18)
        ax.set_ylabel('Spatial X', fontsize=18)
        ax.set_ylim(spatial_location['Y'].max(), 0)
        ax.set_title(batch_name, fontsize=20)
        ax.legend(fontsize=12, markerscale=30, bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout(pad=2)
        fig.savefig(fig_spatial_save_path, dpi=300, bbox_inches='tight')
        plt.close()

        for opacity in [0.2, 0.5, 0.8]:

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=12)
            if overlay_image is not None:
                ax.imshow(overlay_image)

            cmap = plt.get_cmap("Paired")
            colors = [cmap(i % cmap.N) for i in range(len(unique_labels))]
            for label, color in zip(unique_labels, colors):
                mask = np.array(final_assignment_labels) == label
                ax.scatter(spatial_location['pixel_col_in_highres'][mask],
                           spatial_location['pixel_row_in_highres'][mask],
                           c=[color], label=label, alpha=opacity, s=0.1)
            ax.set_xlabel('Spatial Y', fontsize=18)
            ax.set_ylabel('Spatial X', fontsize=18)
            ax.set_ylim(spatial_location['pixel_row_in_highres'].max(), 0)
            ax.set_title(batch_name, fontsize=20)
            ax.legend(fontsize=12, markerscale=30, bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout(pad=2)
            fig.savefig(fig_spatial_save_path.replace('.png', f'_opacity-{opacity}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if overlay_image is not None:
            ax.imshow(overlay_image)

        cmap = plt.get_cmap("Paired")
        colors = [cmap(i % cmap.N) for i in range(len(unique_labels))]
        for label, color in zip(unique_labels, colors):
            if label != "Unassigned":
                continue
            mask = np.array(final_assignment_labels) == label
            ax.scatter(spatial_location['pixel_col_in_highres'][mask],
                       spatial_location['pixel_row_in_highres'][mask],
                       c=[color], label=label, alpha=0.8, s=0.1)
        ax.set_xlabel('Spatial Y', fontsize=18)
        ax.set_ylabel('Spatial X', fontsize=18)
        ax.set_ylim(spatial_location['pixel_row_in_highres'].max(), 0)
        ax.set_title(batch_name, fontsize=20)
        ax.legend(fontsize=12, markerscale=30, bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout(pad=2)
        fig.savefig(fig_spatial_save_path.replace('.png', f'_unassigned.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return sparse.csr_matrix(cell_type_matrix), cell_type_names


def visualize_gene_expression_profiles(gene_matrix: sparse._csr.csr_matrix,
                                       cell_type_matrix: sparse._csr.csr_matrix,
                                       cell_type_names: List[str],
                                       marker_gene_dict: Dict,
                                       gene_to_index: Dict,
                                       batch_name: str = '',
                                       save_path: str = None,
                                       figsize: Tuple[int, int] = (16, 16)) -> None:
    """
    Visualize gene expression profiles across different cell types using a heatmap.

    Parameters:
    -----------
    gene_matrix : sparse._csr.csr_matrix
        Normalized gene expression matrix (n_cells x n_genes)
    cell_type_matrix : sparse._csr.csr_matrix
        Binary cell type assignment matrix (n_cells x n_cell_types)
    cell_type_names : List[str]
        Names of cell types (including 'Unassigned')
    marker_gene_dict : Dict[str, List[str]]
        Dictionary mapping cell type names to marker genes
    gene_to_index : Dict[str, int]
        Mapping from gene names to column indices
    save_path : str, optional
        Path to save the figure
    figsize : Tuple[int, int]
        Figure size for the heatmap
    """
    # Create block-diagonal gene ordering (marker genes grouped by cell type)
    ordered_genes = []
    gene_block_boundaries = []  # Track where each cell type's block starts/ends

    # Add marker genes for each cell type in order
    for cell_type in cell_type_names:
        if cell_type == 'Unassigned' or cell_type not in marker_gene_dict:
            continue

        block_start = len(ordered_genes)
        marker_genes = marker_gene_dict[cell_type]
        for gene in marker_genes:
            if gene in gene_to_index and gene not in ordered_genes:
                ordered_genes.append(gene)

        block_end = len(ordered_genes)
        gene_block_boundaries.append((cell_type, block_start, block_end))

    assert len(ordered_genes) == len(gene_to_index)

    # Normalize gene expression
    adata_temp = ad.AnnData(X=gene_matrix)
    sc.pp.normalize_total(adata_temp, target_sum=1e6)
    sc.pp.log1p(adata_temp)
    normalized_matrix = adata_temp.X

    # Calculate mean expression for each gene in each cell type
    n_genes = len(ordered_genes)
    n_cell_types = len(cell_type_names)
    mean_expression_gene_by_celltype = np.zeros((n_genes, n_cell_types))
    pixel_count_by_celltype = np.zeros(n_cell_types)

    for ct_idx, cell_type in enumerate(cell_type_names):
        cell_mask = cell_type_matrix[:, ct_idx].toarray().flatten().astype(bool)

        if cell_mask.sum() > 0:  # If there are cells of this type
            cell_expressions = normalized_matrix[cell_mask, :]
            mean_expressions = np.array(cell_expressions.mean(axis=0)).flatten()
            pixel_count_by_celltype[ct_idx] = cell_mask.sum()

            # Reorder according to new gene ordering
            for new_idx, gene in enumerate(ordered_genes):
                old_idx = gene_to_index[gene]
                mean_expression_gene_by_celltype[new_idx, ct_idx] = mean_expressions[old_idx]

    xlabel_lines = []
    for ct_idx, cell_type in enumerate(cell_type_names):
        expr_count = mean_expression_gene_by_celltype.mean(0)[ct_idx]
        pixel_count = pixel_count_by_celltype[ct_idx]
        xlabel_lines.append(f'{cell_type}\nmean expr. {expr_count:.1e}' + r'$\times$' + f'{pixel_count:.1e} cells')

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Create heatmap
    im = ax.imshow(mean_expression_gene_by_celltype, cmap='viridis', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(range(n_cell_types))
    ax.set_xticklabels(xlabel_lines, rotation=45, ha='center', fontsize=10)
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(ordered_genes, fontsize=8)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Log-normalized Expression', fontsize=14)

    # Add rectangles to highlight marker gene blocks
    for cell_type, block_start, block_end in gene_block_boundaries:
        if cell_type in cell_type_names:
            ct_col_idx = cell_type_names.index(cell_type)

            # Highlight the diagonal block for this cell type
            block_height = block_end - block_start
            if block_height > 0:
                rect = Rectangle((ct_col_idx-0.5, block_start-0.5), 1, block_height,
                               linewidth=2, edgecolor='firebrick', facecolor='none')
                ax.add_patch(rect)

    ax.set_xlabel('Cell Types', fontsize=18)
    ax.set_ylabel('Genes', fontsize=18)
    ax.set_title(batch_name + '\n1. total-count normalization s.t. each pixel has one million expressions. 2. log transform.', pad=12, fontsize=18)

    fig.tight_layout(pad=2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close()

    return


def quantify_statistics(batch_index: str,
                        disease_name: str,
                        pixel_count: int,
                        cell_type_counts: np.ndarray,
                        cell_type_names: List[str],
                        csv_path: str) -> None:
    '''
    Quantify the statistics and save in a csv file.
    1. Number of cells with gene expression out of total, per sample.
    2. Distribution of cell types per sample and per disease.
    '''
    # Total number of pixels with cell type assigned
    pixel_assigned_count = cell_type_counts[:-1].sum()

    # Build the row dictionary
    row_data = {
        'batch_index' : batch_index,
        'disease_name': disease_name,
        'pixel_count': pixel_count,
        'pixel_assigned_count': pixel_assigned_count
    }

    # Add cell type counts
    for name, count in zip(cell_type_names, cell_type_counts):
        row_data[f'count_{name}'] = count

    # Convert to DataFrame with one row
    df_new = pd.DataFrame([row_data])

    # If file exists, append. Otherwise, create with header.
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)

    return


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'sans-serif'

    # Get all genes of interest.
    celltype_related_genes = np.unique(sum(GENES_BY_CELL_TYPE.values(), []))

    # Find the folders for pixel-by-gene matrices and the corresponding spatial images.
    all_folder_paths = sorted(glob(os.path.join(folder_in, '*', 'filtered_feature_bc_matrix')))
    all_image_paths = sorted(glob(os.path.join(folder_in, '*', 'spatial', 'tissue_hires_image.png')))
    assert len(all_folder_paths) == len(all_image_paths)

    filtered_folder_names, filtered_folders, filtered_image_paths = [], [], []
    for folder_path, image_path in zip(all_folder_paths, all_image_paths):
        if 'normal' in folder_path or 'PAS' in folder_path or 'insufficient' in folder_path:
            assert ('normal' in folder_path) + ('PAS' in folder_path) + ('insufficient' in folder_path) == 1
            filtered_folders.append(folder_path)
            filtered_image_paths.append(image_path)
        if 'normal' in folder_path:
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_normal_placenta')
        elif 'PAS' in folder_path:
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_PAS')
        elif 'insufficient' in folder_path:
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_insufficient')
    del all_folder_paths

    for source_mat_folder, source_image_path, target_folder in tqdm(zip(filtered_folders, filtered_image_paths, filtered_folder_names),
                                                                    total=len(filtered_folders)):
        batch_index = source_mat_folder.split('/')[-2].split('_')[0]
        matrix = ad.io.read_mtx(os.path.join(source_mat_folder, 'matrix.mtx.gz'))
        barcodes = pd.read_csv(os.path.join(source_mat_folder, 'barcodes.tsv.gz'), header=None, sep="\t")
        features = pd.read_csv(os.path.join(source_mat_folder, 'features.tsv.gz'), header=None, sep="\t")

        # Assure the selected genes are all available.
        for gene_name in celltype_related_genes:
            assert gene_name in features[1].tolist()

        # Only take the selected genes.
        celltype_related_feature_indices = features[1].isin(celltype_related_genes).to_numpy()
        celltype_related_features = features[celltype_related_feature_indices]
        gene_to_index = {}
        for idx, key in enumerate(celltype_related_features[1].keys()):
            # Note that the indices need to be "restarted consecutively from 0",
            # while the `keys` are w.r.t. the original numbering, hence the "enumerate".
            gene_to_index[celltype_related_features[1][key]] = idx

        barcodes['barcode'] = barcodes[0]
        barcodes = barcodes.drop(0, axis=1)

        barcodes[['X', 'Y']] = barcodes['barcode'].str.extract(r's_\d+um_(\d+)_(\d+)-\d')
        barcodes['X'] = barcodes['X'].astype(int)
        barcodes['Y'] = barcodes['Y'].astype(int)

        # Image and info.
        image = cv2.cvtColor(cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        scale_factor_path = source_image_path.replace('tissue_hires_image.png', 'scalefactors_json.json')
        with open(scale_factor_path) as f:
            scale_factor_dict = json.load(f)
        scale_factor = scale_factor_dict['tissue_hires_scalef']

        # Load the tissue position information.
        tissue_position_info = pd.read_parquet(os.path.join(os.path.dirname(source_image_path), 'tissue_positions.parquet'))
        # Assert the barcode file matches with all pixels that are inside the tissue.
        assert len(barcodes) == tissue_position_info.in_tissue.sum()

        tissue_position_info['pixel_row_in_highres'] = tissue_position_info['pxl_row_in_fullres'] * scale_factor
        tissue_position_info['pixel_col_in_highres'] = tissue_position_info['pxl_col_in_fullres'] * scale_factor

        # NOTE: Only keep selected genes of interest.
        celltype_related_matrix = matrix.X.T[:, celltype_related_feature_indices]

        # NOTE: Filter pixels outside the image.
        barcode_position = barcodes.merge(tissue_position_info, on='barcode', how='inner')
        row_arr, col_arr = barcode_position['pixel_row_in_highres'].to_numpy(), barcode_position['pixel_col_in_highres'].to_numpy()
        valid_rows = np.logical_and(row_arr > 0, row_arr < image.shape[0])
        valid_cols = np.logical_and(col_arr > 0, col_arr < image.shape[1])
        barcode_position_in_image = np.logical_and(valid_rows, valid_cols)
        assert barcode_position.shape[0] == barcode_position_in_image.shape[0]
        barcode_position = barcode_position[barcode_position_in_image]
        barcode_position['pixel_row_in_highres'] = np.floor(barcode_position['pixel_row_in_highres']).astype(int)
        barcode_position['pixel_col_in_highres'] = np.floor(barcode_position['pixel_col_in_highres']).astype(int)
        celltype_related_matrix = celltype_related_matrix[barcode_position_in_image, :]
        pixel_count = barcode_position.shape[0]

        celltype_label_matrix, cell_type_names = infer_cell_type(
            celltype_related_matrix,
            GENES_BY_CELL_TYPE,
            gene_to_index,
            batch_name=target_folder,
            fig_pc_save_path=f'./{dataset_name}/vis_celltype/{target_folder}_pc.png',
            fig_spatial_save_path=f'./{dataset_name}/vis_celltype/{target_folder}_spatial.png',
            spatial_location=barcode_position[['X', 'Y', 'pixel_row_in_highres', 'pixel_col_in_highres']],
            overlay_image=image)

        visualize_gene_expression_profiles(
            gene_matrix=celltype_related_matrix,
            cell_type_matrix=celltype_label_matrix,
            cell_type_names=cell_type_names,
            marker_gene_dict=GENES_BY_CELL_TYPE,
            gene_to_index=gene_to_index,
            batch_name=target_folder,
            save_path=f'./{dataset_name}/vis_celltype/{target_folder}_expression_heatmap.png'
        )

        # if 'normal' in source_mat_folder:
        #     disease_name = 'normal'
        # elif 'PAS' in source_mat_folder:
        #     disease_name = 'PAS'
        # elif 'insufficient' in source_mat_folder:
        #     disease_name = 'insufficient'

        # quantify_statistics(batch_index=batch_index,
        #                     disease_name=disease_name,
        #                     pixel_count=pixel_count,
        #                     cell_type_counts=celltype_label_matrix.toarray().sum(axis=0),
        #                     cell_type_names=cell_type_names,
        #                     csv_path=f'./{dataset_name}/dataset_statistics.csv')

        # # Subset the data by spatial location.
        # cell_bins = pd.DataFrame({'pixel_row_bin': pd.cut(barcode_position['pixel_row_in_highres'], bins=NUM_BINS, labels=False, include_lowest=True),
        #                           'pixel_col_bin': pd.cut(barcode_position['pixel_col_in_highres'], bins=NUM_BINS, labels=False, include_lowest=True),
        #                           'pixel_row_in_highres': barcode_position['pixel_row_in_highres'],
        #                           'pixel_col_in_highres': barcode_position['pixel_col_in_highres'],
        #                           'cell_index': np.arange(len(barcode_position))})

        # # Iterate over groups and save them separately.
        # iterator_bins = cell_bins.groupby(['pixel_row_bin', 'pixel_col_bin'])
        # for (row_bin, col_bin), group in tqdm(sorted(iterator_bins), total=len(iterator_bins)):
        #     # Extract pixels corresponding to this group.
        #     indices = group['cell_index'].values
        #     if len(indices) < MIN_PIXEL_PER_GRAPH:
        #         print(f'Bin ({row_bin}, {col_bin}) has fewer than {MIN_PIXEL_PER_GRAPH} pixels ({len(indices)}). Skipping this bin.')
        #         continue

        #     sub_matrix = celltype_label_matrix[indices, :]
        #     sub_adata = ad.AnnData(X=sub_matrix, obs=pd.DataFrame({'Location': group['cell_index']}), var=pd.DataFrame({'Cell Types': cell_type_names}))
        #     coords = np.concatenate((group['pixel_row_in_highres'].values[:, None], group['pixel_col_in_highres'].values[:, None]), axis=1)
        #     sub_adata.obsm['spatial'] = coords

        #     os.makedirs(folder_out, exist_ok=True)
        #     sub_adata.write(os.path.join(folder_out, f'{target_folder}_Bin-{str(row_bin).zfill(2)}-{str(col_bin).zfill(2)}_spatial_matrix.h5ad'))
