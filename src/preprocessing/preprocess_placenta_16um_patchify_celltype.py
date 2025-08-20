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

import warnings
warnings.filterwarnings("ignore")

dataset_name = 'spatial_placenta_accreta_16um'
folder_in = f'../../data/spatial_placenta_accreta_16um/raw/'
folder_out = f'../../data/spatial_placenta_accreta_16um/patchified_celltype/'
NUM_BINS = 30
MIN_PIXEL_PER_GRAPH = 15

GENES_BY_CELL_TYPE = {
    'Syncytiotrophoblast': ['ACOXL', 'IGHA1', 'TCHH', 'GH2', 'TRIM40', 'CSH2', 'PSG7', 'PSG4', 'ALPP', 'CYP19A1',
                            'LEP', 'PSG6', 'SDC1', 'MFSD2A'],
    'Cytotrophoblasts': ['LARGE2', 'LGR5', 'LRP2', 'SLC22A11', 'SLC13A3', 'SLC16A12', 'PEG10', 'NFE2L3'],
    'Extravillous_trohpoblast': ['DIO2', 'LAMA3', 'NOG', 'ASCL2', 'PLAC8', 'FSTL3', 'LY6D', 'COL17A1', 'NOTUM', 'PRG2'],
    'Endothelial-cells-1': ['APLN', 'AREG', 'WNT3A', 'EGFL7', 'MMRN2', 'AGTR1', 'COX4I2', 'LRRC36'],
    'Endothelial-cells-2': ['CADM3', 'RSPO2', 'CTHRC1', 'PROM1', 'WNT2', 'SLC16A10', 'MATN2', 'COL8A2',
                            'PITX2'],
    'Smooth-muscle-cells-1': ['RBP4', 'EPYC', 'SERPINA3', 'PRL', 'CHRDL1', 'CA12', 'SCARA5', 'DKK1',
                              'ALDH1A2', 'NDP', 'CHI3L2'],
    'Smooth-muscle-cells-2': ['CCL21', 'MMRN1', 'FHL5', 'LCN6', 'LCN10', 'CCL14', 'RELN', 'SULF1', 'TBX1', 'CPE',
                              'HOXD9', 'THBS2', 'IGFBP7'],
    'Mix-immune-cells': ['IGKC', 'IGHG1', 'DES', 'CNN1', 'ACTG2', 'PAEP', 'TNC', 'MMP12', 'PCP4'],
    'Hofbauer-cells': ['RGS1', 'CTSW', 'DUSP2', 'CCL5', 'CD96', 'GBP5', 'CCL4', 'C1QC', 'FCGBP', 'SCN9A', 'FGL1',
                       'CD28', 'GRIN2C', 'STAB1', 'LPAR5', 'C3AR1'],
}


def infer_cell_type(gene_matrix: sparse._csr.csr_matrix,
                    marker_gene_dict: Dict,
                    gene_to_index: Dict,
                    threshold: float = 0.1,
                    fig_pc_save_path: str = None,
                    fig_spatial_save_path: str = None,
                    spatial_location: pd.DataFrame = None) -> Tuple[sparse._csr.csr_matrix, List[str]]:
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
        DataFrame with fields 'X' and 'Y'. Required if fig_spatial_save_path is provided.

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
    sc.tl.leiden(adata, resolution=1.0)

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
                       c=[color], label=label, alpha=0.6, s=20)
        ax.set_xlabel('PC1', fontsize=18)
        ax.set_ylabel('PC2', fontsize=18)
        ax.set_title('Cell Type Assignments', fontsize=24)
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
            ax.scatter(spatial_location['X'][mask], spatial_location['Y'][mask],
                       c=[color], label=label, alpha=0.6, s=0.1)
        ax.set_xlabel('Spatial X', fontsize=18)
        ax.set_ylabel('Spatial Y', fontsize=18)
        ax.set_title('Cell Type Assignments', fontsize=24)
        ax.legend(fontsize=12, markerscale=30, bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout(pad=2)
        fig.savefig(fig_spatial_save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return sparse.csr_matrix(cell_type_matrix), cell_type_names


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
            fig_pc_save_path=f'./{dataset_name}/vis_celltype/{target_folder}_pc.png',
            fig_spatial_save_path=f'./{dataset_name}/vis_celltype/{target_folder}_spatial.png',
            spatial_location=barcode_position[['X', 'Y']])

        if 'normal' in source_mat_folder:
            disease_name = 'normal'
        elif 'PAS' in source_mat_folder:
            disease_name = 'PAS'
        elif 'insufficient' in source_mat_folder:
            disease_name = 'insufficient'

        quantify_statistics(batch_index=batch_index,
                            disease_name=disease_name,
                            pixel_count=pixel_count,
                            cell_type_counts=celltype_label_matrix.toarray().sum(axis=0),
                            cell_type_names=cell_type_names,
                            csv_path=f'./{dataset_name}/dataset_statistics.csv')

        # Subset the data by spatial location.
        cell_bins = pd.DataFrame({'pixel_row_bin': pd.cut(barcode_position['pixel_row_in_highres'], bins=NUM_BINS, labels=False, include_lowest=True),
                                  'pixel_col_bin': pd.cut(barcode_position['pixel_col_in_highres'], bins=NUM_BINS, labels=False, include_lowest=True),
                                  'pixel_row_in_highres': barcode_position['pixel_row_in_highres'],
                                  'pixel_col_in_highres': barcode_position['pixel_col_in_highres'],
                                  'cell_index': np.arange(len(barcode_position))})

        # Iterate over groups and save them separately.
        iterator_bins = cell_bins.groupby(['pixel_row_bin', 'pixel_col_bin'])
        for (row_bin, col_bin), group in tqdm(sorted(iterator_bins), total=len(iterator_bins)):
            # Extract pixels corresponding to this group.
            indices = group['cell_index'].values
            if len(indices) < MIN_PIXEL_PER_GRAPH:
                print(f'Bin ({row_bin}, {col_bin}) has fewer than {MIN_PIXEL_PER_GRAPH} pixels ({len(indices)}). Skipping this bin.')
                continue

            sub_matrix = celltype_label_matrix[indices, :]
            sub_adata = ad.AnnData(X=sub_matrix, obs=pd.DataFrame({'Location': group['cell_index']}), var=pd.DataFrame({'Cell Types': cell_type_names}))
            coords = np.concatenate((group['pixel_row_in_highres'].values[:, None], group['pixel_col_in_highres'].values[:, None]), axis=1)
            sub_adata.obsm['spatial'] = coords

            os.makedirs(folder_out, exist_ok=True)
            sub_adata.write(os.path.join(folder_out, f'{target_folder}_Bin-{str(row_bin).zfill(2)}-{str(col_bin).zfill(2)}_spatial_matrix.h5ad'))
