from typing import Dict, List, Tuple
import cv2
import os
import json
from glob import glob
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from scipy import sparse

import warnings
warnings.filterwarnings("ignore")

folder_in = '../../data/spatial_placenta_accreta/raw/'
folder_out = '../../data/spatial_placenta_accreta/patchified_celltype/'
NUM_BINS = 100
MIN_PIXEL_PER_GRAPH = 20

GENES_BY_CELL_TYPE = {
    'Cytotrophoblast': ['KRT7', 'STMN1', 'PARP1', 'PAGE4', 'GATA3', 'KRT8', 'SPINT1'],
    'Syncytiotrophoblast': ['CSH2', 'INHA', 'HSD3B1', 'ESR1', 'PGR', 'CD274', 'PSG4', 'ERVFRD-1', 'LGALS16', 'GDF15',
                            'INSL4', 'CGA', 'CYP19A1', 'TFPI'],
    'EVT': ['KRT8', 'HSD3B1', 'CSH2', 'CCNE1', 'MCAM', 'MUC4', 'ASCL2', 'ITGA5', 'ITGB1', 'INHA', 'PAPPA2', 'CDH5'],
    'smooth-muscle-Endothelial': ['PECAM1', 'CDH5', 'CD34', 'KDR', 'IFI27', 'VWF'],
    'Lymphatic-Endothelial': ['TFF3'],
    'Hoffbauer': ['CD163', 'LYVE1', 'VSIG4', 'MRC1', 'HPGDS', 'CD14'],
    'Mesenchymal': ['COL1A1', 'TAGLN', 'LUM', 'APOD', 'DCN', 'ACTA2'],
    'Fibroblasts': ['COL1A1', 'TAGLN', 'LUM', 'DCN'],
    'B-cell': ['CD79A'],
    'T-cell': ['CD3D'],
    'NK': ['KLRB1'],
    'Monocyte': ['CD14', 'FCGR3A'],
    'Plasma': ['XBP1', 'IGHA1', 'IGHA2'],
    'Decidua': ['PRL', 'FCGR3A', 'IGFBP1', 'ITGAX', 'CCNA1', 'RB1', 'CDK1', 'DKK1', 'WNT4'],
    'Myometrial': ['ACTA2', 'CNN1', 'OXTR'],
}


def infer_cell_type(gene_matrix: sparse._csr.csr_matrix,
                    marker_gene_dict: Dict,
                    gene_to_index: Dict,
                    threshold: float = 0) -> Tuple[sparse._csr.csr_matrix, List[str]]:
    '''
    Infer the cell types for each cell from `gene_matrix`, a cell-by-gene matrix.
    In this sub-cellular spatial-seq data, it's actually a pixel-by-gene matrix,
    but the principle is the same.

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

    Returns:
    --------
    cell_type_matrix : sparse._csr.csr_matrix
        Binary matrix (n_cells x n_cell_types + 1) indicating cell type assignments.
        Last column represents unassigned cells (zero expression or below threshold).
    '''

    # Get cell type names
    cell_types = list(marker_gene_dict.keys())
    n_cells = gene_matrix.shape[0]
    n_cell_types = len(cell_types)

    # Initialize score matrix (cells x cell_types)
    scores = np.zeros((n_cells, n_cell_types))

    # Calculate scores for each cell type
    for cell_type_idx, cell_type in enumerate(cell_types):
        marker_genes = marker_gene_dict[cell_type]

        # Find indices of marker genes that exist in our data
        marker_indices = [gene_to_index[gene] for gene in marker_genes if gene in gene_to_index]

        # Extract expression of marker genes for all cells and sum
        marker_expression = gene_matrix[:, marker_indices]
        cell_type_scores = np.array(marker_expression.sum(axis=1)).flatten()
        scores[:, cell_type_idx] = cell_type_scores

    # Assign cell types based on highest scores
    max_scores = np.max(scores, axis=1)
    cell_type_assignments = np.argmax(scores, axis=1)

    # Set to -1 (unassigned) if below or equal to threshold
    cell_type_assignments[max_scores <= threshold] = -1

    # Create binary assignment matrix (always n_cell_types + 1 columns)
    cell_type_matrix = np.zeros((n_cells, n_cell_types + 1), dtype=np.uint8)
    for i in range(n_cells):
        if cell_type_assignments[i] >= 0:
            cell_type_matrix[i, cell_type_assignments[i]] = 1
        else:
            cell_type_matrix[i, -1] = 1  # Unassigned category

    cell_type_names = cell_types + ['Unassigned']
    return sparse.csr_matrix(cell_type_matrix), cell_type_names



if __name__ == '__main__':
    # Get all genes of interest.
    celltype_related_genes = np.unique(sum(GENES_BY_CELL_TYPE.values(), []))

    # Find the folders for pixel-by-gene matrices and the corresponding spatial images.
    all_folder_paths = sorted(glob(os.path.join(folder_in, '0*', 'filtered_feature_bc_matrix')))
    all_image_paths = sorted(glob(os.path.join(folder_in, '0*', 'spatial', 'tissue_hires_image.png')))
    assert len(all_folder_paths) == len(all_image_paths)

    filtered_folder_names, filtered_folders, filtered_image_paths = [], [], []
    for folder_path, image_path in zip(all_folder_paths, all_image_paths):
        if 'normal_placenta' in folder_path or 'PAS' in folder_path or 'insufficient' in folder_path:
            assert ('normal_placenta' in folder_path) + ('PAS' in folder_path) + ('insufficient' in folder_path) == 1
            filtered_folders.append(folder_path)
            filtered_image_paths.append(image_path)
        if 'normal_placenta' in folder_path:
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_normal_placenta')
        elif 'PAS' in folder_path:
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_PAS')
        elif 'insufficient' in folder_path:
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_insufficient')
    del all_folder_paths

    for source_mat_folder, source_image_path, target_folder in tqdm(zip(filtered_folders, filtered_image_paths, filtered_folder_names),
                                                                    total=len(filtered_folders)):
        matrix = ad.io.read_mtx(os.path.join(source_mat_folder, 'matrix.mtx'))
        barcodes = pd.read_csv(os.path.join(source_mat_folder, 'barcodes.tsv'), header=None, sep="\t")
        features = pd.read_csv(os.path.join(source_mat_folder, 'features.tsv'), header=None, sep="\t")
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

        barcode_position = barcodes.merge(tissue_position_info, on='barcode', how='inner')

        # NOTE: Filter pixels outside the image.
        row_arr, col_arr = barcode_position['pixel_row_in_highres'].to_numpy(), barcode_position['pixel_col_in_highres'].to_numpy()
        valid_rows = np.logical_and(row_arr > 0, row_arr < image.shape[0])
        valid_cols = np.logical_and(col_arr > 0, col_arr < image.shape[1])
        barcode_position_in_image = np.logical_and(valid_rows, valid_cols)
        assert barcode_position.shape[0] == barcode_position_in_image.shape[0]

        # NOTE: Only keep selected genes of interest.
        celltype_related_matrix = matrix.X.T[:, celltype_related_feature_indices]

        # Apply filtering.
        barcode_position = barcode_position[barcode_position_in_image]
        barcode_position['pixel_row_in_highres'] = np.floor(barcode_position['pixel_row_in_highres']).astype(int)
        barcode_position['pixel_col_in_highres'] = np.floor(barcode_position['pixel_col_in_highres']).astype(int)
        celltype_related_matrix = celltype_related_matrix[barcode_position_in_image, :]

        celltype_label_matrix, cell_type_names = infer_cell_type(celltype_related_matrix, GENES_BY_CELL_TYPE, gene_to_index)

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
