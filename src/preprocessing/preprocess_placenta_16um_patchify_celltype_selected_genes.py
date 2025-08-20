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
from preprocess_placenta_16um_patchify_celltype import GENES_BY_CELL_TYPE, infer_cell_type

import warnings
warnings.filterwarnings("ignore")

folder_in = '../../data/spatial_placenta_accreta_16um/raw/'
folder_out = '../../data/spatial_placenta_accreta_16um/patchified_celltype_selected_genes/'
NUM_BINS = 30
MIN_PIXEL_PER_GRAPH = 15


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

        celltype_label_matrix, cell_type_names = infer_cell_type(
            celltype_related_matrix,
            GENES_BY_CELL_TYPE,
            gene_to_index)

        # Normalize the gene expression for each cell.
        adata_genes = ad.AnnData(X=celltype_related_matrix, var=pd.DataFrame(index=celltype_related_features))
        sc.pp.normalize_total(adata_genes, target_sum=1e6)
        sc.pp.log1p(adata_genes)
        joint_matrix = sparse.hstack((adata_genes.X, celltype_label_matrix))
        joint_names = celltype_related_features[1].tolist() + cell_type_names

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

            sub_matrix = joint_matrix[indices, :]
            sub_adata = ad.AnnData(X=sub_matrix, obs=pd.DataFrame({'Location': group['cell_index']}), var=pd.DataFrame({'Expression': joint_names}))
            coords = np.concatenate((group['pixel_row_in_highres'].values[:, None], group['pixel_col_in_highres'].values[:, None]), axis=1)
            sub_adata.obsm['spatial'] = coords

            os.makedirs(folder_out, exist_ok=True)
            sub_adata.write(os.path.join(folder_out, f'{target_folder}_Bin-{str(row_bin).zfill(2)}-{str(col_bin).zfill(2)}_spatial_matrix.h5ad'))
