import cv2
import os
import json
from glob import glob
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

folder_in = '../../data/spatial_placenta_accreta/raw/'
folder_out = '../../data/spatial_placenta_accreta/patchified_selected_genes/'
NUM_BINS = 100
MIN_PIXEL_PER_GRAPH = 20

GENES_BY_CELL_TYPE = {
    'Cytotrophoblast': ['KRT7', 'STMN1', 'PARP1', 'PAGE4', 'GATA3', 'KRT8', 'SPINT1'],
    'Syncytiotrophoblast': ['CSH2', 'INHA', 'HSD3B1', 'ESR1', 'PGR', 'CD274', 'PSG4', 'ERVFRD-1', 'LGALS16', 'GDF15',
                            'INSL4', 'CGA', 'CYP19A1', 'TFPI'],
    'EVT': ['KRT8', 'HSD3B1', 'CSH2', 'CCNE1', 'MCAM', 'MUC4', 'ASCL2', 'ITGA5', 'ITGB1', 'INHA', 'PAPPA2', 'CDH5'],
    'smooth_muscle_Endothelial': ['PECAM1', 'CDH5', 'CD34', 'KDR', 'IFI27', 'VWF'],
    'Lymphatic_Endothelial': ['TFF3'],
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

GENES_BY_FUNCTION = {
    'EMT_TGF': ['TGFB1', 'TGFB2', 'TGFB3', 'INHA', 'INHBA', 'INHHBB', 'INHBC', 'INHBE', 'NODAL', 'MSTN', 'BMP1',
                'BMP2', 'BMP3', 'BMP4', 'BMP5', 'BMP6', 'BMP7', 'BMP8A', 'BMP8B', 'GDF2', 'BMP10', 'GDF3', 'GDF5',
                'GDF6', 'GDF7', 'GDF9', 'BMP15', 'GDF10', 'GDF11', 'GDF15', 'AMH', 'LEFTY2', 'LEFTY1'],
    'EMT_FGF': ['FGF1', 'FGF2', 'FGF3', 'FGF7', 'FGF10', 'FGF22', 'FGF22', 'FGF4', 'FGF5', 'FGF6', 'FGF8', 'FGF17',
                'FGF18', 'FGF9', 'FGF16', 'FGF20', 'FGF19', 'FGF21', 'FGF23', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4'],
    'EMT_EGF': ['EGF', 'EGFR'],
    'EMT_HIF': ['HIF1A', 'ARNT', 'EPAS1', 'ARNT2', 'HIF3A', 'ARNT3'],
    'EMT_other': ['SNAI1', 'SNAI2', 'SNAI3', 'ZEB1', 'ZEB2', 'TWIST1', 'TWIST2', 'RREB1', 'SMAD', 'CDH1', 'VIM', 'FSP1',
                  'STAT3', 'FOXC2', 'ITGA5', 'VCAN', 'COL3A1', 'COL5A1', 'MSN', 'FN1', 'WNT5B', 'JAG1', 'NOTCH1',
                  'MCL1', 'CXCR4', 'CXCL12', 'ADAM19', 'ADAM12', 'CTNNB1', 'MSX2', 'SERPINF1', 'CSH1', 'HLA-G',
                  'ERVFRD-1', 'SIGLEC6', 'CLDN1', 'ANK3', 'MARVELD3', 'OCLN', 'KRT19', 'ITGB4', 'ITGA5', 'ANK3'],
    'EMT_MMP': ['MMP2', 'MMP9', 'MMP13', 'MMP14', 'MMP11', 'MMP21'],
    'Hypoxia': ['DUSP1', 'NOX4', 'PLOD', 'BHLHE40', 'VCAM1', 'RND3', 'TXNIP', 'SLC21', 'SLC2A3', 'HK2', 'LDHA', 'LDHC',
                'G6PD', 'TALDO1', 'GAPDH', 'ENO1', 'ENO2', 'HKDC1', 'PDK1', 'GPI', 'PGK1', 'PGM1', 'IDH3A', 'ALDOA',
                'ALDOC', 'VASP', 'HSP90', 'VEGFR1', 'KITLG', 'TEK', 'BACE1', 'ANTXR2', 'BDNF', 'NFKBIA', 'MOV10L1',
                'TP53', 'PCNA', 'CCRK', 'CCND1', 'E2F3', 'E2F6'],
    'Oxidative_stress': ['EPHX1', 'SOD1', 'SOD2', 'SOD3', 'CYP1A1', 'NOS1', 'NOS2', 'NOS3', 'MAOB', 'PTGS2', 'CAT',
                         'GPX', 'GST', 'TXN', 'HMOX1', 'NQO1', 'TXRD1', 'PRDX4', 'NEIL3'],
}


if __name__ == '__main__':
    # Get all genes of interest.
    selected_genes = np.unique(sum(GENES_BY_CELL_TYPE.values(), []) + sum(GENES_BY_FUNCTION.values(), []))

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
        selected_feature_indices = features[1].isin(selected_genes).to_numpy()
        features = features[selected_feature_indices]
        joined_features = [f"{f0}_{f1}" for f0, f1 in zip(features[0], features[1])]
        del features

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
        final_matrix = matrix.X.T
        final_matrix = final_matrix[:, selected_feature_indices]
        # NOTE: Filter underexpressed pixels. Remove pixels where zero gene is expressed.
        barcode_position_expressed = np.array(final_matrix.sum(axis=1) > 0).reshape(-1)

        # Apply filtering.
        barcode_position_valid = np.logical_and(barcode_position_in_image, barcode_position_expressed)
        barcode_position = barcode_position[barcode_position_valid]
        barcode_position['pixel_row_in_highres'] = np.floor(barcode_position['pixel_row_in_highres']).astype(int)
        barcode_position['pixel_col_in_highres'] = np.floor(barcode_position['pixel_col_in_highres']).astype(int)
        final_matrix = final_matrix[barcode_position_valid, :]

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

            sub_matrix = final_matrix[indices, :]
            sub_adata = ad.AnnData(X=sub_matrix, obs=pd.DataFrame({'Location': group['cell_index']}), var=pd.DataFrame({'Gene Expression': joined_features}))
            coords = np.concatenate((group['pixel_row_in_highres'].values[:, None], group['pixel_col_in_highres'].values[:, None]), axis=1)
            sub_adata.obsm['spatial'] = coords

            os.makedirs(folder_out, exist_ok=True)
            sub_adata.write(os.path.join(folder_out, f'{target_folder}_Bin-{str(row_bin).zfill(2)}-{str(col_bin).zfill(2)}_spatial_matrix.h5ad'))
