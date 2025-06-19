import os
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

folder_in = '../../data/MIBI/raw/'
folder_out = '../../data/MIBI/patchified_all_genes/'
MIN_CELL_PER_GRAPH = 20
BIN_SIZE_RATIO = 0.25
TARGET_NUM_SUBGRAPHS = 128


if __name__ == '__main__':
    '''
    We have 2 csv files that covers 54 patients.
    1. `cell_protein_data.csv` records the cell-by-protein matrices and (x, y, area) for each cell.
    2. `patient_info.csv` records the class labels for the patients.
    '''
    cell_by_protein = pd.read_csv(os.path.join(folder_in, 'cell_protein_data.csv'))
    patient_labels = pd.read_csv(os.path.join(folder_in, 'patient_info.csv'))

    for patient_id in tqdm(patient_labels.id):
        assert patient_id in cell_by_protein.id.values, 'Patient ID mismatch.'

        # Construct `barcodes`.
        cell_info = cell_by_protein.copy()
        cell_info = cell_info.loc[cell_info.id == patient_id]
        cell_info = cell_info.drop(['area'], axis=1)
        barcodes = 'patient_' + cell_info.id + '-cell_' + cell_info['label'].apply(lambda x: str(x).zfill(4))
        barcodes = pd.DataFrame(barcodes, columns=['barcode'])
        barcodes['X'] = cell_info['x_centroid'].astype(int)
        barcodes['Y'] = cell_info['y_centroid'].astype(int)
        del cell_info

        # Construct `features`.
        features = cell_by_protein.columns.values.tolist()
        for key in ['id', 'unique_id', 'label', 'area', 'x_centroid', 'y_centroid']:
            features.remove(key)

        # Construct `matrix`.
        matrix = cell_by_protein.copy()
        matrix = matrix.loc[matrix['id'] == patient_id]
        matrix = matrix.drop(['id', 'unique_id', 'label', 'area'], axis=1)

        variable_df = pd.DataFrame({'Gene Expression': features})
        variable_df.index = features

        label_binary = patient_labels.loc[patient_labels.id == patient_id]['response_binary'].values.item()
        label_multi = patient_labels.loc[patient_labels.id == patient_id]['response_multi'].values.item()

        # Subsample the graph into smaller graphs. Otherwise the dataset size is too small for learning.
        x_min, x_max, y_min, y_max = barcodes.X.min(), barcodes.X.max(), barcodes.Y.min(), barcodes.Y.max()
        size_x, size_y = x_max - x_min, y_max - y_min
        bin_size_x = BIN_SIZE_RATIO * size_x
        bin_size_y = BIN_SIZE_RATIO * size_y
        num_subgraphs_nonoverlap = int((size_x / bin_size_x) * (size_y / bin_size_y))
        oversampling = np.ceil(np.sqrt(TARGET_NUM_SUBGRAPHS / num_subgraphs_nonoverlap))
        step_size_x = bin_size_x / oversampling
        step_size_y = bin_size_y / oversampling

        x_start, y_start = x_min, y_min
        while x_start + bin_size_x < x_max + step_size_x:
            x_end = x_start + bin_size_x
            while y_start + bin_size_y < y_max + step_size_y:
                y_end = y_start + bin_size_y

                subsample_indices = np.logical_and.reduce([
                    matrix['x_centroid'] >= x_start,
                    matrix['x_centroid'] < x_end,
                    matrix['y_centroid'] >= y_start,
                    matrix['y_centroid'] < y_end,
                ])

                sub_barcodes = barcodes.loc[subsample_indices]
                sub_matrix = matrix.loc[subsample_indices]

                sub_matrix = sub_matrix.drop(['x_centroid', 'y_centroid'], axis=1)

                # NOTE: Filter underexpressed cells. Not filtering unexpressed genes because they may vary across subjects.
                # Remove cells where zero gene is expressed.
                barcode_position_valid = np.array(sub_matrix.sum(axis=1) > 0).reshape(-1)
                sub_barcodes = sub_barcodes[barcode_position_valid]
                sub_matrix = sub_matrix[barcode_position_valid]

                sub_barcodes = sub_barcodes.reset_index(drop=True)
                sub_matrix = sub_matrix.reset_index(drop=True)

                # Check if bin has sufficient number of cells.
                num_cells = len(sub_matrix)
                if num_cells < MIN_CELL_PER_GRAPH:
                    print(f'Subgraph (x={x_start}~{x_end}, y={y_start}~{y_end}) has fewer than {MIN_CELL_PER_GRAPH} pixels ({num_cells}). Skipping this bin.')
                else:
                    sub_adata = ad.AnnData(X=sub_matrix, obs=pd.DataFrame({'Location': np.arange(len(sub_matrix))}), var=variable_df)
                    coords = np.concatenate((sub_barcodes['X'].values[:, None], sub_barcodes['Y'].values[:, None]), axis=1)
                    sub_adata.obsm['spatial'] = coords

                    os.makedirs(folder_out, exist_ok=True)
                    sub_adata.write(os.path.join(folder_out, f'patient_{patient_id}_Box-{int(x_start)}_{int(x_end)}_{int(y_start)}_{int(y_end)}-responseB_{label_binary}-responseM_{label_multi}_spatial_matrix.h5ad'))

                # Increment rules.
                y_start += step_size_y
            # Increment rules.
            y_start = y_min
            x_start += step_size_x

