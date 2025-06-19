import os
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

folder_in = '../../data/MIBI/raw/'
folder_out = '../../data/MIBI/all_genes/'
MIN_PIXEL_PER_GRAPH = 20


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
        matrix = matrix.drop(['id', 'unique_id', 'label', 'area', 'x_centroid', 'y_centroid'], axis=1)

        barcodes = barcodes.reset_index(drop=True)
        matrix = matrix.reset_index(drop=True)
        variable_df = pd.DataFrame({'Gene Expression': features})
        variable_df.index = features
        adata = ad.AnnData(X=matrix, obs=pd.DataFrame({'Location': np.arange(len(matrix))}), var=variable_df)
        coords = np.concatenate((barcodes['X'].values[:, None], barcodes['Y'].values[:, None]), axis=1)
        adata.obsm['spatial'] = coords

        label_binary = patient_labels.loc[patient_labels.id == patient_id]['response_binary'].values.item()
        label_multi = patient_labels.loc[patient_labels.id == patient_id]['response_multi'].values.item()
        os.makedirs(folder_out, exist_ok=True)
        adata.write(os.path.join(folder_out, f'patient_{patient_id}-responseB_{label_binary}-responseM_{label_multi}_spatial_matrix.h5ad'))
