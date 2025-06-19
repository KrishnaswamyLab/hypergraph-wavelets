import os
from glob import glob
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm


folder_in = '../../data/spatial_placenta_accreta/raw/'
folder_out_highres = '../../data/spatial_placenta_accreta/resolution_2um_h5ad/'
folder_out_lowres = '../../data/spatial_placenta_accreta/resolution_57um_h5ad/'
num_bins = 100


if __name__ == '__main__':
    all_folder_paths = sorted(glob(os.path.join(folder_in, '*', 'filtered_feature_bc_matrix')))

    filtered_folder_names, filtered_folder_paths = [], []
    for folder_path in all_folder_paths:
        if 'normal_placenta' in folder_path:
            filtered_folder_paths.append(folder_path)
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_normal_placenta')
        elif 'PAS' in folder_path:
            filtered_folder_paths.append(folder_path)
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_PAS')
    del all_folder_paths

    for source_path, target_folder in tqdm(zip(filtered_folder_paths, filtered_folder_names), total=len(filtered_folder_paths)):
        # NOTE: Save the original high ressolution dataset.
        matrix = ad.io.read_mtx(os.path.join(source_path, 'matrix.mtx'))
        barcodes = pd.read_csv(os.path.join(source_path, 'barcodes.tsv'), header=None, sep="\t")
        features = pd.read_csv(os.path.join(source_path, 'features.tsv'), header=None, sep="\t")
        joined_features = [f"{f0}_{f1}" for f0, f1 in zip(features[0], features[1])]

        barcodes[['X', 'Y']] = barcodes[0].str.extract(r's_\d+um_(\d+)_(\d+)-\d')
        barcodes['X'] = barcodes['X'].astype(int)
        barcodes['Y'] = barcodes['Y'].astype(int)
        coords = np.concatenate((barcodes['X'].values[:, None], barcodes['Y'].values[:, None]), axis=1)

        adata = ad.AnnData(X=matrix.X.T,
                           obs=pd.DataFrame({'Location': barcodes[0]}),
                           var=pd.DataFrame({'Gene Expression': joined_features}))
        adata.obsm['spatial'] = coords

        os.makedirs(folder_out_highres, exist_ok=True)
        adata.write(os.path.join(folder_out_highres, f'{target_folder}_spatial_matrix.h5ad'))

        # NOTE: Spatial grouping ("superpixel") to create lower resolution dataset.
        cell_bins = pd.DataFrame({'X_bin': pd.cut(barcodes['X'], bins=num_bins, labels=False, include_lowest=True),
                                  'Y_bin': pd.cut(barcodes['Y'], bins=num_bins, labels=False, include_lowest=True),
                                  'cell_index': np.arange(len(barcodes))})

        # Aggregate sparse data based on bin groups
        grouped_data = []
        # Iterate over groups and compute mean for sparse values
        for (x_bin, y_bin), group in cell_bins.groupby(['X_bin', 'Y_bin']):
            # Extract rows corresponding to this group
            indices = group['cell_index'].values
            sub_matrix = adata.X[indices, :]  # Select rows for the group
            # Compute mean for the group, column by column
            group_means = np.array(sub_matrix.mean(axis=0)).ravel()
            grouped_data.append(((x_bin, y_bin), group_means))

        bins = np.concatenate((np.array([x[0] for x, _ in grouped_data])[:, None], np.array([x[1] for x, _ in grouped_data])[:, None]), axis=1)
        binned_matrix = np.stack([y for _, y in grouped_data], axis=0)
        lowres_adata = ad.AnnData(X=binned_matrix, obs=pd.DataFrame({'Location': ['Bin_' + str(x[0]) + '_' + str(x[1]) for x, _ in grouped_data]}), var=pd.DataFrame({'Gene Expression': joined_features}))
        lowres_adata.obsm['spatial'] = bins

        os.makedirs(folder_out_lowres, exist_ok=True)
        lowres_adata.write(os.path.join(folder_out_lowres, f'{target_folder}_spatial_matrix.h5ad'))
