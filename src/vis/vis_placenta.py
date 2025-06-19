import os
from glob import glob
import phate
import scprep
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from matplotlib import pyplot as plt
# from squidpy.read import visium


def all_almost_integers(lst, tolerance=1e-6):
    '''Checks if all numbers in a list are close to integers.'''
    return np.isclose(lst, np.round(lst), atol=tolerance).all()

def load_data(folder = '../../data/spatial_placenta_accreta/raw/',
              reduction = 'downsample'):

    all_folder_paths = sorted(glob(os.path.join(folder, '*', 'filtered_feature_bc_matrix')))

    filtered_folder_names, filtered_folder_paths = [], []
    for folder_path in all_folder_paths:
        if 'normal_placenta' in folder_path:
            filtered_folder_paths.append(folder_path)
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_normal_placenta')
        elif 'PAS' in folder_path:
            filtered_folder_paths.append(folder_path)
            filtered_folder_names.append('batch_' + folder_path.split('/')[-2].split('_')[0] + '_PAS')

    del all_folder_paths

    combined_sc_data = []
    for folder_path in tqdm(filtered_folder_paths):
        num_bins = 50

        if reduction == 'downsample':
            sc_data = scprep.io.load_10X(folder_path, sparse=True, gene_labels='both')
            sc_data = sc_data.sample(n=num_bins**2, random_state=1)

        elif reduction == 'aggregate':

            matrix = ad.io.read_mtx(os.path.join(folder_path, 'matrix.mtx'))
            barcodes = pd.read_csv(os.path.join(folder_path, 'barcodes.tsv'), header=None, sep="\t")
            features = pd.read_csv(os.path.join(folder_path, 'features.tsv'), header=None, sep="\t")
            joined_features = [f"{f0}_{f1}" for f0, f1 in zip(features[0], features[1])]
            del features

            barcodes[['X', 'Y']] = barcodes[0].str.extract(r's_\d+um_(\d+)_(\d+)-\d')
            barcodes['X'] = barcodes['X'].astype(int)
            barcodes['Y'] = barcodes['Y'].astype(int)
            coords = np.concatenate((barcodes['X'].values[:, None], barcodes['Y'].values[:, None]), axis=1)

            adata = ad.AnnData(X=matrix.X.T,
                               obs=pd.DataFrame({'Location': barcodes[0]}),
                               var=pd.DataFrame({'Gene Expression': joined_features}))
            adata.obsm['spatial'] = coords
            del matrix

            # Bin the X and Y coordinates into equal-sized bins
            cell_bins = pd.DataFrame({'X_bin': pd.cut(barcodes['X'], bins=num_bins, labels=False, include_lowest=True),
                                      'Y_bin': pd.cut(barcodes['Y'], bins=num_bins, labels=False, include_lowest=True),
                                      'cell_index': np.arange(len(barcodes))})
            del barcodes

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

            # Combine results
            sc_data = pd.DataFrame({
                'X_bin': [x[0] for x, _ in grouped_data],
                'Y_bin': [x[1] for x, _ in grouped_data],
                'gene_means': [y for _, y in grouped_data]
            })

            sc_data.index = ['Bin_' + str(row[0]) + '_' + str(row[1]) for row in sc_data.values]
            sc_data = sc_data.drop(columns=['X_bin', 'Y_bin'])
            sc_data = pd.DataFrame(sc_data['gene_means'].tolist(),
                                   columns=[item[0] for item in adata.var.values],
                                   index=sc_data.index).join(sc_data)
            sc_data = sc_data.drop(columns=['gene_means'])
            del adata

        combined_sc_data.append(sc_data)

    combined_sc_data, _ = scprep.utils.combine_batches(
        combined_sc_data,
        filtered_folder_names,
        append_to_cell_names=True
    )

    return combined_sc_data


if __name__ == '__main__':
    reduction_method = 'aggregate' # downsample, aggregate

    combined_sc_data = load_data(reduction=reduction_method)

    # Remove rare genes.
    combined_sc_data = scprep.filter.filter_rare_genes(combined_sc_data, min_cells=10)
    # Normalization.
    combined_sc_data = scprep.normalize.library_size_normalize(combined_sc_data)
    # Transformation.
    combined_sc_data = scprep.transform.sqrt(combined_sc_data)

    # Phate.
    phate_operator = phate.PHATE(n_jobs=8)
    Y_phate = phate_operator.fit_transform(combined_sc_data)
    label = [item.split('_batch_')[1] for item in combined_sc_data.index.tolist()]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    scprep.plot.scatter2d(Y_phate, c=label, figsize=(12,8), cmap="Spectral", ticks=False,  label_prefix="PHATE", ax=ax)
    fig.savefig(f'placenta_{reduction_method}.png')
    import pdb; pdb.set_trace()

