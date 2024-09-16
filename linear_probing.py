import os 
import numpy as np
import pandas
import argparse 
import anndata as ad
import torch

# imports for linear probing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, default='/vast/palmer/pi/krishnaswamy_smita/jcr222/hypergraphs/data/')
    argparser.add_argument('--wavelet_feature_dir', type=str, default='wavelet_features/')
    argparser.add_argument('--lin_prob_target', type = str, default = 'Braak')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--wavelets', type=int, default=1)
    argparser.add_argument('--pts_per_dataset', type=int, default=1000)
    args = argparser.parse_args()

    DATA_DIR = args.data_dir
    FEATURE_DIR = args.wavelet_feature_dir
    save_end = '_neighborhood_feat.pt' if args.wavelets else '_node_features.pt'

    lin_prob_target = args.lin_prob_target
    points_per_dataset = args.pts_per_dataset
    
    datasets = os.listdir(DATA_DIR)
    num_datasets = len(datasets)
    print(f'Number of datasets: {num_datasets}')
    example_features = torch.load(FEATURE_DIR + datasets[0] + save_end, weights_only=True)
    num_features = example_features.shape[1]
    del example_features

    features = np.zeros((num_datasets, points_per_dataset, num_features))
    labels = np.zeros((num_datasets, points_per_dataset))
    label_dict = {}
    for ind, dataset_name in enumerate(datasets):
        print(DATA_DIR + dataset_name)
        adata = ad.read_h5ad(DATA_DIR + dataset_name)

        # load in the wavelet features
        hyperedge_feat = torch.load(FEATURE_DIR + dataset_name + save_end, weights_only=True)
        print(f'num features: {hyperedge_feat.shape[1]}')

        if hyperedge_feat.shape[1] != num_features:
            print('Number of features does not match')
            #print(f'Expected: {num_features}, Got: {hyperedge_feat.shape[1]}')
            continue
            #import pdb; pdb.set_trace()
        
        # convert to numpy and set nans to 0
        hyperedge_feat = hyperedge_feat.detach().numpy()
        hyperedge_feat = np.nan_to_num(hyperedge_feat)
        # randomly sample points_per_dataset points
        idx = np.random.choice(hyperedge_feat.shape[0], points_per_dataset, replace=False)
        hyperedge_feat = hyperedge_feat[idx, :]
        dataset_labels = adata.obs[lin_prob_target].values
        dataset_labels = dataset_labels[idx]
        remapped_labels = np.zeros_like(dataset_labels)
        # use the label_dict to convert labels to integers
        # first check if the label is in the label_dict
        for i, label in enumerate(dataset_labels):
            if label not in label_dict:
                label_dict[label] = int(len(label_dict))
            remapped_labels[i] = label_dict[label]
        features[ind, :, :] = hyperedge_feat
        labels[ind, :] = remapped_labels

    # now we have the features and labels
    # we can do linear probing
    # first we need to flatten the features and labels
    features = features.reshape(-1, num_features)
    labels = labels.reshape(-1)
    # convert labels to one hot
    #from sklearn.preprocessing import OneHotEncoder
    #labels_one_hot = OneHotEncoder().fit_transform(labels.reshape(-1, 1)).toarray()

    # labels_one_hot = np.zeros((labels.size, len(label_dict)))
    # labels_one_hot[np.arange(labels.size), labels] = 1
    # now we can do linear probing
    # first we need to split the data into training and testing
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    accuracies = []
    balanced_accuracies = []
    f1_scores = []
    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        #clf = LogisticRegression(random_state=args.seed, max_iter=1000)
        clf = LogisticRegression(random_state=args.seed, max_iter=1000, multi_class='multinomial')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    print(f'Accuracy: {np.mean(accuracies)} +/- {np.std(accuracies)}')
    print(f'Balanced Accuracy: {np.mean(balanced_accuracies)} +/- {np.std(balanced_accuracies)}')
    print(f'F1 Score: {np.mean(f1_scores)} +/- {np.std(f1_scores)}')
    # save the results




