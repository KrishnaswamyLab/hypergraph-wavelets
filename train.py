import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from model.hypergraph_scattering import HypergraphScatteringNet
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything
from log_utils import log
from data_utils import split_dataset, split_indices
from scheduler import LinearWarmupCosineAnnealingLR

sys.path.insert(0, import_dir + '/src/dataset/')
from placenta import PlacentaDatasetHypergraph
from mibi import MIBIDataset, MIBISubsetHypergraph
from extend import ExtendedDataset


ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])


def prepare_dataloaders(args):
    if args.dataset == 'placenta':
        dataset = PlacentaDatasetHypergraph(data_folder=args.data_folder, k_hop=args.k_hop)

        # Train/val/test split
        ratios = [float(c) for c in args.train_val_test_ratio.split(':')]
        ratios = tuple([c / sum(ratios) for c in ratios])

        train_set, val_set, test_set = split_dataset(
            dataset=dataset,
            splits=ratios,
            random_seed=0)  # Fix the dataset.

    elif args.dataset == 'mibi':
        dataset = MIBIDataset(data_folder=args.data_folder, k_hop=args.k_hop)

        ratios = [float(c) for c in args.train_val_test_ratio.split(':')]
        ratios = tuple([c / sum(ratios) for c in ratios])
        indices = list(range(len(dataset)))

        # NOTE: This is a hack to make sure all sets have all classes.
        all_class_subset1 = [0, 1, 4, 6]
        all_class_subset2 = [7, 8, 10, 11]
        all_class_subset3 = [12, 13, 14, 15]
        indices = list(set(indices) - set(all_class_subset1 + all_class_subset2 + all_class_subset3))
        adjusted_ratios = len(dataset) * np.array(ratios) - np.array([4, 4, 4])
        adjusted_ratios = tuple([c / sum(adjusted_ratios) for c in adjusted_ratios])
        train_indices, val_indices, test_indices = \
            split_indices(indices=indices, splits=adjusted_ratios, random_seed=0)
        train_indices += all_class_subset1
        val_indices += all_class_subset2
        test_indices += all_class_subset3

        train_set = MIBISubsetHypergraph(
            dataset=dataset,
            subset_indices=train_indices)
        val_set = MIBISubsetHypergraph(
            dataset=dataset,
            subset_indices=val_indices)
        test_set = MIBISubsetHypergraph(
            dataset=dataset,
            subset_indices=test_indices)

    min_batch_per_epoch = 5
    desired_len = args.batch_size * min_batch_per_epoch
    if len(train_set) < desired_len:
        train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, dataset.num_classes

def train_epoch(model, train_loader, optimizer, loss_fn, device, max_iter, num_classes):
    train_loss = 0
    y_true_arr, y_pred_arr = None, None
    optimizer.zero_grad()
    batch_per_backprop = int(args.desired_batch_size / args.batch_size)

    for iter_idx, data_item in enumerate(tqdm(train_loader)):
        if max_iter is not None and iter_idx > max_iter:
            break

        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        y_pred = model(
            x=data_item.x,
            hyperedge_index=data_item.edge_index,
            hyperedge_attr=data_item.edge_attr,
            batch=data_item.batch)

        loss = loss_fn(y_pred, y_true)

        loss_ = loss / batch_per_backprop
        loss_.backward()

        train_loss += loss.mean().item()

        # Simulate bigger batch size by batched optimizer update.
        if iter_idx % batch_per_backprop == batch_per_backprop - 1:
            optimizer.step()
            optimizer.zero_grad()

        y_true_np = y_true.detach().cpu().numpy()                        # shape: (batch size, 1)
        y_pred_np = torch.softmax(y_pred, dim=1).detach().cpu().numpy()  # shape: (batch size, num classes)
        if y_true_arr is None:
            y_true_arr = y_true_np
            y_pred_arr = y_pred_np
        else:
            y_true_arr = np.hstack((y_true_arr, y_true_np))
            y_pred_arr = np.vstack((y_pred_arr, y_pred_np))

    train_loss /= min(max_iter, len(train_loader))
    accuracy = accuracy_score(y_true_arr, np.argmax(y_pred_arr, axis=1))
    auroc = roc_auc_score(y_true_arr, y_pred_arr, multi_class='ovo', average='macro', labels=np.arange(num_classes))
    return model, train_loss, accuracy, auroc

@torch.no_grad()
def val_epoch(model, val_loader, loss_fn, device, max_iter, num_classes):
    val_loss = 0
    y_true_arr, y_pred_arr = None, None

    for iter_idx, data_item in enumerate(val_loader):
        if max_iter is not None and iter_idx > max_iter:
            break

        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        y_pred = model(
            x=data_item.x,
            hyperedge_index=data_item.edge_index,
            hyperedge_attr=data_item.edge_attr,
            batch=data_item.batch)
        loss = loss_fn(y_pred, y_true)

        val_loss += loss.mean().item()

        y_true_np = y_true.detach().cpu().numpy()                        # shape: (batch size, 1)
        y_pred_np = torch.softmax(y_pred, dim=1).detach().cpu().numpy()  # shape: (batch size, num classes)
        if y_true_arr is None:
            y_true_arr = y_true_np
            y_pred_arr = y_pred_np
        else:
            y_true_arr = np.hstack((y_true_arr, y_true_np))
            y_pred_arr = np.vstack((y_pred_arr, y_pred_np))

    val_loss /= min(max_iter, len(val_loader))
    accuracy = accuracy_score(y_true_arr, np.argmax(y_pred_arr, axis=1))
    auroc = roc_auc_score(y_true_arr, y_pred_arr, multi_class='ovo', average='macro', labels=np.arange(num_classes))
    return model, val_loss, accuracy, auroc

@torch.no_grad()
def test_model(model, test_loader, loss_fn, device, num_classes):
    test_loss = 0
    y_true_arr, y_pred_arr = None, None

    for data_item in test_loader:
        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        y_pred = model(
            x=data_item.x,
            hyperedge_index=data_item.edge_index,
            hyperedge_attr=data_item.edge_attr,
            batch=data_item.batch)
        loss = loss_fn(y_pred, y_true)

        test_loss += loss.mean().item()


        y_true_np = y_true.detach().cpu().numpy()                        # shape: (batch size, 1)
        y_pred_np = torch.softmax(y_pred, dim=1).detach().cpu().numpy()  # shape: (batch size, num classes)
        if y_true_arr is None:
            y_true_arr = y_true_np
            y_pred_arr = y_pred_np
        else:
            y_true_arr = np.hstack((y_true_arr, y_true_np))
            y_pred_arr = np.vstack((y_pred_arr, y_pred_np))

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true_arr, np.argmax(y_pred_arr, axis=1))
    auroc = roc_auc_score(y_true_arr, y_pred_arr, multi_class='ovo', average='macro', labels=np.arange(num_classes))
    return model, test_loss, accuracy, auroc


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Entry point.')
    args.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    args.add_argument('--max-epochs', default=50, type=int)
    args.add_argument('--max-training-iters', default=512, type=int)
    args.add_argument('--max-validation-iters', default=256, type=int)
    args.add_argument('--batch-size', default=1, type=int)
    args.add_argument('--desired-batch-size', default=16, type=int)
    args.add_argument('--learning-rate', default=1e-2, type=float)
    args.add_argument('--trainable-scales', action='store_true')
    args.add_argument('--k-hop', default=1, type=int)
    args.add_argument('--num-workers', default=8, type=int)
    args.add_argument('--random-seed', default=1, type=int)
    args.add_argument('--dataset', default='placenta', type=str)
    args.add_argument('--data-folder', default='$ROOT/data/spatial_placenta_accreta/patchified_selected_genes', type=str)
    args.add_argument('--num-features', default=212, type=int)  # number of genes or features

    args = args.parse_known_args()[0]
    seed_everything(args.random_seed)

    # Update paths with absolute path.
    args.data_folder = args.data_folder.replace('$ROOT', ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data.
    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(args)

    model = HypergraphScatteringNet(
        in_channels=64,
        hidden_channels=16,
        out_channels=num_classes,
        num_features=args.num_features,
        trainable_laziness=False,
        trainable_scales=args.trainable_scales,
        activation=None,  # just get one layer of wavelet transform
        fixed_weights=True,
        layout=['hsm'],
        normalize='right',
        pooling='attention',
        scale_list=[0,1,2,4]
    )
    model.eval()
    model.to(device)

    # Set up training tools.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=min(10, args.max_epochs),
        warmup_start_lr=args.learning_rate * 1e-2,
        max_epochs=args.max_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    subset_name = os.path.basename(args.data_folder.rstrip('/'))
    current_run_identifier = f'dataset-{args.dataset}-{subset_name}_kHop-{args.k_hop}_features-{args.num_features}_trainable_scales-{args.trainable_scales}_seed-{args.random_seed}'
    log_file = os.path.join(ROOT_DIR, 'results', args.dataset, current_run_identifier, 'log.txt')
    model_save_path = os.path.join(ROOT_DIR, 'results', args.dataset, current_run_identifier, 'model.pt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Log the config.
    config_str = 'Config: \n'
    for key in vars(args).keys():
        config_str += '%s: %s\n' % (key, vars(args)[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=log_file, to_console=True)

    log(f'[HypergraphScattering] Training begins.', filepath=log_file)
    best_val_auroc = 0
    for epoch_idx in tqdm(range(args.max_epochs)):
        model.train()
        model, train_loss, train_accuracy, train_auroc = train_epoch(model, train_loader, optimizer, loss_fn, device, args.max_training_iters, num_classes)
        scheduler.step()
        log(f'Epoch {epoch_idx + 1}/{args.max_epochs}: (LR={optimizer.param_groups[0]['lr']}) Training Loss {train_loss:.3f}, ACC {train_accuracy:.3f}, macro AUROC {train_auroc:.3f}.',
            filepath=log_file)

        model.eval()
        model, val_loss, val_accuracy, val_auroc = val_epoch(model, val_loader, loss_fn, device, args.max_validation_iters, num_classes)
        log(f'Validation Loss {val_loss:.3f}, ACC {val_accuracy:.3f}, macro AUROC {val_auroc:.3f}.',
            filepath=log_file)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), model_save_path)
            log('Model weights successfully saved.', filepath=log_file)

    model.eval()
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model, test_loss, test_accuracy, test_auroc = test_model(model, test_loader, loss_fn, device, num_classes)
    log(f'\n\nTest Loss {test_loss:.3f}, ACC {test_accuracy:.3f}, macro AUROC {test_auroc:.3f}.',
        filepath=log_file)
