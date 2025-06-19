import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from gat_model import GATClassifier
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything
from log_utils import log
from data_utils import split_dataset
from scheduler import LinearWarmupCosineAnnealingLR

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/src/dataset/')
from placenta import PlacentaDataset

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


def prepare_dataloaders(args):
    dataset = PlacentaDataset(data_folder=args.data_folder)

    # Train/val/test split
    ratios = [float(c) for c in args.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])

    train_set, val_set, test_set = split_dataset(
        dataset=dataset,
        splits=ratios,
        random_seed=0)  # Fix the dataset.

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, optimizer, loss_fn, device, max_iter):
    train_loss = 0
    y_true_arr, y_pred_arr = None, None

    for iter_idx, data_item in enumerate(tqdm(train_loader)):
        if max_iter is not None and iter_idx > max_iter:
            break

        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        y_pred = model(data_item)
        loss = loss_fn(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.mean().item()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.hstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

    train_loss /= min(max_iter, len(train_loader))
    accuracy = accuracy_score(y_true_arr, y_pred_arr.argmax(axis=1))
    auroc = roc_auc_score(y_true_arr, y_pred_arr, multi_class='ovo', average='macro')
    return model, train_loss, accuracy, auroc

@torch.no_grad()
def val_epoch(model, val_loader, loss_fn, device):
    val_loss = 0
    y_true_arr, y_pred_arr = None, None

    for data_item in val_loader:
        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        y_pred = model(data_item)
        loss = loss_fn(y_pred, y_true)

        val_loss += loss.mean().item()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.hstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

    val_loss /= len(val_loader)
    accuracy = accuracy_score(y_true_arr, y_pred_arr.argmax(axis=1))
    auroc = roc_auc_score(y_true_arr, y_pred_arr, multi_class='ovo', average='macro')
    return model, val_loss, accuracy, auroc

@torch.no_grad()
def test_model(model, test_loader, loss_fn, device):
    test_loss = 0
    y_true_arr, y_pred_arr = None, None

    for data_item in test_loader:
        data_item = data_item.to(device)
        y_true = torch.Tensor(data_item.y).long().to(device)
        y_pred = model(data_item)
        loss = loss_fn(y_pred, y_true)

        test_loss += loss.mean().item()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.hstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true_arr, y_pred_arr.argmax(axis=1))
    auroc = roc_auc_score(y_true_arr, y_pred_arr, multi_class='ovo', average='macro')
    return model, test_loss, accuracy, auroc


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Entry point.')
    args.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    args.add_argument('--max-epochs', default=50, type=int)
    args.add_argument('--max-training-iters', default=256, type=int)
    args.add_argument('--batch-size', default=64, type=int)
    args.add_argument('--learning-rate', default=1e-3, type=float)
    args.add_argument('--num-workers', default=8, type=int)
    args.add_argument('--random-seed', default=1, type=int)
    args.add_argument('--data-folder', default='$ROOT/data/spatial_placenta_accreta/patchified/', type=str)

    args = args.parse_known_args()[0]
    seed_everything(args.random_seed)

    # Update paths with absolute path.
    args.data_folder = args.data_folder.replace('$ROOT', ROOT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GATClassifier()
    model.to(device)

    # Set up training tools.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=min(10, args.max_epochs),
        warmup_start_lr=args.learning_rate * 1e-2,
        max_epochs=args.max_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Load the data.
    train_loader, val_loader, test_loader = prepare_dataloaders(args)

    log_file = os.path.join(ROOT_DIR, 'comparisons', 'results', 'GAT', f'log_seed-{args.random_seed}.txt')
    model_save_path = os.path.join(ROOT_DIR, 'comparisons', 'results', 'GAT', f'model_seed-{args.random_seed}.pt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    log(f'[GAT] Training begins.', filepath=log_file)
    best_val_loss = np.inf
    for epoch_idx in tqdm(range(args.max_epochs)):
        model.train()
        model, loss, accuracy, auroc = train_epoch(model, train_loader, optimizer, loss_fn, device, args.max_training_iters)
        scheduler.step()
        log(f'Epoch {epoch_idx}/{args.max_epochs}: Training Loss {loss:.3f}, ACC {accuracy:.3f}, macro AUROC {auroc:.3f}.',
            filepath=log_file)

        model.eval()
        model, loss, accuracy, auroc = val_epoch(model, val_loader, loss_fn, device)
        log(f'Validation Loss {loss:.3f}, ACC {accuracy:.3f}, macro AUROC {auroc:.3f}.',
            filepath=log_file)

        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), model_save_path)
            log('Model weights successfully saved.', filepath=log_file)

    model.eval()
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model, loss, accuracy, auroc = test_model(model, test_loader, loss_fn, device)
    log(f'\n\nTest Loss {loss:.3f}, ACC {accuracy:.3f}, macro AUROC {auroc:.3f}.',
        filepath=log_file)
