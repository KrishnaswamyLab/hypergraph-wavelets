import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Circle as MplCircle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, ROOT_DIR + '/src/')
from utils.seed import seed_everything
from models.hypergraph_scattering import HypergraphScatteringNet
from train import prepare_dataloaders


def _get_class_map(dataset) -> Dict[int, str]:
    if hasattr(dataset, 'class_map'):
        return dataset.class_map
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'class_map'):
        return dataset.dataset.class_map
    raise AttributeError('Could not find class_map on dataset.')


def _get_graph_path(dataset, local_idx: int) -> Optional[str]:
    if hasattr(dataset, 'graph_path_arr'):
        return dataset.graph_path_arr[local_idx]
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        base_dataset = dataset.dataset
        if hasattr(base_dataset, 'graph_path_arr'):
            return base_dataset.graph_path_arr[dataset.indices[local_idx]]
    return None


def _get_original_index(dataset, local_idx: int) -> int:
    if hasattr(dataset, 'indices'):
        return int(dataset.indices[local_idx])
    return int(local_idx)


@torch.no_grad()
def collect_logits(model, dataset, device, num_workers: int) -> List[dict]:
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    records: List[dict] = []
    for local_idx, data_item in enumerate(tqdm(loader)):
        data_item = data_item.to(device)
        y_true = int(data_item.y.view(-1)[0].item())
        logits = model(
            x=data_item.x,
            hyperedge_index=data_item.edge_index,
            hyperedge_attr=data_item.edge_attr,
            batch=data_item.batch)
        logits = logits.view(-1)
        pred_label = int(torch.argmax(logits).item())

        record = {
            'sample_idx': local_idx,
            'dataset_index': _get_original_index(dataset, local_idx),
            'path': _get_graph_path(dataset, local_idx),
            'true_label': y_true,
            'pred_label': pred_label,
            'logits': logits.cpu().numpy().tolist(),
        }
        records.append(record)
    return records


def rank_top_k(records: List[dict], class_idx: int, top_k: int) -> List[dict]:
    class_records = [r for r in records if r['true_label'] == class_idx]
    class_records.sort(key=lambda r: r['logits'][class_idx], reverse=True)
    return class_records[:top_k]


def _load_sample_visual_data(dataset, local_idx: int) -> Tuple[np.ndarray, np.ndarray, List[str], torch.Tensor]:
    graph_path = _get_graph_path(dataset, local_idx)
    if graph_path is None:
        raise ValueError('Could not resolve graph path for sample.')
    adata = ad.read_h5ad(graph_path)
    coords = np.asarray(adata.obsm['spatial'])
    cell_type_names = [str(name) for name in adata.var.to_numpy().flatten().tolist()]

    data_item = dataset[local_idx]
    if data_item.x.dim() != 2:
        raise ValueError('Expected node feature matrix with shape [num_nodes, num_features].')
    cell_type_ids = torch.argmax(data_item.x, dim=1).cpu().numpy()
    return coords, cell_type_ids, cell_type_names, data_item.edge_index


def load_attentions(attention_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-computed attentions from the same npz saved by vis_attention.save_test_set_attentions.
    Returns (niche_attention_arr, feature_attention_arr, mlp_weights, logits_arr).
    niche_attention_arr[i] is the niche attention for test sample i.
    logits_arr[i] is the raw model logits for test sample i (used for class-specific weighting).
    """
    if not os.path.isfile(attention_path):
        raise FileNotFoundError(
            f'Attentions not found at {attention_path}. Run vis_attention.py first to generate attentions.npz.'
        )
    data = np.load(attention_path, allow_pickle=True)
    return (
        data['niche_attention_arr'],
        data['feature_attention_arr'],
        np.array(data['mlp_weights']),
        np.array(data['y_pred_arr']),
    )


def _compute_hyperedge_importance(
    niche_attn: np.ndarray,
    hyperedge_index: torch.Tensor,
) -> Optional[np.ndarray]:
    """
    Per-hyperedge importance from per-node niche attention (mean over members).
    Returns None on shape mismatch.
    """
    if hyperedge_index is None or hyperedge_index.numel() == 0 or niche_attn.size == 0:
        return None
    node_ids = hyperedge_index[0].cpu().numpy()
    hyperedge_ids = hyperedge_index[1].cpu().numpy()
    if node_ids.size == 0:
        return None
    num_hyperedges = int(hyperedge_ids.max()) + 1
    num_nodes = int(node_ids.max()) + 1
    niche_attn = np.asarray(niche_attn, dtype=np.float64)
    if niche_attn.size != num_nodes:
        return None
    hyperedge_importance = np.zeros(num_hyperedges, dtype=np.float64)
    for hyperedge_id in range(num_hyperedges):
        members = node_ids[hyperedge_ids == hyperedge_id]
        if members.size > 0:
            hyperedge_importance[hyperedge_id] = np.mean(niche_attn[members])
    return hyperedge_importance


def _plot_hyperedges_contour(
    ax,
    coords: np.ndarray,
    hyperedge_index: torch.Tensor,
    niche_attn: np.ndarray,
    max_hyperedges: int,
    contour_alpha: float,
    contour_cmap: str,
    contour_filled: bool = True,
) -> None:
    """
    Draw each hyperedge as a curved contour (convex hull or circle for 2 nodes),
    colored by niche attention aggregated per hyperedge.
    coords: (n, 2) with [:,0]=y, [:,1]=x for plotting.
    """
    if hyperedge_index is None or hyperedge_index.numel() == 0 or niche_attn.size == 0:
        return
    node_ids = hyperedge_index[0].cpu().numpy()
    hyperedge_ids = hyperedge_index[1].cpu().numpy()
    if node_ids.size == 0:
        return

    importance = _compute_hyperedge_importance(niche_attn, hyperedge_index)
    if importance is None:
        return
    i_min, i_max = importance.min(), importance.max()
    if i_max > i_min:
        importance = (importance - i_min) / (i_max - i_min)
    else:
        importance = np.zeros_like(importance)

    unique_hyperedges, counts = np.unique(hyperedge_ids, return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    selected_hyperedges = unique_hyperedges[sort_idx][:max_hyperedges]

    patches = []
    values = []
    for hyperedge_id in selected_hyperedges:
        members = node_ids[hyperedge_ids == hyperedge_id]
        if members.size < 2:
            continue
        member_coords = coords[members]
        val = importance[int(hyperedge_id)]
        values.append(val)
        if members.size == 2:
            mid = member_coords.mean(axis=0)
            r = np.linalg.norm(member_coords[1] - member_coords[0]) / 2 + 1e-6
            # Circle: xy is (x, y) for matplotlib, coords are (y, x)
            patches.append(MplCircle((mid[1], mid[0]), r))
        else:
            try:
                hull = ConvexHull(member_coords)
                hull_pts = member_coords[hull.vertices]
                # matplotlib patch expects (x, y) = (col, row)
                patches.append(MplPolygon(hull_pts[:, [1, 0]], closed=True))
            except Exception:
                continue

    if not patches:
        return
    if contour_filled:
        cmap = plt.get_cmap(contour_cmap)
        collection = PatchCollection(
            patches, cmap=cmap, linewidths=0,
            edgecolors='none', alpha=contour_alpha,
        )
        values = np.array(values)
        vmin, vmax = values.min(), values.max()
        collection.set_array(values)
        collection.set_clim(vmin, vmax)
        ax.add_collection(collection)
        collection.set_zorder(1)


def plot_niche_prototypes(records_by_class: Dict[int, List[dict]],
                          dataset,
                          class_map: Dict[int, str],
                          output_path: str,
                          max_hyperedges: int,
                          node_size: float = 80,
                          niche_attn_by_sample: Optional[Dict[int, np.ndarray]] = None,
                          contour_alpha: float = 0.15,
                          contour_cmap: str = 'Reds') -> None:

    num_classes = len(class_map)
    # Two columns per class: outline (contours not filled), then filled (contours with importance color)
    num_cols = num_classes * 2 + 1
    max_rows = max(len(records) for records in records_by_class.values())
    fig, axes = plt.subplots(max_rows, num_cols, figsize=(4 * num_cols, 4 * max_rows))
    if max_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    cmap = plt.get_cmap("Paired")
    sample_cache: Dict[int, Tuple[np.ndarray, np.ndarray, List[str], torch.Tensor, float, float]] = {}
    x_max, y_max = 0.0, 0.0

    for records in records_by_class.values():
        for record in records:
            sample_idx = record['sample_idx']
            if sample_idx not in sample_cache:
                coords, cell_type_ids, cell_type_names, hyperedge_index = _load_sample_visual_data(dataset, sample_idx)
                coords = coords - coords.min(axis=0, keepdims=True)
                width = float(coords[:, 1].max())
                height = float(coords[:, 0].max())
                sample_cache[sample_idx] = (coords, cell_type_ids, cell_type_names, hyperedge_index, width, height)
            coords = sample_cache[sample_idx][0]
            x_max = max(x_max, coords[:, 1].max())
            y_max = max(y_max, coords[:, 0].max())

    pad_x = max(x_max * 0.03, 1.0)
    pad_y = max(y_max * 0.03, 1.0)

    for class_idx, class_name in class_map.items():
        records = records_by_class.get(class_idx, [])
        col_outline = class_idx * 2
        col_filled = class_idx * 2 + 1
        for row_idx in range(max_rows):
            for ax_col, contour_filled in [
                (col_outline, False),
                (col_filled, True),
            ]:
                ax = axes[row_idx, ax_col]
                ax.set_axis_off()
                if row_idx >= len(records):
                    continue

                record = records[row_idx]
                coords, cell_type_ids, cell_type_names, hyperedge_index, width, height = sample_cache[record['sample_idx']]
                x_offset = 0.5 * (x_max - width)
                y_offset = 0.5 * (y_max - height)
                coords_shifted = coords + np.array([y_offset, x_offset])
                num_cell_types = len(cell_type_names)
                sorted_names = sorted(cell_type_names)
                id_to_sorted_idx = np.array([sorted_names.index(cell_type_names[i]) for i in range(num_cell_types)])
                scatter_colors = np.array([cmap(i % cmap.N) for i in range(num_cell_types)])[id_to_sorted_idx[cell_type_ids]]
                ax.scatter(coords_shifted[:, 1], coords_shifted[:, 0],
                           c=scatter_colors,
                           s=node_size,
                           alpha=0.8,
                           zorder=2)

                if max_hyperedges > 0 and niche_attn_by_sample is not None:
                    niche_attn = niche_attn_by_sample.get(record['sample_idx'])
                    if niche_attn is not None:
                        _plot_hyperedges_contour(
                            ax,
                            coords_shifted,
                            hyperedge_index,
                            niche_attn,
                            max_hyperedges=max_hyperedges,
                            contour_alpha=contour_alpha,
                            contour_cmap=contour_cmap,
                            contour_filled=contour_filled,
                        )

                ax.set_aspect('equal')
                ax.set_xlim(-pad_x, x_max + pad_x)
                ax.set_ylim(y_max + pad_y, -pad_y)
                if row_idx == 0:
                    ax.set_title(class_name, fontsize=24)


    if max_rows > 0:
        legend_ax = axes[0, num_classes * 2]
        legend_ax.set_axis_off()
        first_record = list(records_by_class.values())[0][0]
        _, _, cell_type_names, _, _, _ = sample_cache[first_record['sample_idx']]
        sorted_names = sorted(cell_type_names)
        for sorted_idx, cell_type_name in enumerate(sorted_names):
            legend_ax.scatter([], [], c=[cmap(sorted_idx % cmap.N)], label=cell_type_name, s=120, alpha=0.8)
        legend_ax.legend(loc='center left', frameon=False, fontsize=14,
                         title='Cell type', title_fontsize=20)
        if niche_attn_by_sample is not None:
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(contour_cmap), norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            div = make_axes_locatable(legend_ax)
            cax = div.append_axes('right', size='8%', pad=0.4)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('Niche attention weights', fontsize=12, labelpad=16)

    for row_idx in range(max_rows):
        for col_idx in range(num_classes * 2, num_cols):
            if row_idx == 0 and col_idx == num_classes * 2:
                continue
            axes[row_idx, col_idx].set_axis_off()

    fig.tight_layout(pad=1.0)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cooccurrence_heatmap(
    dataset,
    niche_attention_arr: np.ndarray,
    logits_arr: np.ndarray,
    class_map: Dict[int, str],
    output_path: str,
) -> None:
    """
    Heatmap of co-occurring cell types (A x B) per class.

    Per-hyperedge importance = mean niche attention over nodes in the hyperedge (which
    nodes the model attends to). Class-specific weighting uses the raw class-c logit:
    each sample's contribution to the class-c heatmap is weighted by logits[sample, c],
    so heatmaps differ by class and can be negative (pushing away from class c).
    Value (A,B) for class c = sum over samples and hyperedges containing both A and B
    of (hyperedge importance) * logit_c(sample).
    """
    _, _, cell_type_names, _ = _load_sample_visual_data(dataset, 0)
    num_cell_types = len(cell_type_names)
    n_samples = min(len(dataset), len(niche_attention_arr), len(logits_arr))
    num_classes = len(class_map)

    fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]
    for ax, (class_idx, class_name) in zip(axes, class_map.items()):
        w = np.asarray(logits_arr[:n_samples, class_idx], dtype=np.float64)
        M = np.zeros((num_cell_types, num_cell_types), dtype=np.float64)
        for sample_idx in tqdm(range(n_samples)):
            _, cell_type_ids, _, hyperedge_index = _load_sample_visual_data(dataset, sample_idx)
            cell_type_ids = np.asarray(cell_type_ids)
            niche_attn = np.asarray(niche_attention_arr[sample_idx])
            importance = _compute_hyperedge_importance(niche_attn, hyperedge_index)
            if importance is None:
                continue
            node_ids = hyperedge_index[0].cpu().numpy()
            hyperedge_ids = hyperedge_index[1].cpu().numpy()
            num_hyperedges = importance.size
            for hyperedge_id in range(num_hyperedges):
                members = node_ids[hyperedge_ids == hyperedge_id]
                if members.size < 2:
                    continue
                cell_types_in_hyperedge = np.unique(cell_type_ids[members])
                imp = importance[hyperedge_id] * w[sample_idx]
                for i in range(len(cell_types_in_hyperedge)):
                    for j in range(i + 1, len(cell_types_in_hyperedge)):
                        a, b = int(cell_types_in_hyperedge[i]), int(cell_types_in_hyperedge[j])
                        M[a, b] += imp
                        M[b, a] += imp
        two_std = 2 * np.std(M)
        im = ax.imshow(M, cmap='coolwarm', aspect='auto', vmin=-two_std, vmax=two_std)
        ax.set_xticks(range(num_cell_types))
        ax.set_yticks(range(num_cell_types))
        ax.set_xticklabels(cell_type_names, rotation=45, ha='right')
        ax.set_yticklabels(cell_type_names)
        ax.set_title(class_name)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Rank top niche prototypes by logits.')
    args.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    args.add_argument('--trainable-scales', action='store_true')
    args.add_argument('--k-hop', default=3, type=int)
    args.add_argument('--num-workers', default=8, type=int)
    args.add_argument('--random-seed', default=1, type=int)
    args.add_argument('--dataset', default='placenta', type=str)
    args.add_argument('--data-folder', default='$ROOT/data/spatial_placenta_accreta/patchified_selected_genes', type=str)
    args.add_argument('--num-features', default=212, type=int)  # number of genes or features
    args.add_argument('--top-k', default=5, type=int)
    args.add_argument('--max-hyperedges', default=10000, type=int)

    args = args.parse_known_args()[0]
    args.batch_size = 1
    args.desired_batch_size = 1
    seed_everything(args.random_seed)

    args.data_folder = args.data_folder.replace('$ROOT', ROOT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, num_classes = prepare_dataloaders(args)
    dataset = test_loader.dataset
    class_map = _get_class_map(dataset)

    model = HypergraphScatteringNet(
        in_channels=64,
        hidden_channels=64,
        out_channels=num_classes,
        num_features=args.num_features,
        trainable_laziness=False,
        trainable_scales=args.trainable_scales,
        activation=None,  # just get one layer of wavelet transform
        fixed_weights=True,
        layout=['hsm'],
        normalize='right',
        pooling='attention',
        scale_list=[0, 1, 2, 4]
    )
    model.eval()
    model.to(device)

    subset_name = os.path.basename(args.data_folder.rstrip('/'))
    current_run_identifier = f'dataset-{args.dataset}-{subset_name}_kHop-{args.k_hop}_features-{args.num_features}_trainable_scales-{args.trainable_scales}_seed-{args.random_seed}'
    model_save_path = os.path.join(ROOT_DIR, 'results', args.dataset, current_run_identifier, 'model.pt')
    output_dir = os.path.join(ROOT_DIR, 'results', args.dataset, current_run_identifier)
    os.makedirs(output_dir, exist_ok=True)

    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    records = collect_logits(model, dataset, device=device, num_workers=args.num_workers)

    top_by_class: Dict[int, List[dict]] = {}
    for class_idx, class_name in class_map.items():
        top_by_class[class_idx] = rank_top_k(records, class_idx=class_idx, top_k=args.top_k)

    attention_path = os.path.join(output_dir, 'attentions.npz')
    if not os.path.isfile(attention_path):
        raise FileNotFoundError(
            f'Attentions not found at {attention_path}. Run vis_attention.py first to generate attentions.npz.'
        )
    niche_attention_arr, feature_attention_arr, mlp_weights, logits_arr = load_attentions(attention_path)

    figure_path_cooccur = os.path.join(output_dir, 'niche_cooccurrence_heatmap.png')
    plot_cooccurrence_heatmap(dataset, niche_attention_arr, logits_arr, class_map, figure_path_cooccur)
    print(f'Saved: {figure_path_cooccur}')

    sample_indices = list({r['sample_idx'] for records in top_by_class.values() for r in records})
    niche_attn_by_sample = {i: niche_attention_arr[i] for i in sample_indices}

    figure_path_topk = os.path.join(output_dir, 'niche_topk_prototypes.png')
    plot_niche_prototypes(
        records_by_class=top_by_class,
        dataset=dataset,
        class_map=class_map,
        output_path=figure_path_topk,
        max_hyperedges=args.max_hyperedges,
        niche_attn_by_sample=niche_attn_by_sample,
    )
    print(f'Saved: {figure_path_topk}')
