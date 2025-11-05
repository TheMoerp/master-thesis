#!/usr/bin/env python3
"""
Anatomix feature extractor + KNN unsupervised anomaly detection on BraTS

Requirements derived from user brief:
- Use EXACT SAME preprocessing/patch pipeline and CLI parameter style as ae_brats.py
- Use Anatomix as feature extractor on patches
- Train KNN anomaly detector in a truly unsupervised way:
  - Train/Val on NORMAL patches only (subject-level split)
  - Determine threshold from normal validation scores only
  - Test on mixed patches
- Default anomaly labels: [1, 4] (NCR/NET and Enhancing Tumor), but keep the same CLI (allow override)
- Keep same metrics and report style as ae_brats.py
"""

import os
import sys
import glob
import argparse
import time
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
from sklearn.manifold import TSNE


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Config:
    """Mirror ae_brats.py config but adapt for KNN + Anatomix."""
    def __init__(self):
        # Dataset paths
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "anatomix_knn_results"

        # Patch extraction parameters (same semantics as ae_brats.py)
        self.patch_size = 32
        self.patches_per_volume = 50
        self.min_non_zero_ratio = 0.2
        self.max_normal_to_anomaly_ratio = 3
        self.min_tumor_ratio_in_patch = 0.05

        # Patch quality parameters (mirroring ae_brats.py)
        self.min_patch_std = 0.01
        self.min_patch_mean = 0.05
        self.max_tumor_ratio_normal = 0.01
        self.min_tumor_ratio_anomaly = 0.05
        self.max_normal_patches_per_subject = 100
        self.max_anomaly_patches_per_subject = 50

        # Default anomaly labels per user: [1, 4] (NCR/NET, ET)
        self.anomaly_labels = [1, 4]

        # Brain tissue quality parameters
        self.min_brain_tissue_ratio = 0.3
        self.max_background_intensity = 0.1
        self.min_brain_mean_intensity = 0.1
        self.max_high_intensity_ratio = 0.7
        self.high_intensity_threshold = 0.9
        self.edge_margin = 8

        # Splits
        self.train_test_split = 0.8
        self.validation_split = 0.2

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if torch.cuda.is_available() else 0

        # Anatomix + feature parameters
        self.anatomix_batch_size = 4
        self.feature_pooling = "mean_max_std"  # fixed pooling scheme

        # KNN params
        self.k_neighbors = 7

        # Threshold percentile for validation NORMAL scores
        self.threshold_percentile = 90.0

        os.makedirs(self.output_dir, exist_ok=True)


class BraTSDataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def load_volume(self, subject_path: str, modality: str = 't1c') -> Tuple[np.ndarray, np.ndarray]:
        volume_file = glob.glob(os.path.join(subject_path, f"*-{modality}.nii.gz"))[0]
        seg_file = glob.glob(os.path.join(subject_path, "*-seg.nii.gz"))[0]
        volume = nib.load(volume_file).get_fdata()
        segmentation = nib.load(seg_file).get_fdata()
        return volume, segmentation

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        volume = volume.astype(np.float32)
        non_zero = volume > 0
        if non_zero.sum() == 0:
            return volume
        v = volume[non_zero]
        low, high = np.percentile(v, 1), np.percentile(v, 99)
        volume = np.clip(volume, low, high)
        volume[non_zero] = (volume[non_zero] - low) / (high - low + 1e-8)
        return np.clip(volume, 0.0, 1.0)

    def get_anomaly_ratio_in_patch(self, seg_patch: np.ndarray) -> float:
        return np.isin(seg_patch, self.config.anomaly_labels).sum() / seg_patch.size

    def is_brain_tissue_patch(self, patch: np.ndarray) -> bool:
        brain_mask = patch > self.config.max_background_intensity
        if brain_mask.mean() < self.config.min_brain_tissue_ratio:
            return False
        vals = patch[brain_mask]
        if vals.size == 0:
            return False
        if vals.mean() < self.config.min_brain_mean_intensity:
            return False
        high_mask = patch > self.config.high_intensity_threshold
        if high_mask.mean() > self.config.max_high_intensity_ratio:
            return False
        if patch.std() < self.config.min_patch_std * 2:
            return False
        reasonable = ((patch > 0.05) & (patch < 0.95)).mean()
        return reasonable >= 0.5

    def extract_normal_patches(self, volume: np.ndarray, seg: np.ndarray) -> List[np.ndarray]:
        patches = []
        brain_mask = (volume > self.config.max_background_intensity) & (volume < 0.95)
        anomaly_mask = np.isin(seg, self.config.anomaly_labels)
        coords_normal = np.where(~anomaly_mask)
        coords_brain = np.where(brain_mask)
        set_normal = set(zip(*coords_normal))
        set_brain = set(zip(*coords_brain))
        valid = list(set_normal.intersection(set_brain))
        if len(valid) == 0:
            return patches
        edge = self.config.edge_margin
        filtered = []
        sx, sy, sz = volume.shape
        ps = self.config.patch_size
        for x, y, z in valid:
            if x < edge or y < edge or z < edge or x >= sx - edge or y >= sy - edge or z >= sz - edge:
                continue
            x0, y0, z0 = x - ps // 2, y - ps // 2, z - ps // 2
            x1, y1, z1 = x0 + ps, y0 + ps, z0 + ps
            if x0 >= 0 and y0 >= 0 and z0 >= 0 and x1 <= sx and y1 <= sy and z1 <= sz:
                filtered.append((x, y, z))
        if len(filtered) == 0:
            return patches
        max_patches = min(len(filtered) // 20, self.config.max_normal_patches_per_subject)
        max_patches = max(max_patches, 10)
        sample_n = min(max_patches * 5, len(filtered))
        idxs = np.random.choice(len(filtered), size=sample_n, replace=False)
        for i in tqdm(idxs, desc="Extracting normal patches", leave=False):
            x, y, z = filtered[i]
            x0, y0, z0 = x - ps // 2, y - ps // 2, z - ps // 2
            x1, y1, z1 = x0 + ps, y0 + ps, z0 + ps
            patch = volume[x0:x1, y0:y1, z0:z1]
            seg_patch = seg[x0:x1, y0:y1, z0:z1]
            if self.get_anomaly_ratio_in_patch(seg_patch) > self.config.max_tumor_ratio_normal:
                continue
            if not self.is_brain_tissue_patch(patch):
                continue
            if patch.std() < self.config.min_patch_std:
                continue
            patches.append(patch)
            if len(patches) >= max_patches:
                break
        return patches

    def extract_anomalous_patches(self, volume: np.ndarray, seg: np.ndarray) -> List[np.ndarray]:
        patches = []
        anomaly_coords = np.where(np.isin(seg, self.config.anomaly_labels))
        if len(anomaly_coords[0]) == 0:
            return patches
        ps = self.config.patch_size
        sx, sy, sz = volume.shape
        max_patches = min(len(anomaly_coords[0]) // 50, self.config.max_anomaly_patches_per_subject)
        if max_patches == 0:
            return patches
        idxs = np.random.choice(len(anomaly_coords[0]), size=min(max_patches, len(anomaly_coords[0])), replace=False)
        for i in tqdm(idxs, desc="Extracting anomaly patches", leave=False):
            x, y, z = anomaly_coords[0][i], anomaly_coords[1][i], anomaly_coords[2][i]
            x0, y0, z0 = max(0, x - ps // 2), max(0, y - ps // 2), max(0, z - ps // 2)
            x1, y1, z1 = min(sx, x0 + ps), min(sy, y0 + ps), min(sz, z0 + ps)
            if (x1 - x0 != ps) or (y1 - y0 != ps) or (z1 - z0 != ps):
                continue
            patch = volume[x0:x1, y0:y1, z0:z1]
            if patch.std() <= self.config.min_patch_std or patch.mean() <= self.config.min_patch_mean:
                continue
            seg_patch = seg[x0:x1, y0:y1, z0:z1]
            if self.get_anomaly_ratio_in_patch(seg_patch) >= self.config.min_tumor_ratio_anomaly:
                patches.append(patch)
        return patches

    def process_dataset(self, num_subjects: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        subject_dirs = [d for d in os.listdir(self.config.dataset_path) if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        if num_subjects:
            subject_dirs = subject_dirs[:num_subjects]
        all_normal, all_anom, subj_normal, subj_anom = [], [], [], []
        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            sp = os.path.join(self.config.dataset_path, subject_dir)
            try:
                vol, seg = self.load_volume(sp)
                vol = self.normalize_volume(vol)
                normals = self.extract_normal_patches(vol, seg)
                anoms = self.extract_anomalous_patches(vol, seg)
                all_normal.extend(normals)
                all_anom.extend(anoms)
                subj_normal.extend([subject_dir] * len(normals))
                subj_anom.extend([subject_dir] * len(anoms))
            except Exception:
                continue
        # Balance
        max_normal = int(len(all_anom) * self.config.max_normal_to_anomaly_ratio) if len(all_anom) > 0 else len(all_normal)
        if len(all_normal) > max_normal and max_normal > 0:
            idx = np.random.choice(len(all_normal), max_normal, replace=False)
            all_normal = [all_normal[i] for i in idx]
            subj_normal = [subj_normal[i] for i in idx]
        patches = all_normal + all_anom
        labels = [0] * len(all_normal) + [1] * len(all_anom)
        subjects = subj_normal + subj_anom
        return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int64), subjects


def install_anatomix_if_needed():
    # Prefer local repo to avoid Python version constraints
    repo_path = os.path.join(os.path.dirname(__file__), 'anatomix')
    pkg_root = os.path.join(repo_path, 'anatomix')
    if os.path.isdir(pkg_root):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from anatomix.model.network import Unet  # noqa: F401
            return True
        except Exception:
            pass
    # Fallback: try editable install if local not importable
    try:
        from anatomix.model.network import Unet  # noqa: F401
        return True
    except Exception:
        print("Could not import anatomix. Ensure the local repo exists at ./anatomix or install it.")
        return False


class AnatomixFeatureExtractor:
    def __init__(self, device: torch.device, pooling: str = "mean_max_std"):
        self.device = device
        self.pooling = pooling
        from anatomix.model.network import Unet
        # Initialize UNet per anatomix README and load weights
        self.model = Unet(
            dimension=3,
            input_nc=1,
            output_nc=16,
            num_downs=4,
            ngf=16,
        )
        # Load weights from local repo if available
        repo_path = os.path.join(os.path.dirname(__file__), 'anatomix')
        weights_primary = os.path.join(repo_path, 'model-weights', 'anatomix.pth')
        weights_brains = os.path.join(repo_path, 'model-weights', 'anatomix+brains.pth')
        weights_path = weights_primary if os.path.isfile(weights_primary) else (weights_brains if os.path.isfile(weights_brains) else None)
        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)

    @staticmethod
    def _standardize_size(patch: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        out = np.zeros(target_size, dtype=patch.dtype)
        sx, sy, sz = patch.shape
        tx, ty, tz = target_size
        xs = max(0, (sx - tx) // 2); xe = xs + min(tx, sx)
        ys = max(0, (sy - ty) // 2); ye = ys + min(ty, sy)
        zs = max(0, (sz - tz) // 2); ze = zs + min(tz, sz)
        xsp = max(0, (tx - sx) // 2); ysp = max(0, (ty - sy) // 2); zsp = max(0, (tz - sz) // 2)
        out[xsp:xsp + (xe - xs), ysp:ysp + (ye - ys), zsp:zsp + (ze - zs)] = patch[xs:xe, ys:ye, zs:ze]
        return out

    def extract_batch(self, patches: List[np.ndarray], patch_size: int) -> np.ndarray:
        # Prepare batch tensor
        standardized = [self._standardize_size(p, (patch_size, patch_size, patch_size)) for p in patches]
        x = torch.from_numpy(np.stack(standardized)[..., None].transpose(0, 4, 1, 2, 3)).float().to(self.device)
        with torch.no_grad():
            # Use encoder-only (bottleneck) features from Anatomix UNet
            feats = self.model(x, encode_only=True)  # Expect shape [B, C, D, H, W]
        f = feats.detach().cpu().numpy()
        # Move to (B, D, H, W, C)
        f = np.transpose(f, (0, 2, 3, 4, 1))
        # Global pooling
        mean = f.mean(axis=(1, 2, 3))
        mx = f.max(axis=(1, 2, 3))
        sd = f.std(axis=(1, 2, 3))
        return np.concatenate([mean, mx, sd], axis=1)

    def extract_all(self, patches: np.ndarray, batch_size: int, patch_size: int) -> np.ndarray:
        features = []
        for i in tqdm(range(0, len(patches), batch_size), desc="Extracting Anatomix features"):
            batch = patches[i:i + batch_size]
            feats = self.extract_batch(list(batch), patch_size)
            features.append(feats)
        return np.vstack(features) if len(features) else np.zeros((0,))


def subject_level_split(subjects: List[str], labels: np.ndarray, train_frac=0.6, val_frac=0.2):
    uniq = list(set(subjects))
    random.shuffle(uniq)
    n = len(uniq)
    train_subj = set(uniq[:int(train_frac * n)])
    val_subj = set(uniq[int(train_frac * n):int((train_frac + val_frac) * n)])
    test_subj = set(uniq[int((train_frac + val_frac) * n):])
    idx_train = [i for i, s in enumerate(subjects) if s in train_subj]
    idx_val = [i for i, s in enumerate(subjects) if s in val_subj]
    idx_test = [i for i, s in enumerate(subjects) if s in test_subj]
    return idx_train, idx_val, idx_test, train_subj, val_subj, test_subj


def aggregate_channel_scores_supervised(features: np.ndarray, labels: np.ndarray, num_base_channels: int) -> np.ndarray:
    """Compute per-base-channel discriminability via AUC, aggregating mean/max/std dims.
    features shape: [N, num_base_channels*3] ordered as [mean C][max C][std C].
    Returns: channel_scores [C], NaN-safe with 0.0 fallback.
    """
    scores = []
    C = num_base_channels
    for c in range(C):
        aucs = []
        for block in range(3):  # mean, max, std
            dim_idx = block * C + c
            try:
                aucs.append(roc_auc_score(labels, features[:, dim_idx]))
            except Exception:
                aucs.append(0.0)
        scores.append(float(np.nanmean(aucs)))
    return np.array(scores)


def select_channels_unsupervised(train_features: np.ndarray, num_base_channels: int, top_k: int,
                                 corr_threshold: float = 0.95) -> List[int]:
    """Unsupervised channel selection using variance ranking + correlation pruning.
    Returns list of selected base-channel indices (length <= top_k).
    """
    C = num_base_channels
    # Compute per-channel variance (aggregate across mean/max/std dims)
    variances = []
    for c in range(C):
        ch_dims = train_features[:, [c, c + C, c + 2 * C]]
        variances.append(np.var(ch_dims, axis=0).mean())
    order = np.argsort(variances)[::-1]  # high variance first
    selected = []
    # Build channel representation vector by concatenating the three dims
    ch_vecs = [train_features[:, [c, c + C, c + 2 * C]].reshape(len(train_features), 3) for c in range(C)]
    # Normalize vectors for correlation computation
    ch_vecs = [((v - v.mean(axis=0)) / (v.std(axis=0) + 1e-8)).reshape(len(train_features), -1) for v in ch_vecs]
    for c in order:
        if len(selected) >= top_k:
            break
        ok = True
        for s in selected:
            # compute Pearson correlation between flattened vectors
            a = ch_vecs[c].reshape(-1)
            b = ch_vecs[s].reshape(-1)
            corr = np.corrcoef(a, b)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            if abs(corr) >= corr_threshold:
                ok = False
                break
        if ok:
            selected.append(int(c))
    return selected


def select_channels_pca_unsupervised(train_features: np.ndarray, num_base_channels: int, top_k: int,
                                     var_explained: float = 0.95, corr_threshold: float = 0.95) -> List[int]:
    """Unsupervised channel selection via PCA saliency + correlation pruning.
    Score each base channel by cumulative squared PCA loadings over components
    explaining var_explained of variance, then prune highly correlated channels.
    """
    C = num_base_channels
    # Standardize features
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(train_features)
    # Fit PCA
    pca = PCA(svd_solver='auto', random_state=42)
    pca.fit(X)
    # Determine number of components to reach var_explained
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, var_explained) + 1)
    comps = pca.components_[:n_comp]  # shape [n_comp, D]
    D = X.shape[1]
    assert D == C * 3, "Feature dimensionality must be 3x base channels (mean,max,std)."
    # Saliency per channel = sum over components of sum of squared loadings for the 3 dims
    saliency = np.zeros(C, dtype=np.float64)
    for c in range(C):
        idxs = [c, c + C, c + 2 * C]
        load = comps[:, idxs]
        saliency[c] = float(np.sum(load ** 2))
    order = np.argsort(saliency)[::-1]
    # Correlation pruning using same representation as variance method
    ch_vecs = [train_features[:, [c, c + C, c + 2 * C]].reshape(len(train_features), 3) for c in range(C)]
    ch_vecs = [((v - v.mean(axis=0)) / (v.std(axis=0) + 1e-8)).reshape(len(train_features), -1) for v in ch_vecs]
    selected = []
    for c in order:
        if len(selected) >= top_k:
            break
        ok = True
        for s in selected:
            a = ch_vecs[c].reshape(-1)
            b = ch_vecs[s].reshape(-1)
            corr = np.corrcoef(a, b)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            if abs(corr) >= corr_threshold:
                ok = False
                break
        if ok:
            selected.append(int(c))
    return selected


def indices_for_selected_channels(selected_channels: List[int], num_base_channels: int) -> List[int]:
    """Map base-channel indices to feature indices including mean/max/std blocks."""
    C = num_base_channels
    idxs = []
    for c in selected_channels:
        idxs.extend([c, c + C, c + 2 * C])
    return idxs


def fit_knn_on_normal(train_features: np.ndarray, k: int):
    # Use sklearn NearestNeighbors for simplicity here
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, len(train_features)), algorithm='auto')
    nbrs.fit(train_features)
    return nbrs


def knn_anomaly_scores(nbrs, ref_features: np.ndarray, query_features: np.ndarray, k: int) -> np.ndarray:
    # Compute distances to kNN in reference (training) set; anomaly score = mean kNN distance
    distances, _ = nbrs.kneighbors(query_features, n_neighbors=min(k, len(ref_features)))
    return distances.mean(axis=1)


def threshold_from_normal(val_scores: np.ndarray, percentile: float) -> float:
    # Unsupervised: percentile on validation NORMAL scores (e.g., 90 for p90)
    return float(np.percentile(val_scores, percentile))


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    preds = (scores > threshold).astype(int)
    try:
        roc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)
    except Exception:
        roc, ap = 0.0, 0.0
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = (sensitivity + specificity) / 2
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    dsc = (2 * tp) / max(2 * tp + fp + fn, 1)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {
        'roc_auc': roc,
        'average_precision': ap,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'balanced_accuracy': bal_acc,
        'mcc': mcc,
        'dsc': dsc,
        'fpr': fpr,
        'fnr': fnr,
        'predictions': preds,
    }


def save_results(config: Config, results: Dict, y_true: np.ndarray, scores: np.ndarray, threshold: float,
                 train_n: int, val_n: int, test_n: int,
                 train_subjects_n: int, val_subjects_n: int, test_subjects_n: int):
    # Minimal text report aligned with ae_brats style naming
    out = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(out, 'w') as f:
        f.write("TRULY UNSUPERVISED Anatomix+KNN Anomaly Detection Results\n")
        f.write("="*60 + "\n")
        f.write(f"ROC AUC:           {results['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {results['average_precision']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"F1 Score:          {results['f1_score']:.4f}\n")
        f.write(f"Precision:         {results['precision']:.4f}\n")
        f.write(f"Recall:            {results['recall']:.4f}\n")
        f.write(f"Specificity:       {(2*results['balanced_accuracy'] - results['recall']):.4f}\n")
        f.write(f"Accuracy:          {results['accuracy']:.4f}\n")
        f.write(f"MCC:               {results['mcc']:.4f}\n")
        f.write(f"DSC:               {results['dsc']:.4f}\n")
        f.write(f"Threshold Used:    {threshold:.6f}\n")
        f.write("="*60 + "\n")
        f.write("Configuration:\n")
        f.write(f"  Patch size: {config.patch_size}\n")
        f.write(f"  Patches per volume: {config.patches_per_volume}\n")
        f.write(f"  K neighbors: {config.k_neighbors}\n")
        f.write(f"  Threshold percentile: {config.threshold_percentile}\n")
        f.write(f"  Training samples: {train_n}\n")
        f.write(f"  Validation samples: {val_n}\n")
        f.write(f"  Test samples: {test_n}\n")
        f.write(f"  Training subjects: {train_subjects_n}\n")
        f.write(f"  Validation subjects: {val_subjects_n}\n")
        f.write(f"  Test subjects: {test_subjects_n}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")
    print(f"Results saved to: {out}")


class Visualizer:
    def __init__(self, config: Config):
        self.config = config

    def plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray):
        cm = confusion_matrix(true_labels, predictions)
        if cm.shape != (2, 2):
            return
        tn, fp, fn, tp = cm.ravel()
        z = [[tn, fp], [fn, tp]]
        x = ['Normal (0)', 'Anomaly (1)']
        y = ['Normal (0)', 'Anomaly (1)']
        row_sums = cm.sum(axis=1)
        z_text = [
            [f"{tn}<br>({tn/row_sums[0]*100:.1f}%)" if row_sums[0] > 0 else str(tn),
             f"{fp}<br>({fp/row_sums[0]*100:.1f}%)" if row_sums[0] > 0 else str(fp)],
            [f"{fn}<br>({fn/row_sums[1]*100:.1f}%)" if row_sums[1] > 0 else str(fn),
             f"{tp}<br>({tp/row_sums[1]*100:.1f}%)" if row_sums[1] > 0 else str(tp)]
        ]
        fig = ff.create_annotated_heatmap(
            z, x=x, y=y, annotation_text=z_text, colorscale='Blues',
            font_colors=['black', 'white']
        )
        fig.update_layout(
            title_text='<b>Confusion Matrix</b><br>(Count and Percentage)',
            title_x=0.5,
            xaxis=dict(title='<b>Predicted Label</b>'),
            yaxis=dict(title='<b>True Label</b>', autorange='reversed'),
            font=dict(size=14)
        )
        fig.update_xaxes(side="bottom")
        output_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        try:
            fig.write_image(output_path, width=800, height=800, scale=2)
        except ValueError:
            pass

    def plot_roc_curve(self, true_labels: np.ndarray, scores: np.ndarray):
        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc_value = roc_auc_score(true_labels, scores)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc_value:.2f})")
        plt.fill_between(fpr, tpr, step="pre", alpha=0.25, color="#aec7e8")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1, label="Chance")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        plt.savefig(os.path.join(self.config.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curve(self, true_labels: np.ndarray, scores: np.ndarray):
        precision, recall, _ = precision_recall_curve(true_labels, scores)
        pr_auc_value = auc(recall, precision)
        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc_value:.2f})")
        plt.fill_between(recall, precision, step="pre", alpha=0.25, color="#ffbb78")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        plt.savefig(os.path.join(self.config.output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_score_histogram(self, scores: np.ndarray, true_labels: np.ndarray, threshold: float):
        normal_scores = scores[true_labels == 0]
        anomaly_scores = scores[true_labels == 1]
        plt.figure(figsize=(12, 6))
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.6f}')
        plt.xlabel('Anomaly Score (kNN distance)')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Fix axes to consistent ranges as requested
        plt.xlim(0, 2.5)
        plt.ylim(0, 3.5)
        # Set major ticks at every 0.5 on both axes
        plt.xticks(np.arange(0.0, 2.5 + 0.001, 0.5))
        plt.yticks(np.arange(0.0, 3.5 + 0.001, 0.5))
        plt.savefig(os.path.join(self.config.output_dir, 'score_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_latent_space_visualization(self, features: np.ndarray, true_labels: np.ndarray):
        # Mirror ae_brats.py visualization (PCA + t-SNE) and styling
        print("Creating latent space visualizations...")

        feats = features
        labels = true_labels

        if len(feats) == 0 or feats.ndim != 2:
            return

        # Limit samples for visualization
        if len(feats) > 2000:
            idx = np.random.choice(len(feats), 2000, replace=False)
            feats = feats[idx]
            labels = labels[idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # PCA
        print("Computing PCA...")
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(feats)

        ax1.scatter(pca_features[labels == 0, 0], pca_features[labels == 0, 1],
                    c='blue', alpha=0.6, label='Normal', s=20)
        ax1.scatter(pca_features[labels == 1, 0], pca_features[labels == 1, 1],
                    c='red', alpha=0.6, label='Anomaly', s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA of Latent Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(feats)//4)))
        tsne_features = tsne.fit_transform(feats)

        ax2.scatter(tsne_features[labels == 0, 0], tsne_features[labels == 0, 1],
                    c='blue', alpha=0.6, label='Normal', s=20)
        ax2.scatter(tsne_features[labels == 1, 0], tsne_features[labels == 1, 1],
                    c='red', alpha=0.6, label='Anomaly', s=20)
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.set_title('t-SNE of Latent Features')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'latent_space_visualization.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def main():
    set_seeds(42)

    parser = argparse.ArgumentParser(description='Anatomix+KNN Unsupervised Anomaly Detection for BraTS')
    parser.add_argument('--num_subjects', type=int, default=None, help='Number of subjects to use (default: all)')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of 3D patches (default: 32)')
    parser.add_argument('--patches_per_volume', type=int, default=50, help='Number of patches per volume (default: 50)')
    parser.add_argument('--output_dir', type=str, default='anatomix_knn_results', help='Output directory')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData', help='Path to BraTS dataset')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3, help='Max ratio of normal to anomaly patches')
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05, help='Minimum tumor ratio in anomalous patches')
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 4], help='BraTS labels to consider anomalous (default: 1 4)')
    parser.add_argument('--k_neighbors', type=int, default=7, help='Number of neighbors for KNN')
    parser.add_argument('--threshold_percentile', type=float, default=90.0, help='Percentile for threshold on validation NORMAL scores (0-100)')
    parser.add_argument('--select_topk_channels', type=int, default=0, help='Select top-K Anatomix base channels (0 = use all)')
    parser.add_argument('--channel_selection_mode', type=str, choices=['supervised', 'unsupervised', 'pca'], default='supervised', help='Channel selection strategy')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    cfg = Config()
    cfg.num_subjects = args.num_subjects
    cfg.patch_size = args.patch_size
    cfg.patches_per_volume = args.patches_per_volume
    cfg.output_dir = args.output_dir
    cfg.dataset_path = args.dataset_path
    cfg.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    cfg.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    cfg.anomaly_labels = args.anomaly_labels
    cfg.k_neighbors = args.k_neighbors
    cfg.threshold_percentile = args.threshold_percentile
    cfg.verbose = args.verbose
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Anatomix+KNN | Anomaly labels: {cfg.anomaly_labels} | Output: {cfg.output_dir}")

    # 1) Extract patches with ae_brats-like processor
    processor = BraTSDataProcessor(cfg)
    patches, labels, subjects = processor.process_dataset(cfg.num_subjects)
    if len(patches) == 0:
        print("No patches extracted. Exiting.")
        return

    # 2) Subject-level split (train/val use only normal patches)
    idx_train, idx_val, idx_test, train_subj, val_subj, test_subj = subject_level_split(subjects, labels)
    X_train_all, y_train_all = patches[idx_train], labels[idx_train]
    X_val_all, y_val_all = patches[idx_val], labels[idx_val]
    X_test, y_test = patches[idx_test], labels[idx_test]

    train_normal_mask = (y_train_all == 0)
    val_normal_mask = (y_val_all == 0)
    X_train_normal = X_train_all[train_normal_mask]
    X_val_normal = X_val_all[val_normal_mask]

    print(f"Train normal: {len(X_train_normal)} | Val normal: {len(X_val_normal)} | Test total: {len(X_test)}")

    # 3) Anatomix feature extraction
    if not install_anatomix_if_needed():
        return
    extractor = AnatomixFeatureExtractor(cfg.device, pooling=cfg.feature_pooling)
    train_feats = extractor.extract_all(X_train_normal, cfg.anatomix_batch_size, cfg.patch_size)
    val_feats = extractor.extract_all(X_val_normal, cfg.anatomix_batch_size, cfg.patch_size)
    test_feats = extractor.extract_all(X_test, cfg.anatomix_batch_size, cfg.patch_size)
    # For supervised channel selection, we need mixed validation features (not only normals)
    val_all_feats = None
    if args.select_topk_channels and args.select_topk_channels > 0 and args.channel_selection_mode == 'supervised':
        val_all_feats = extractor.extract_all(X_val_all, cfg.anatomix_batch_size, cfg.patch_size)

    # Optional: channel selection on Anatomix base channels
    selected_channel_indices = None
    # Infer number of base channels from pooled feature dimensionality (mean/max/std blocks)
    base_channels = int(test_feats.shape[1] // 3) if test_feats.ndim == 2 else 16
    if cfg.anatomix_batch_size and cfg.patch_size:  # no-op guard
        if args.select_topk_channels and args.select_topk_channels > 0:
            top_k = min(args.select_topk_channels, base_channels)
            if args.channel_selection_mode == 'supervised':
                # Use validation set (mixed) to avoid test leakage
                ch_scores = aggregate_channel_scores_supervised(val_all_feats, y_val_all, base_channels)
                top_ch = np.argsort(ch_scores)[-top_k:][::-1].tolist()
            elif args.channel_selection_mode == 'unsupervised':
                top_ch = select_channels_unsupervised(train_feats, base_channels, top_k)
            elif args.channel_selection_mode == 'pca':
                top_ch = select_channels_pca_unsupervised(train_feats, base_channels, top_k)
            else:
                top_ch = list(range(base_channels))[:top_k]
            selected_channel_indices = indices_for_selected_channels(top_ch, base_channels)
            # Apply sub-selection
            train_feats = train_feats[:, selected_channel_indices]
            val_feats = val_feats[:, selected_channel_indices]
            test_feats = test_feats[:, selected_channel_indices]
            print(f"Selected base channels (mode={args.channel_selection_mode}): {top_ch}")

    # 4) Fit KNN on normal train features
    knn = fit_knn_on_normal(train_feats, cfg.k_neighbors)

    # 5) Compute scores
    val_scores = knn_anomaly_scores(knn, train_feats, val_feats, cfg.k_neighbors)
    test_scores = knn_anomaly_scores(knn, train_feats, test_feats, cfg.k_neighbors)

    # 6) Threshold from normal validation
    thr = threshold_from_normal(val_scores, cfg.threshold_percentile)

    # 7) Evaluate on test
    results = evaluate_scores(y_test, test_scores, thr)

    # 8) Save results
    save_results(
        cfg, results, y_test, test_scores, thr,
        train_n=len(X_train_normal), val_n=len(X_val_normal), test_n=len(X_test),
        train_subjects_n=len(train_subj), val_subjects_n=len(val_subj), test_subjects_n=len(test_subj)
    )

    # 9) Visualizations (same style/colors as ae_brats)
    vis = Visualizer(cfg)
    vis.plot_confusion_matrix(y_test, results['predictions'])
    vis.plot_roc_curve(y_test, test_scores)
    vis.plot_precision_recall_curve(y_test, test_scores)
    vis.plot_score_histogram(test_scores, y_test, thr)
    vis.plot_latent_space_visualization(test_feats, y_test)


if __name__ == "__main__":
    main()


