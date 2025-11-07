import os
import glob
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
from tqdm import tqdm


def _project_root_from_script_dir(script_dir: str) -> str:
    return os.path.dirname(script_dir) if os.path.basename(script_dir) == 'src' else script_dir


def resolve_dataset_path(dataset_path: str, script_dir: str) -> str:
    """Resolve dataset path relative to project root when given as relative path."""
    if os.path.isabs(dataset_path):
        return dataset_path
    project_root = _project_root_from_script_dir(script_dir)
    return os.path.join(project_root, dataset_path)


def create_unique_results_dir(model_tag: str, script_dir: str = None) -> str:
    """Create results/results_<model_tag>[N] under project root (dedup with numeric suffix).

    If script_dir is None, infer project root from this module location (src/common/..).
    """
    if script_dir is None:
        # src/common -> src -> project_root
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(module_dir))
    else:
        project_root = _project_root_from_script_dir(script_dir)
    base_results = os.path.join(project_root, 'results')
    os.makedirs(base_results, exist_ok=True)
    base_name = f"results_{model_tag}"
    candidate = os.path.join(base_results, base_name)
    if not os.path.exists(candidate):
        os.makedirs(candidate)
        return candidate
    suffix = 2
    while True:
        cand = os.path.join(base_results, f"{base_name}{suffix}")
        if not os.path.exists(cand):
            os.makedirs(cand)
            return cand
        suffix += 1


def subject_level_split(subjects: List[str], train_frac: float = 0.6, val_frac: float = 0.2):
    """Split subject ids into train/val/test sets by subject, return index lists and subject sets."""
    import random

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


class BraTSPreprocessor:
    """Preprocessing and patch extraction for BraTS volumes.

    Expects a config object with attributes used below; sensible defaults are used if missing.
    """

    def __init__(self, config):
        self.config = config

    # Defaults helper
    def _get(self, name: str, default):
        return getattr(self.config, name, default)

    def load_volume(self, subject_path: str, modality: str = 't1c') -> Tuple[np.ndarray, np.ndarray]:
        volume_file = glob.glob(os.path.join(subject_path, f"*-{modality}.nii.gz"))[0]
        seg_file = glob.glob(os.path.join(subject_path, "*-seg.nii.gz"))
        if len(seg_file) == 0:
            seg_file = glob.glob(os.path.join(subject_path, "*seg.nii.gz"))
        seg_file = seg_file[0]
        volume = nib.load(volume_file).get_fdata()
        segmentation = nib.load(seg_file).get_fdata()
        return volume, segmentation

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        volume = volume.astype(np.float32)
        non_zero_mask = volume > 0
        if np.sum(non_zero_mask) == 0:
            return volume
        v = volume[non_zero_mask]
        p1 = np.percentile(v, 1)
        p99 = np.percentile(v, 99)
        volume = np.clip(volume, p1, p99)
        volume[non_zero_mask] = (volume[non_zero_mask] - p1) / (p99 - p1 + 1e-8)
        return np.clip(volume, 0.0, 1.0)

    def _is_brain_tissue_patch(self, patch: np.ndarray) -> bool:
        max_background_intensity = self._get('max_background_intensity', 0.1)
        min_brain_tissue_ratio = self._get('min_brain_tissue_ratio', 0.3)
        min_brain_mean_intensity = self._get('min_brain_mean_intensity', 0.1)
        high_intensity_threshold = self._get('high_intensity_threshold', 0.9)
        max_high_intensity_ratio = self._get('max_high_intensity_ratio', 0.7)
        min_patch_std = self._get('min_patch_std', 0.01)

        brain_mask = patch > max_background_intensity
        if brain_mask.mean() < min_brain_tissue_ratio:
            return False
        vals = patch[brain_mask]
        if vals.size == 0:
            return False
        if vals.mean() < min_brain_mean_intensity:
            return False
        high_mask = patch > high_intensity_threshold
        if high_mask.mean() > max_high_intensity_ratio:
            return False
        if patch.std() < min_patch_std * 2:
            return False
        reasonable = ((patch > 0.05) & (patch < 0.95)).mean()
        return reasonable >= 0.5

    def _anomaly_ratio(self, seg_patch: np.ndarray) -> float:
        labels = np.array(self._get('anomaly_labels', [1, 2, 4]))
        return np.isin(seg_patch, labels).sum() / seg_patch.size

    def extract_normal_patches(self, volume: np.ndarray, seg: np.ndarray) -> List[np.ndarray]:
        patches: List[np.ndarray] = []
        max_background_intensity = self._get('max_background_intensity', 0.1)
        max_tumor_ratio_normal = self._get('max_tumor_ratio_normal', 0.01)
        min_patch_std = self._get('min_patch_std', 0.01)
        edge = self._get('edge_margin', 8)
        ps = self._get('patch_size', 32)
        max_normal_patches_per_subject = self._get('max_normal_patches_per_subject', 100)

        brain_mask = (volume > max_background_intensity) & (volume < 0.95)
        anomaly_mask = np.isin(seg, self._get('anomaly_labels', [1, 2, 4]))
        coords_normal = np.where(~anomaly_mask)
        coords_brain = np.where(brain_mask)
        valid = list(set(zip(*coords_normal)).intersection(set(zip(*coords_brain))))
        if not valid:
            return patches
        sx, sy, sz = volume.shape
        filtered = []
        for x, y, z in valid:
            if x < edge or y < edge or z < edge or x >= sx - edge or y >= sy - edge or z >= sz - edge:
                continue
            x0, y0, z0 = x - ps // 2, y - ps // 2, z - ps // 2
            x1, y1, z1 = x0 + ps, y0 + ps, z0 + ps
            if x0 >= 0 and y0 >= 0 and z0 >= 0 and x1 <= sx and y1 <= sy and z1 <= sz:
                filtered.append((x, y, z))
        if not filtered:
            return patches
        max_patches = min(len(filtered) // 20, max_normal_patches_per_subject)
        max_patches = max(max_patches, 10)
        sample_n = min(max_patches * 5, len(filtered))
        idxs = np.random.choice(len(filtered), size=sample_n, replace=False)
        for i in tqdm(idxs, desc="Extracting normal patches", leave=False):
            x, y, z = filtered[i]
            x0, y0, z0 = x - ps // 2, y - ps // 2, z - ps // 2
            x1, y1, z1 = x0 + ps, y0 + ps, z0 + ps
            patch = volume[x0:x1, y0:y1, z0:z1]
            seg_patch = seg[x0:x1, y0:y1, z0:z1]
            if self._anomaly_ratio(seg_patch) > max_tumor_ratio_normal:
                continue
            if not self._is_brain_tissue_patch(patch):
                continue
            if patch.std() < min_patch_std:
                continue
            patches.append(patch)
            if len(patches) >= max_patches:
                break
        return patches

    def extract_anomalous_patches(self, volume: np.ndarray, seg: np.ndarray) -> List[np.ndarray]:
        patches: List[np.ndarray] = []
        ps = self._get('patch_size', 32)
        max_anomaly_patches_per_subject = self._get('max_anomaly_patches_per_subject', 50)
        min_patch_std = self._get('min_patch_std', 0.01)
        min_patch_mean = self._get('min_patch_mean', 0.05)
        min_tumor_ratio_anomaly = self._get('min_tumor_ratio_anomaly', 0.05)
        anomaly_coords = np.where(np.isin(seg, self._get('anomaly_labels', [1, 2, 4])))
        if len(anomaly_coords[0]) == 0:
            return patches
        sx, sy, sz = volume.shape
        max_patches = min(len(anomaly_coords[0]) // 50, max_anomaly_patches_per_subject)
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
            if patch.std() <= min_patch_std or patch.mean() <= min_patch_mean:
                continue
            seg_patch = seg[x0:x1, y0:y1, z0:z1]
            if self._anomaly_ratio(seg_patch) >= min_tumor_ratio_anomaly:
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
        max_ratio = self._get('max_normal_to_anomaly_ratio', 3)
        max_normal = int(len(all_anom) * max_ratio) if len(all_anom) > 0 else len(all_normal)
        if len(all_normal) > max_normal and max_normal > 0:
            idx = np.random.choice(len(all_normal), max_normal, replace=False)
            all_normal = [all_normal[i] for i in idx]
            subj_normal = [subj_normal[i] for i in idx]
        patches = all_normal + all_anom
        labels = [0] * len(all_normal) + [1] * len(all_anom)
        subjects = subj_normal + subj_anom
        return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int64), subjects

# Simple patch quality validator (used by some pipelines for quick sanity checks)
def validate_patch_quality(patches: np.ndarray, labels: np.ndarray, verbose: bool = False):
    normal = patches[labels == 0]
    anomaly = patches[labels == 1]
    stats = {
        'normal_patches': {
            'count': int(len(normal)),
            'mean_intensity': float(np.mean([p.mean() for p in normal])) if len(normal) else 0.0,
            'std_intensity': float(np.std([p.mean() for p in normal])) if len(normal) else 0.0,
            'non_zero_ratio': float(np.mean([(p > 0).sum() / p.size for p in normal])) if len(normal) else 0.0,
        },
        'anomaly_patches': {
            'count': int(len(anomaly)),
            'mean_intensity': float(np.mean([p.mean() for p in anomaly])) if len(anomaly) else 0.0,
            'std_intensity': float(np.std([p.mean() for p in anomaly])) if len(anomaly) else 0.0,
            'non_zero_ratio': float(np.mean([(p > 0).sum() / p.size for p in anomaly])) if len(anomaly) else 0.0,
        },
    }
    if verbose:
        print("Normal patches statistics:")
        print(f"  Count: {stats['normal_patches']['count']}")
        print(f"  Mean intensity: {stats['normal_patches']['mean_intensity']:.4f}")
        print(f"  Non-zero ratio: {stats['normal_patches']['non_zero_ratio']:.4f}")
        print("Anomaly patches statistics:")
        print(f"  Count: {stats['anomaly_patches']['count']}")
        print(f"  Mean intensity: {stats['anomaly_patches']['mean_intensity']:.4f}")
        print(f"  Non-zero ratio: {stats['anomaly_patches']['non_zero_ratio']:.4f}")
    return stats
