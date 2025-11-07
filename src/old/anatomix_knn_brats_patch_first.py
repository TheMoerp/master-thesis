#!/usr/bin/env python3
"""
CORRECTED: Anatomix Feature-based KNN Anomaly Detection for BraTS Dataset
PATCH-FIRST WORKFLOW IMPLEMENTATION

CRITICAL CORRECTION APPLIED:
- Extract 3D patches from brain volumes FIRST
- THEN apply anatomix feature extraction to each patch
- Keep all brain tissue validation and quality checks
- Maintain subject-level splitting to prevent data leakage
- True unsupervised anomaly detection approach

CORRECTED WORKFLOW:
1. Load BraTS volumes and segmentations
2. Extract 3D patches with brain tissue validation
3. Apply anatomix feature extraction to patches
4. Subject-level split (no data leakage)
5. Train KNN on normal patches only
6. Evaluate with comprehensive metrics

Author: Data Science Analysis (CORRECTED)
Date: December 2024
"""

import os
import sys
import glob
import argparse
import random
import warnings
import time
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import contextlib

import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

import faiss

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Install and import anatomix
def install_anatomix():
    if not os.path.exists("anatomix"):
        print("Cloning anatomix repository...")
        os.system("git clone https://github.com/neel-dey/anatomix.git")
        os.chdir("anatomix")
        os.system("pip install -e .")
        os.chdir("..")

try:
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            from anatomix.anatomix.model.network import Unet
except ImportError:
    install_anatomix()
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            from anatomix.anatomix.model.network import Unet

class Config:
    """Configuration for corrected patch-first anatomix processing"""
    
    def __init__(self):
        # Dataset paths
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "anatomix_knn_brats_patch_first_results"
        self.features_dir = "anatomix_features_patch_first"
        
        # CORRECTED: Patch extraction parameters  
        self.patch_size = 32  # Must be divisible by 16 for anatomix
        self.edge_margin = 16  # Safe margin from volume edges
        
        # Brain tissue quality parameters (from original ae_brats.py)
        self.min_brain_tissue_ratio = 0.4
        self.max_background_intensity = 0.05
        self.min_brain_mean_intensity = 0.1
        self.high_intensity_threshold = 0.9
        self.max_high_intensity_ratio = 0.1
        self.min_patch_std = 0.01
        self.min_patch_mean = 0.05
        self.min_non_zero_ratio = 0.3
        
        # Patch selection parameters
        self.max_normal_patches_per_subject = 50
        self.max_anomaly_patches_per_subject = 25
        self.max_tumor_ratio_normal = 0.01   # 1% max tumor in normal patches
        self.min_tumor_ratio_anomaly = 0.05  # 5% min tumor in anomaly patches
        
        # Segmentation labels
        self.anomaly_labels = [1, 2, 4]  # NCR/NET, ED, ET
        
        # KNN parameters
        self.k_neighbors = 5
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

def minmax_normalize(arr):
    """Robust min-max normalization"""
    arr_min, arr_max = np.percentile(arr[arr > 0], [1, 99])
    normalized = np.clip(arr, arr_min, arr_max)
    normalized = (normalized - arr_min) / (arr_max - arr_min + 1e-8)
    normalized[arr == 0] = 0
    return normalized

def load_anatomix_model():
    """Load pre-trained anatomix model"""
    print("Loading anatomix U-Net model...")
    
    model = Unet(
        dimension=3,
        input_nc=1,
        output_nc=16,
        num_downs=4,
        ngf=16,
    )
    
    weights_path = "anatomix/model-weights/anatomix.pth"
    if not os.path.exists(weights_path):
        os.makedirs("anatomix/model-weights", exist_ok=True)
        print("Downloading anatomix weights...")
        os.system(f"wget -O {weights_path} https://github.com/neel-dey/anatomix/raw/main/model-weights/anatomix.pth")
    
    model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=True)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    
    print(f"✓ Anatomix model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    return model

class PatchFirstBraTSProcessor:
    """CORRECTED: Patch-first BraTS processor with all brain tissue validations"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_subject_data(self, subject_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load T1c and segmentation for a subject"""
        try:
            t1c_path = glob.glob(os.path.join(subject_path, "*-t1c.nii.gz"))[0]
            seg_path = glob.glob(os.path.join(subject_path, "*-seg.nii.gz"))[0]
            
            t1c = nib.load(t1c_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()
            
            return t1c, seg
        except (IndexError, FileNotFoundError) as e:
            raise ValueError(f"Could not load data for {subject_path}: {e}")
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Enhanced volume normalization (from ae_brats.py)"""
        brain_mask = volume > 0
        
        if np.sum(brain_mask) == 0:
            print("Warning: No brain tissue found in volume")
            return volume
        
        brain_voxels = volume[brain_mask]
        p1, p99 = np.percentile(brain_voxels, [1, 99])
        
        volume_clipped = np.clip(volume, p1, p99)
        volume_normalized = (volume_clipped - p1) / (p99 - p1 + 1e-8)
        volume_normalized[~brain_mask] = 0
        
        return volume_normalized
    
    def is_brain_tissue_patch(self, patch: np.ndarray) -> bool:
        """Enhanced brain tissue validation (from ae_brats.py)"""
        # Check 1: Minimum ratio of non-background voxels
        brain_tissue_mask = patch > self.config.max_background_intensity
        brain_tissue_ratio = np.sum(brain_tissue_mask) / patch.size
        
        if brain_tissue_ratio < self.config.min_brain_tissue_ratio:
            return False
        
        # Check 2: Mean intensity should be in brain tissue range
        brain_tissue_values = patch[brain_tissue_mask]
        if len(brain_tissue_values) == 0:
            return False
            
        mean_brain_intensity = np.mean(brain_tissue_values)
        if mean_brain_intensity < self.config.min_brain_mean_intensity:
            return False
        
        # Check 3: Not too many very bright pixels (avoid skull, CSF)
        high_intensity_mask = patch > self.config.high_intensity_threshold
        high_intensity_ratio = np.sum(high_intensity_mask) / patch.size
        
        if high_intensity_ratio > self.config.max_high_intensity_ratio:
            return False
        
        # Check 4: Patch should have reasonable contrast
        if patch.std() < self.config.min_patch_std * 2:
            return False
        
        # Check 5: Intensity distribution should be reasonable for brain tissue
        reasonable_intensity_mask = (patch > 0.05) & (patch < 0.95)
        reasonable_ratio = np.sum(reasonable_intensity_mask) / patch.size
        
        if reasonable_ratio < 0.5:
            return False
        
        return True
    
    def get_anomaly_ratio_in_patch(self, segmentation_patch: np.ndarray) -> float:
        """Get ratio of anomalous voxels in patch"""
        anomaly_mask = np.isin(segmentation_patch, self.config.anomaly_labels)
        return np.sum(anomaly_mask) / segmentation_patch.size
    
    def extract_patches_from_subject(self, volume: np.ndarray, segmentation: np.ndarray, 
                                   subject_name: str) -> Tuple[List[np.ndarray], List[int]]:
        """CORRECTED: Extract 3D patches FIRST, with all brain tissue validations"""
        
        # Normalize volume
        volume_norm = self.normalize_volume(volume)
        
        h, w, d = volume_norm.shape
        patch_size = self.config.patch_size
        edge_margin = self.config.edge_margin
        
        patches = []
        labels = []
        
        max_attempts = 1000
        attempts = 0
        
        normal_count = 0
        anomaly_count = 0
        
        print(f"Extracting patches from {subject_name}...")
        
        while (normal_count < self.config.max_normal_patches_per_subject or 
               anomaly_count < self.config.max_anomaly_patches_per_subject) and attempts < max_attempts:
            
            attempts += 1
            
            # Random patch location with margin
            x = np.random.randint(edge_margin, h - patch_size - edge_margin)
            y = np.random.randint(edge_margin, w - patch_size - edge_margin)
            z = np.random.randint(edge_margin, d - patch_size - edge_margin)
            
            # Extract patch
            patch = volume_norm[x:x+patch_size, y:y+patch_size, z:z+patch_size]
            seg_patch = segmentation[x:x+patch_size, y:y+patch_size, z:z+patch_size]
            
            # Brain tissue quality validation
            if not self.is_brain_tissue_patch(patch):
                continue
            
            # Additional quality checks
            if np.mean(patch) < self.config.min_patch_mean:
                continue
                
            if np.std(patch) < self.config.min_patch_std:
                continue
            
            non_zero_ratio = np.sum(patch > 0) / patch.size
            if non_zero_ratio < self.config.min_non_zero_ratio:
                continue
            
            # Determine label based on tumor content
            tumor_ratio = self.get_anomaly_ratio_in_patch(seg_patch)
            
            if tumor_ratio <= self.config.max_tumor_ratio_normal and normal_count < self.config.max_normal_patches_per_subject:
                # Normal patch
                patches.append(patch.copy())
                labels.append(0)
                normal_count += 1
            elif tumor_ratio >= self.config.min_tumor_ratio_anomaly and anomaly_count < self.config.max_anomaly_patches_per_subject:
                # Anomaly patch
                patches.append(patch.copy())
                labels.append(1)
                anomaly_count += 1
        
        print(f"Subject {subject_name}: {normal_count} normal, {anomaly_count} anomaly patches after {attempts} attempts")
        
        return patches, labels
    
    def extract_anatomix_features_from_patches(self, anatomix_model, patches: List[np.ndarray]) -> np.ndarray:
        """CORRECTED: Extract anatomix features from individual 3D patches"""
        
        if len(patches) == 0:
            return np.array([])
        
        print(f"Extracting anatomix features from {len(patches)} patches...")
        
        batch_size = 8
        all_features = []
        
        for i in tqdm(range(0, len(patches), batch_size), desc="Anatomix feature extraction"):
            batch_patches = patches[i:i+batch_size]
            
            # Convert to tensor: (B, 1, H, W, D)
            batch_tensor = torch.stack([
                torch.from_numpy(patch).float().unsqueeze(0) 
                for patch in batch_patches
            ]).to(self.config.device)
            
            with torch.no_grad():
                # Extract features
                features = anatomix_model(batch_tensor)  # Shape: (B, 16, H/16, W/16, D/16)
                
                # Global average pooling for fixed-size feature vector
                features_pooled = torch.mean(features, dim=[2, 3, 4])  # Shape: (B, 16)
                
                all_features.append(features_pooled.cpu().numpy())
        
        features_array = np.vstack(all_features)
        print(f"✓ Extracted anatomix features shape: {features_array.shape}")
        
        return features_array
    
    def process_dataset(self, anatomix_model, num_subjects: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """CORRECTED: Process dataset with patch-first workflow"""
        
        subject_folders = sorted(glob.glob(os.path.join(self.config.dataset_path, "BraTS-GLI-*")))
        
        if num_subjects is not None and num_subjects > 0:
            subject_folders = subject_folders[:num_subjects]
        
        print(f"=== PATCH-FIRST PROCESSING: {len(subject_folders)} subjects ===")
        print("CORRECTED WORKFLOW: 1) Extract patches → 2) Apply anatomix → 3) Train KNN")
        
        all_patches = []
        all_labels = []
        all_subject_names = []
        
        for subject_folder in tqdm(subject_folders, desc="Processing subjects"):
            subject_name = os.path.basename(subject_folder)
            
            try:
                # Load data
                volume, segmentation = self.load_subject_data(subject_folder)
                
                # STEP 1: Extract patches FIRST
                patches, labels = self.extract_patches_from_subject(volume, segmentation, subject_name)
                
                if len(patches) == 0:
                    print(f"⚠ No valid patches for {subject_name}")
                    continue
                
                # Store patches and labels
                all_patches.extend(patches)
                all_labels.extend(labels)
                all_subject_names.extend([subject_name] * len(labels))
                
            except Exception as e:
                print(f"❌ Error processing {subject_name}: {e}")
                continue
        
        if len(all_patches) == 0:
            return np.array([]), np.array([]), []
        
        # STEP 2: Apply anatomix to ALL collected patches
        print(f"\n=== APPLYING ANATOMIX TO {len(all_patches)} PATCHES ===")
        features = self.extract_anatomix_features_from_patches(anatomix_model, all_patches)
        
        labels_array = np.array(all_labels)
        
        print(f"\n✓ PATCH-FIRST PROCESSING COMPLETE:")
        print(f"  Total patches: {len(features)}")
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"  Normal patches: {np.sum(labels_array == 0)}")
        print(f"  Anomaly patches: {np.sum(labels_array == 1)}")
        print(f"  Subjects: {len(set(all_subject_names))}")
        print(f"  Class balance: {np.sum(labels_array == 1) / len(labels_array) * 100:.1f}% anomalous")
        
        return features, labels_array, all_subject_names

def subject_level_split(features: np.ndarray, labels: np.ndarray, 
                       subject_names: List[str], test_size: float = 0.2, 
                       val_size: float = 0.15) -> Tuple:
    """Subject-level split to prevent data leakage"""
    
    # Group by subject
    subject_data = {}
    for i, subject_name in enumerate(subject_names):
        if subject_name not in subject_data:
            subject_data[subject_name] = {'indices': [], 'has_anomaly': False}
        subject_data[subject_name]['indices'].append(i)
        if labels[i] == 1:
            subject_data[subject_name]['has_anomaly'] = True
    
    subjects = list(subject_data.keys())
    np.random.shuffle(subjects)
    
    n_subjects = len(subjects)
    n_test = max(1, int(n_subjects * test_size))
    n_val = max(1, int(n_subjects * val_size))
    
    test_subjects = subjects[:n_test]
    val_subjects = subjects[n_test:n_test + n_val]
    train_subjects = subjects[n_test + n_val:]
    
    def get_split_data(subject_list):
        indices = []
        for subject in subject_list:
            indices.extend(subject_data[subject]['indices'])
        return (features[indices], labels[indices], 
                [subject_names[i] for i in indices])
    
    train_data = get_split_data(train_subjects)
    val_data = get_split_data(val_subjects)
    test_data = get_split_data(test_subjects)
    
    print(f"✓ Subject-level split:")
    print(f"  Train: {len(train_subjects)} subjects, {len(train_data[0])} samples")
    print(f"  Val: {len(val_subjects)} subjects, {len(val_data[0])} samples")
    print(f"  Test: {len(test_subjects)} subjects, {len(test_data[0])} samples")
    
    return train_data, val_data, test_data

def build_faiss_index(features: np.ndarray) -> faiss.Index:
    """Build FAISS index for normal data only"""
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)
    
    features_f32 = features.astype(np.float32)
    index.add(features_f32)
    
    print(f"✓ FAISS index built: {index.ntotal} vectors, {d} dimensions")
    return index

def unsupervised_anomaly_detection(index: faiss.Index, test_features: np.ndarray, 
                                 k: int = 5) -> Tuple[np.ndarray, float]:
    """Unsupervised anomaly detection using k-NN distances"""
    
    test_features_f32 = test_features.astype(np.float32)
    distances, _ = index.search(test_features_f32, k)
    
    # Anomaly score = mean distance to k nearest neighbors
    anomaly_scores = np.mean(distances, axis=1)
    
    # Statistical threshold: 85th percentile
    threshold = np.percentile(anomaly_scores, 85)
    
    return anomaly_scores, threshold

def evaluate_performance(true_labels: np.ndarray, anomaly_scores: np.ndarray, 
                        threshold: float) -> Dict:
    """Comprehensive evaluation"""
    
    predictions = (anomaly_scores >= threshold).astype(int)
    
    metrics = {
        'ROC_AUC': roc_auc_score(true_labels, anomaly_scores),
        'Average_Precision': average_precision_score(true_labels, anomaly_scores),
        'Accuracy': accuracy_score(true_labels, predictions),
        'Precision': precision_score(true_labels, predictions, zero_division=0),
        'Recall': recall_score(true_labels, predictions, zero_division=0),
        'F1_Score': f1_score(true_labels, predictions, zero_division=0),
        'Threshold': threshold,
        'True_Positives': np.sum((predictions == 1) & (true_labels == 1)),
        'True_Negatives': np.sum((predictions == 0) & (true_labels == 0)),
        'False_Positives': np.sum((predictions == 1) & (true_labels == 0)),
        'False_Negatives': np.sum((predictions == 0) & (true_labels == 1))
    }
    
    return metrics

def main():
    """Main function with corrected patch-first workflow"""
    
    parser = argparse.ArgumentParser(description="CORRECTED Anatomix + KNN (Patch-First)")
    parser.add_argument("--dataset_path", type=str, default="datasets/BraTS2025-GLI-PRE-Challenge-TrainingData")
    parser.add_argument("--num_subjects", type=int, default=10)
    parser.add_argument("--k_neighbors", type=int, default=5)
    
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    config.dataset_path = args.dataset_path
    config.k_neighbors = args.k_neighbors
    
    start_time = time.time()
    
    print("=== CORRECTED ANATOMIX + KNN ANOMALY DETECTION ===")
    print("CRITICAL FIX: Patch-first workflow implemented!")
    print("1) Extract 3D patches → 2) Apply anatomix → 3) Unsupervised KNN")
    
    # Step 1: Load anatomix model
    print("\n1. Loading anatomix model...")
    anatomix_model = load_anatomix_model()
    
    # Step 2: Process dataset with patch-first workflow
    print("\n2. Processing dataset with PATCH-FIRST workflow...")
    processor = PatchFirstBraTSProcessor(config)
    features, labels, subject_names = processor.process_dataset(
        anatomix_model, num_subjects=args.num_subjects if args.num_subjects > 0 else None
    )
    
    if len(features) == 0:
        print("❌ ERROR: No features extracted!")
        return
    
    # Step 3: Subject-level split
    print("\n3. Subject-level data splitting...")
    train_data, val_data, test_data = subject_level_split(features, labels, subject_names)
    
    X_train_all, y_train_all, _ = train_data
    X_val_all, y_val_all, _ = val_data
    X_test, y_test, _ = test_data
    
    # Step 4: Unsupervised setup (normal data only)
    print("\n4. Unsupervised anomaly detection setup...")
    train_normal_mask = (y_train_all == 0)
    X_train_normal = X_train_all[train_normal_mask]
    
    val_normal_mask = (y_val_all == 0)
    X_val_normal = X_val_all[val_normal_mask]
    
    print(f"Training on {len(X_train_normal)} normal patches only")
    
    # Step 5: Build KNN index
    print("\n5. Building KNN index...")
    index = build_faiss_index(X_train_normal)
    
    # Step 6: Determine threshold using validation
    print("\n6. Determining threshold using validation set...")
    val_scores, _ = unsupervised_anomaly_detection(index, X_val_normal, k=config.k_neighbors)
    threshold = np.percentile(val_scores, 85)
    print(f"Threshold: {threshold:.4f}")
    
    # Step 7: Test evaluation
    print("\n7. Evaluating on test set...")
    test_scores, _ = unsupervised_anomaly_detection(index, X_test, k=config.k_neighbors)
    results = evaluate_performance(y_test, test_scores, threshold)
    
    # Step 8: Results
    print("\n=== CORRECTED RESULTS (PATCH-FIRST) ===")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Save summary
    runtime = time.time() - start_time
    
    summary_path = os.path.join(config.output_dir, "corrected_summary_report.txt")
    with open(summary_path, 'w') as f:
        f.write("CORRECTED Anatomix + KNN PATCH-FIRST Anomaly Detection - Summary Report\n")
        f.write("=" * 80 + "\n\n")
        f.write("🔧 CRITICAL CORRECTION APPLIED:\n")
        f.write("✓ Patch-first workflow (extract patches → apply anatomix)\n")
        f.write("✓ Subject-level split (prevents patient data leakage)\n")
        f.write("✓ Unsupervised learning (only normal patches for training)\n")
        f.write("✓ All brain tissue validation tests preserved\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"Patch size: {config.patch_size}x{config.patch_size}x{config.patch_size}\n")
        f.write(f"K neighbors: {config.k_neighbors}\n")
        f.write(f"Device: {config.device}\n\n")
        
        f.write("PERFORMANCE METRICS (CORRECTED):\n")
        f.write("-" * 40 + "\n")
        for metric, value in results.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        
        f.write(f"\nTotal Runtime: {runtime:.2f} seconds\n")
    
    print(f"\n✓ CORRECTED Analysis complete! Runtime: {runtime:.2f} seconds")
    print(f"Results saved to: {config.output_dir}")
    print("\nCORRECTION SUMMARY:")
    print("✅ PATCH-FIRST workflow implemented")
    print("✅ Anatomix correctly applied to individual patches")
    print("✅ All brain tissue validations preserved")
    print("✅ Subject-level split prevents data leakage")
    print("✅ True unsupervised anomaly detection")

if __name__ == "__main__":
    main() 