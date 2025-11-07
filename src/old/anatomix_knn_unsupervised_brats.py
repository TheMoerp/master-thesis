#!/usr/bin/env python3
"""
Anatomix + KNN Unsupervised Anomaly Detection for BraTS Dataset

This program implements unsupervised anomaly detection using:
- Anatomix pre-trained features for high-quality representation learning
- KNN-based anomaly detection with FAISS for efficient similarity search
- True unsupervised approach: only normal patches for training
- Subject-level data splitting to prevent data leakage
- Quality-assured patch extraction from ae_brats.py

Features:
- Anatomix pre-trained 3D U-Net for feature extraction
- Quality-controlled 3D patch extraction from brain tissue
- Subject-level train/val/test splitting (no patient overlap)
- KNN anomaly detection with statistical threshold determination
- Comprehensive evaluation metrics and visualizations
"""

import os
import sys
import glob
import argparse
import random
import warnings
import time
import pickle
import contextlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import plotly.figure_factory as ff

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# ADD: Feature normalization and robust statistics
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats

import faiss

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Ensure anatomix is installed
def install_anatomix():
    """Install anatomix if not already available"""
    if not os.path.exists("anatomix"):
        print("Cloning anatomix repository...")
        os.system("git clone https://github.com/neel-dey/anatomix.git")
        os.chdir("anatomix")
        os.system("pip install -e .")
        os.chdir("..")
    else:
        print("Anatomix already installed")

# Try to import anatomix, install if needed
try:
    # Temporarily suppress stdout during import
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            from anatomix.anatomix.model.network import Unet
except ImportError:
    install_anatomix()
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            from anatomix.anatomix.model.network import Unet

class Config:
    """Central configuration class for all parameters"""
    
    def __init__(self):
        # Dataset paths
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "anatomix_knn_unsupervised_results"
        
        # Patch extraction parameters (from ae_brats.py)
        self.patch_size = 32  # 32x32x32 patches
        self.patches_per_volume = 50  # Number of patches to extract per volume
        self.min_non_zero_ratio = 0.2  # Minimum ratio of non-zero voxels in patch
        self.max_normal_to_anomaly_ratio = 10  # FIXED: Much higher ratio for anomaly detection (was 3)
        self.min_tumor_ratio_in_patch = 0.05  # Minimum tumor ratio for anomaly patches
        
        # Additional patch quality parameters (from ae_brats.py)
        self.min_patch_std = 0.01  # Minimum standard deviation for patch quality
        self.min_patch_mean = 0.05  # Minimum mean intensity for patch quality
        self.max_tumor_ratio_normal = 0.01  # Maximum allowed tumor ratio in normal patches
        self.min_tumor_ratio_anomaly = 0.05  # Minimum required tumor ratio in anomaly patches
        self.max_normal_patches_per_subject = 100  # Maximum normal patches per subject
        self.max_anomaly_patches_per_subject = 50  # Maximum anomaly patches per subject
        
        # Segmentation labels for anomaly detection (from ae_brats.py)
        self.anomaly_labels = [1, 2, 4]  # Default: all tumor labels are anomalies
        
        # Brain tissue quality parameters for normal patches (from ae_brats.py)
        self.min_brain_tissue_ratio = 0.3  # Minimum 30% of patch should be brain tissue
        self.max_background_intensity = 0.1  # Values below this are considered background
        self.min_brain_mean_intensity = 0.1  # Minimum mean intensity for brain tissue patches
        self.max_high_intensity_ratio = 0.7  # Maximum ratio of very bright pixels
        self.high_intensity_threshold = 0.9  # Threshold for "very bright" pixels
        self.edge_margin = 8  # Minimum distance from volume edges to extract patches
        
        # KNN parameters
        self.k_neighbors = 7
        self.train_test_split = 0.8
        self.validation_split = 0.2
        
        # Anatomix parameters
        self.anatomix_batch_size = 4
        self.anatomix_feature_dim = 16
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if torch.cuda.is_available() else 0
        
        # Data Science configuration
        self.verbose = True  # Enable detailed logging by default
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

def load_anatomix_model():
    """Load pre-trained anatomix model"""
    print("Initializing anatomix U-Net model...")
    model = Unet(
        dimension=3,
        input_nc=1,
        output_nc=16,
        num_downs=4,
        ngf=16,
    )
    
    # Check if weights are already downloaded
    weights_path = "anatomix/model-weights/anatomix.pth"
    if not os.path.exists(weights_path):
        if not os.path.exists("anatomix/model-weights"):
            os.makedirs("anatomix/model-weights", exist_ok=True)
        
        print("Downloading anatomix model weights...")
        os.system(f"wget -O {weights_path} https://github.com/neel-dey/anatomix/raw/main/model-weights/anatomix.pth")
    
    print("Loading anatomix model weights...")
    model.load_state_dict(
        torch.load(weights_path, map_location='cpu'),
        strict=True,
    )
    
    # Move to GPU if available
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    
    print(f"✓ Anatomix model loaded successfully on {model.device if hasattr(model, 'device') else 'CPU'}")
    return model

class BraTSPatchDataset(Dataset):
    """Dataset class for 3D patches from BraTS data"""
    
    def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None):
        self.patches = patches
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # Convert to tensor and add channel dimension
        patch = torch.FloatTensor(patch).unsqueeze(0)  # Add channel dimension
        label = torch.FloatTensor([label])
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, label

class BraTSDataProcessor:
    """Class for processing BraTS data and extracting patches (from ae_brats.py)"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_volume(self, subject_path: str, modality: str = 't1c') -> Tuple[np.ndarray, np.ndarray]:
        """Load a specific modality volume and its segmentation mask"""
        volume_file = glob.glob(os.path.join(subject_path, f"*-{modality}.nii.gz"))[0]
        seg_file = glob.glob(os.path.join(subject_path, "*-seg.nii.gz"))[0]
        
        volume = nib.load(volume_file).get_fdata()
        segmentation = nib.load(seg_file).get_fdata()
        
        return volume, segmentation
    
    def is_brain_tissue_patch(self, patch: np.ndarray) -> bool:
        """Check if a patch contains primarily brain tissue (from ae_brats.py)"""
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
        
        # Check 4: Patch should have reasonable contrast (brain has structure)
        if patch.std() < self.config.min_patch_std * 2:  # Stricter std for brain tissue
            return False
        
        # Check 5: Intensity distribution should be reasonable for brain tissue
        reasonable_intensity_mask = (patch > 0.05) & (patch < 0.95)
        reasonable_ratio = np.sum(reasonable_intensity_mask) / patch.size
        
        if reasonable_ratio < 0.5:  # At least 50% should be in reasonable range
            return False
        
        return True

    def is_anomaly_segmentation(self, segmentation_patch: np.ndarray) -> bool:
        """Check if a segmentation patch contains any of the specified anomaly labels"""
        for label in self.config.anomaly_labels:
            if np.any(segmentation_patch == label):
                return True
        return False
    
    def get_anomaly_ratio_in_patch(self, segmentation_patch: np.ndarray) -> float:
        """Calculate the ratio of voxels with anomaly labels in the patch"""
        anomaly_mask = np.isin(segmentation_patch, self.config.anomaly_labels)
        return np.sum(anomaly_mask) / segmentation_patch.size

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range while preserving tissue contrast (from ae_brats.py)"""
        volume = volume.astype(np.float32)
        
        # Only normalize non-zero regions to preserve background
        non_zero_mask = volume > 0
        
        if np.sum(non_zero_mask) == 0:
            return volume
        
        # Use more conservative percentile clipping to preserve tumor contrast
        non_zero_values = volume[non_zero_mask]
        percentile_1 = np.percentile(non_zero_values, 1)   # Lower bound
        percentile_99 = np.percentile(non_zero_values, 99)  # Upper bound
        
        # Clip extreme values but preserve more of the distribution
        volume = np.clip(volume, percentile_1, percentile_99)
        
        # Normalize only non-zero regions
        volume[non_zero_mask] = (volume[non_zero_mask] - percentile_1) / (percentile_99 - percentile_1)
        
        # Ensure values are in [0, 1] range
        volume = np.clip(volume, 0, 1)
            
        return volume 
    
    def extract_normal_patches(self, volume: np.ndarray, segmentation: np.ndarray) -> List[np.ndarray]:
        """Extract normal patches from regions without tumor (from ae_brats.py)"""
        patches = []
        
        # Create a brain tissue mask (non-zero regions with reasonable intensity)
        brain_mask = (volume > self.config.max_background_intensity) & (volume < 0.95)
        
        # Get coordinates where there are no specified anomaly labels AND brain tissue is present
        anomaly_mask = np.isin(segmentation, self.config.anomaly_labels)
        normal_tissue_coords = np.where(~anomaly_mask)  # Invert to get normal tissue
        brain_coords = np.where(brain_mask)
        
        # Find intersection of normal tissue and brain tissue coordinates
        normal_tissue_set = set(zip(normal_tissue_coords[0], normal_tissue_coords[1], normal_tissue_coords[2]))
        brain_set = set(zip(brain_coords[0], brain_coords[1], brain_coords[2]))
        valid_coords_set = normal_tissue_set.intersection(brain_set)
        
        if len(valid_coords_set) == 0:
            return patches
        
        # Convert back to coordinate arrays
        valid_coords_list = list(valid_coords_set)
        
        # Filter coordinates to avoid edges (prevent edge artifacts)
        edge_margin = self.config.edge_margin
        filtered_coords = []
        for x, y, z in valid_coords_list:
            if (x >= edge_margin and x < volume.shape[0] - edge_margin and
                y >= edge_margin and y < volume.shape[1] - edge_margin and
                z >= edge_margin and z < volume.shape[2] - edge_margin):
                # Also check if we can extract a full patch at this location
                x_start = x - self.config.patch_size // 2
                x_end = x_start + self.config.patch_size
                y_start = y - self.config.patch_size // 2
                y_end = y_start + self.config.patch_size
                z_start = z - self.config.patch_size // 2
                z_end = z_start + self.config.patch_size
                
                if (x_start >= 0 and x_end <= volume.shape[0] and
                    y_start >= 0 and y_end <= volume.shape[1] and
                    z_start >= 0 and z_end <= volume.shape[2]):
                    filtered_coords.append((x, y, z))
        
        if len(filtered_coords) == 0:
            if self.config.verbose:
                print("Warning: No valid coordinates found for normal patch extraction")
            return patches
        
        # Calculate number of patches to extract
        max_patches = min(len(filtered_coords) // 20, self.config.max_normal_patches_per_subject)
        max_patches = max(max_patches, 10)  # At least try for 10 patches
        
        # Sample coordinates
        num_to_sample = min(max_patches * 5, len(filtered_coords))  # Sample more to filter later
        indices = np.random.choice(len(filtered_coords), size=num_to_sample, replace=False)
        
        patch_coords = [filtered_coords[i] for i in indices]
        
        patches_extracted = 0
        patches_rejected = 0
        
        # Extract patches with enhanced quality control
        for x, y, z in tqdm(patch_coords, desc="Extracting normal patches", leave=False):
            # Calculate patch boundaries (we already checked they're valid)
            x_start = x - self.config.patch_size // 2
            x_end = x_start + self.config.patch_size
            y_start = y - self.config.patch_size // 2
            y_end = y_start + self.config.patch_size
            z_start = z - self.config.patch_size // 2
            z_end = z_start + self.config.patch_size
            
            patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
            patch_seg = segmentation[x_start:x_end, y_start:y_end, z_start:z_end]
            
            # Enhanced quality checks for brain tissue
            anomaly_ratio = self.get_anomaly_ratio_in_patch(patch_seg)
            
            # Check 1: No anomaly labels (according to specified anomaly_labels)
            if anomaly_ratio > self.config.max_tumor_ratio_normal:
                patches_rejected += 1
                continue
            
            # Check 2: Brain tissue quality check
            if not self.is_brain_tissue_patch(patch):
                patches_rejected += 1
                continue
            
            # Check 3: Additional contrast and structure checks
            if patch.std() < self.config.min_patch_std:
                patches_rejected += 1
                continue
            
            # If all checks pass, add the patch
            patches.append(patch)
            patches_extracted += 1
            
            # Stop if we have enough patches
            if patches_extracted >= max_patches:
                break
        
        if self.config.verbose:
            print(f"Normal patch extraction: {patches_extracted} accepted, {patches_rejected} rejected")
        return patches
    
    def extract_anomalous_patches(self, volume: np.ndarray, segmentation: np.ndarray) -> List[np.ndarray]:
        """Extract anomalous patches from regions with specified anomaly labels (from ae_brats.py)"""
        patches = []
        
        # Get coordinates where there are specified anomaly labels
        anomaly_mask = np.isin(segmentation, self.config.anomaly_labels)
        anomaly_coords = np.where(anomaly_mask)
        
        if len(anomaly_coords[0]) == 0:
            return patches
        
        # Calculate number of patches to extract
        max_patches = min(len(anomaly_coords[0]) // 50, self.config.max_anomaly_patches_per_subject)
        
        if max_patches == 0:
            return patches
        
        # Sample coordinates
        indices = np.random.choice(len(anomaly_coords[0]), 
                                 size=min(max_patches, len(anomaly_coords[0])), 
                                 replace=False)
        
        patch_coords = [(anomaly_coords[0][i], anomaly_coords[1][i], anomaly_coords[2][i]) 
                       for i in indices]
        
        # Extract patches with progress bar
        for x, y, z in tqdm(patch_coords, desc="Extracting anomaly patches", leave=False):
            # Calculate patch boundaries
            x_start = max(0, x - self.config.patch_size // 2)
            x_end = min(volume.shape[0], x_start + self.config.patch_size)
            y_start = max(0, y - self.config.patch_size // 2)
            y_end = min(volume.shape[1], y_start + self.config.patch_size)
            z_start = max(0, z - self.config.patch_size // 2)
            z_end = min(volume.shape[2], z_start + self.config.patch_size)
            
            # Adjust start coordinates if patch would be too small
            if x_end - x_start < self.config.patch_size:
                x_start = max(0, x_end - self.config.patch_size)
            if y_end - y_start < self.config.patch_size:
                y_start = max(0, y_end - self.config.patch_size)
            if z_end - z_start < self.config.patch_size:
                z_start = max(0, z_end - self.config.patch_size)
            
            # Skip if we can't get a full-sized patch
            if (x_end - x_start != self.config.patch_size or 
                y_end - y_start != self.config.patch_size or 
                z_end - z_start != self.config.patch_size):
                continue
            
            patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
            
            # Quality checks
            if patch.std() > self.config.min_patch_std and patch.mean() > self.config.min_patch_mean:
                # Verify this patch contains specified anomaly labels
                patch_seg = segmentation[x_start:x_end, y_start:y_end, z_start:z_end]
                anomaly_ratio = self.get_anomaly_ratio_in_patch(patch_seg)
                
                if anomaly_ratio >= self.config.min_tumor_ratio_anomaly:
                    patches.append(patch)
        
        return patches

class AnatomixFeatureExtractor:
    """Class for extracting Anatomix features from patches"""
    
    def __init__(self, config: Config, anatomix_model):
        self.config = config
        self.anatomix_model = anatomix_model
        
    def validate_anatomix_compatibility(self, target_size: tuple, num_downs: int = 4) -> tuple:
        """Validate size compatibility with anatomix"""
        divisor = 2 ** num_downs
        
        validated_size = []
        for dim in target_size:
            if dim % divisor != 0:
                new_dim = ((dim + divisor // 2) // divisor) * divisor
                validated_size.append(new_dim)
            else:
                validated_size.append(dim)
        
        return tuple(validated_size)

    def standardize_patch_size(self, patch: np.ndarray, target_size: tuple = (32, 32, 32)) -> np.ndarray:
        """Standardize patch size for anatomix compatibility"""
        target_size = self.validate_anatomix_compatibility(target_size)
        
        current_size = patch.shape
        standardized_patch = np.zeros(target_size, dtype=patch.dtype)
        
        ranges = []
        for i in range(3):
            current_dim = current_size[i]
            target_dim = target_size[i]
            
            if current_dim >= target_dim:
                start = (current_dim - target_dim) // 2
                end = start + target_dim
                ranges.append((start, end, 0, target_dim))
            else:
                start_pad = (target_dim - current_dim) // 2
                end_pad = start_pad + current_dim
                ranges.append((0, current_dim, start_pad, end_pad))
        
        standardized_patch[
            ranges[0][2]:ranges[0][3],
            ranges[1][2]:ranges[1][3], 
            ranges[2][2]:ranges[2][3]
        ] = patch[
            ranges[0][0]:ranges[0][1],
            ranges[1][0]:ranges[1][1],
            ranges[2][0]:ranges[2][1]
        ]
        
        return standardized_patch
        
    def extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract anatomix features from a single patch"""
        try:
            # Standardize patch size
            standardized_patch = self.standardize_patch_size(patch)
            
            # Convert to tensor and add batch and channel dimensions
            patch_tensor = torch.tensor(standardized_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            patch_tensor = patch_tensor.to(self.config.device)
            
            with torch.no_grad():
                features = self.anatomix_model(patch_tensor)
                features = features.squeeze(0).cpu().numpy()  # Remove batch dimension
                features = np.transpose(features, (1, 2, 3, 0))  # (H, W, D, C)
                
                # Global pooling to get a single feature vector per patch
                # Use multiple pooling strategies for richer representation
                mean_features = np.mean(features, axis=(0, 1, 2))  # Global average pooling
                max_features = np.max(features, axis=(0, 1, 2))   # Global max pooling
                std_features = np.std(features, axis=(0, 1, 2))   # Global std pooling
                
                # Combine different pooling strategies
                combined_features = np.concatenate([mean_features, max_features, std_features])
            
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features from patch: {e}")
            # Return zero features as fallback
            return np.zeros(self.config.anatomix_feature_dim * 3)  # 3 pooling strategies
    
    def extract_features_from_patches(self, patches: List[np.ndarray]) -> np.ndarray:
        """Extract anatomix features from a list of patches"""
        features_list = []
        
        for patch in tqdm(patches, desc="Extracting Anatomix features"):
            features = self.extract_patch_features(patch)
            features_list.append(features)
        
        return np.array(features_list, dtype=np.float32) 

class AnatomixKNNProcessor:
    """Main processor combining patch extraction with Anatomix feature extraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = BraTSDataProcessor(config)
        
    def process_dataset(self, anatomix_model, num_subjects: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Process the BraTS dataset and extract patches with subject tracking (from ae_brats.py)"""
        
        # Get list of subjects
        subject_dirs = [d for d in os.listdir(self.config.dataset_path) 
                       if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        
        if num_subjects:
            subject_dirs = subject_dirs[:num_subjects]
        
        if self.config.verbose:
            print(f"Processing {len(subject_dirs)} subjects...")
        
        # Initialize feature extractor
        feature_extractor = AnatomixFeatureExtractor(self.config, anatomix_model)
        
        all_normal_patches = []
        all_anomalous_patches = []
        all_normal_subjects = []  # Track which subject each patch comes from
        all_anomalous_subjects = []
        
        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            subject_path = os.path.join(self.config.dataset_path, subject_dir)
            
            try:
                # Load volume and segmentation
                volume, segmentation = self.data_processor.load_volume(subject_path)
                
                if volume is None or segmentation is None:
                    continue
                
                # Normalize volume
                volume = self.data_processor.normalize_volume(volume)
                
                # Extract patches
                normal_patches = self.data_processor.extract_normal_patches(volume, segmentation)
                anomalous_patches = self.data_processor.extract_anomalous_patches(volume, segmentation)
                
                # Extract Anatomix features from patches
                if len(normal_patches) > 0:
                    normal_features = feature_extractor.extract_features_from_patches(normal_patches)
                    all_normal_patches.extend(normal_features)
                    all_normal_subjects.extend([subject_dir] * len(normal_features))
                
                if len(anomalous_patches) > 0:
                    anomalous_features = feature_extractor.extract_features_from_patches(anomalous_patches)
                    all_anomalous_patches.extend(anomalous_features)
                    all_anomalous_subjects.extend([subject_dir] * len(anomalous_features))
                
            except Exception as e:
                print(f"Error processing {subject_dir}: {e}")
                continue
        
        if self.config.verbose:
            print(f"Extracted {len(all_normal_patches)} normal patches and {len(all_anomalous_patches)} anomalous patches")
        
        # Balance dataset
        num_anomalous = len(all_anomalous_patches)
        max_normal = int(num_anomalous * self.config.max_normal_to_anomaly_ratio)
        
        if len(all_normal_patches) > max_normal:
            indices = np.random.choice(len(all_normal_patches), max_normal, replace=False)
            all_normal_patches = [all_normal_patches[i] for i in indices]
            all_normal_subjects = [all_normal_subjects[i] for i in indices]
        
        if self.config.verbose:
            print(f"Final dataset: {len(all_normal_patches)} normal, {len(all_anomalous_patches)} anomalous patches")
        
        # Combine patches and create labels
        all_features = all_normal_patches + all_anomalous_patches
        labels = [0] * len(all_normal_patches) + [1] * len(all_anomalous_patches)
        subjects = all_normal_subjects + all_anomalous_subjects
        
        # Convert to numpy arrays
        features_array = np.array(all_features, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)
        
        # ADD: CRITICAL DATA SCIENCE FIX - Feature Normalization
        if self.config.verbose:
            print(f"\n🔧 DATA SCIENCE FIX: Applying feature normalization...")
            print(f"Features before normalization - Mean: {features_array.mean():.4f}, Std: {features_array.std():.4f}")
            print(f"Feature range: [{features_array.min():.4f}, {features_array.max():.4f}]")
        
        # Use RobustScaler to handle outliers better than StandardScaler
        self.feature_scaler = RobustScaler()
        features_array = self.feature_scaler.fit_transform(features_array)
        
        if self.config.verbose:
            print(f"Features after normalization - Mean: {features_array.mean():.4f}, Std: {features_array.std():.4f}")
            print(f"Feature range: [{features_array.min():.4f}, {features_array.max():.4f}]")
            print("✓ Feature normalization complete")
        
        return features_array, labels_array, subjects

# KNN and FAISS functions
def build_faiss_index(features: np.ndarray, device_id: int = 0, use_gpu: bool = True):
    """Build FAISS index for KNN search with CPU fallback"""
    d = features.shape[1]
    n_samples = len(features)
    
    print(f"Building FAISS index: {n_samples} samples, {d} features")
    
    # Ensure float32 and contiguous
    if features.dtype != np.float32:
        features = features.astype(np.float32)
    
    features = np.ascontiguousarray(features)
    
    if use_gpu and torch.cuda.is_available():
        try:
            print("Building GPU FAISS index...")
            res = faiss.StandardGpuResources()
            gpu_config = faiss.GpuIndexFlatConfig()
            gpu_config.device = device_id
            gpu_config.useFloat16 = False  # Use full precision
            
            index = faiss.GpuIndexFlatL2(res, d, gpu_config)
            
            batch_size = min(2000, n_samples)
            
            for i in tqdm(range(0, len(features), batch_size), desc="Building GPU index"):
                batch = features[i:i+batch_size]
                batch = np.ascontiguousarray(batch)
                index.add(batch)
                
            print(f"✓ GPU FAISS index built: {index.ntotal} vectors")
            return index
            
        except Exception as e:
            print(f"GPU FAISS failed: {e}")
            print("Falling back to CPU FAISS...")
    
    # CPU fallback
    print("Building CPU FAISS index...")
    index = faiss.IndexFlatL2(d)
    index.add(features)
    
    print(f"✓ CPU FAISS index built: {index.ntotal} vectors")
    return index

def unsupervised_knn_anomaly_detection(index, test_features: np.ndarray, k: int = 5) -> Tuple[np.ndarray, float]:
    """
    Unsupervised anomaly detection using KNN distances with statistical threshold
    """
    print(f"Performing unsupervised KNN anomaly detection on {len(test_features)} samples")
    
    # Ensure proper format for FAISS
    if test_features.dtype != np.float32:
        test_features = test_features.astype(np.float32)
    
    test_features = np.ascontiguousarray(test_features)
    
    try:
        # Get k-NN distances
        distances, _ = index.search(test_features, k)
        
        # Anomaly score = mean distance to k nearest neighbors
        anomaly_scores = np.mean(distances, axis=1)
        
        # Unsupervised threshold: 95th percentile as outlier threshold
        threshold = np.percentile(anomaly_scores, 95)
        
        print(f"Anomaly scores: min={np.min(anomaly_scores):.4f}, max={np.max(anomaly_scores):.4f}")
        print(f"Statistical threshold (95th percentile): {threshold:.4f}")
        print(f"Predicted outliers: {np.sum(anomaly_scores > threshold)} / {len(anomaly_scores)}")
        
        return anomaly_scores, threshold
        
    except Exception as e:
        print(f"Error in unsupervised anomaly detection: {e}")
        raise e

def find_unsupervised_threshold(val_features: np.ndarray, index, k: int = 5) -> float:
    """Find threshold using ONLY normal validation data (unsupervised) - IMPROVED VERSION"""
    print("🔧 IMPROVED: Determining robust threshold using ONLY normal validation data...")
    
    # Calculate distances on validation set
    if val_features.dtype != np.float32:
        val_features = val_features.astype(np.float32)
    
    val_features = np.ascontiguousarray(val_features)
    
    try:
        distances, _ = index.search(val_features, k)
        val_scores = np.mean(distances, axis=1)
        
        print(f"Validation scores - Mean: {val_scores.mean():.6f}, Std: {val_scores.std():.6f}")
        print(f"Validation scores - Median: {np.median(val_scores):.6f}")
        print(f"Validation scores - Range: [{val_scores.min():.6f}, {val_scores.max():.6f}]")
        
        # IMPROVED: Multiple threshold strategies for robustness
        threshold_methods = {
            'percentile_95': np.percentile(val_scores, 95),
            'percentile_90': np.percentile(val_scores, 90),
            'percentile_85': np.percentile(val_scores, 85),  # Less conservative
            'mean_plus_2std': val_scores.mean() + 2 * val_scores.std(),
            'mean_plus_3std': val_scores.mean() + 3 * val_scores.std(),
            'median_plus_2mad': np.median(val_scores) + 2 * stats.median_abs_deviation(val_scores),
            'iqr_outlier': np.percentile(val_scores, 75) + 1.5 * (np.percentile(val_scores, 75) - np.percentile(val_scores, 25))
        }
        
        print(f"\n📊 THRESHOLD CANDIDATES:")
        print(f"{'Method':<20} {'Threshold':<12} {'Outliers %':<12}")
        print(f"{'-'*45}")
        
        for method, threshold in threshold_methods.items():
            outliers_pct = (np.sum(val_scores > threshold) / len(val_scores)) * 100
            print(f"{method:<20} {threshold:<12.6f} {outliers_pct:<12.1f}%")
        
        # IMPROVED: Adaptive threshold selection based on score distribution
        # Check if data follows normal distribution
        _, p_value = stats.normaltest(val_scores)
        is_normal = p_value > 0.05
        
        if is_normal:
            # If normal distribution, use statistical approach
            selected_threshold = threshold_methods['mean_plus_2std']
            selected_method = 'mean_plus_2std (normal distribution detected)'
        else:
            # If not normal, use robust percentile-based approach
            # Adapt percentile based on score distribution characteristics
            cv = val_scores.std() / val_scores.mean()  # Coefficient of variation
            if cv > 0.5:  # High variability
                selected_threshold = threshold_methods['percentile_85']
                selected_method = 'percentile_85 (high variability detected)'
            else:  # Lower variability
                selected_threshold = threshold_methods['percentile_90']
                selected_method = 'percentile_90 (moderate variability)'
        
        outliers_pct = (np.sum(val_scores > selected_threshold) / len(val_scores)) * 100
        
        print(f"\n🎯 SELECTED THRESHOLD: {selected_threshold:.6f}")
        print(f"Method: {selected_method}")
        print(f"Expected outlier rate: {outliers_pct:.1f}%")
        print(f"Distribution normality p-value: {p_value:.4f}")
        print(f"Coefficient of variation: {cv:.4f}")
        
        return selected_threshold
        
    except Exception as e:
        print(f"Error in improved threshold determination: {e}")
        raise e

class AnomalyDetectionEvaluator:
    """Evaluator for anomaly detection performance (from ae_brats.py)"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def evaluate(self, test_features: np.ndarray, test_labels: np.ndarray, 
                 val_features: np.ndarray, train_index) -> Dict:
        """Evaluate KNN anomaly detection with unsupervised threshold"""
        
        # Step 1: Determine threshold using ONLY normal validation data
        optimal_threshold = find_unsupervised_threshold(val_features, train_index, self.config.k_neighbors)
        
        # Step 2: Calculate anomaly scores on test set
        anomaly_scores, _ = unsupervised_knn_anomaly_detection(train_index, test_features, self.config.k_neighbors)
        
        # Step 3: Apply threshold
        predictions = (anomaly_scores > optimal_threshold).astype(int)
        
        print(f"\nEVALUATION: Anatomix + KNN Unsupervised Anomaly Detection")
        print(f"{'='*60}")
        print(f"Total test samples: {len(test_features)}")
        print(f"Normal samples: {np.sum(test_labels == 0)}")
        print(f"Anomalous samples: {np.sum(test_labels == 1)}")
        
        # Step 4: Calculate metrics
        try:
            roc_auc = roc_auc_score(test_labels, anomaly_scores)
            average_precision = average_precision_score(test_labels, anomaly_scores)
        except:
            roc_auc = 0.0
            average_precision = 0.0
        
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        
        # Additional metrics for imbalanced anomaly detection
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0
        
        # Dice Similarity Coefficient
        dsc = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # False Positive Rate and False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\nANATOMIX + KNN UNSUPERVISED PERFORMANCE:")
        print(f"{'='*60}")
        print(f"ROC AUC:                  {roc_auc:.4f}")
        print(f"Average Precision (AP):   {average_precision:.4f}")
        print(f"Matthews Correlation:     {mcc:.4f}")
        print(f"Dice Similarity Coeff:    {dsc:.4f}")
        print(f"Balanced Accuracy:        {balanced_accuracy:.4f}")
        print(f"F1 Score:                 {f1:.4f}")
        print(f"Precision:                {precision:.4f}")
        print(f"Recall (Sensitivity):     {recall:.4f}")
        print(f"Specificity:              {specificity:.4f}")
        print(f"Accuracy:                 {accuracy:.4f}")
        print(f"False Positive Rate:      {fpr:.4f}")
        print(f"False Negative Rate:      {fnr:.4f}")
        print(f"Threshold Used:           {optimal_threshold:.6f}")
        print(f"{'='*60}")
        
        results = {
            'anomaly_scores': anomaly_scores,
            'true_labels': test_labels,
            'predictions': predictions,
            'optimal_threshold': optimal_threshold,
            'roc_auc': roc_auc,
            'average_precision': average_precision,
            'mcc': mcc,
            'dsc': dsc,
            'balanced_accuracy': balanced_accuracy,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr,
            'fnr': fnr,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        return results 

class PatchFeatureVisualizer:
    """Class for visualizing patch-based Anatomix features as images (oriented by anatomix_knn_brats.py)"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def create_patch_feature_visualizations(self, processor: 'AnatomixKNNProcessor', 
                                          anatomix_model, features: np.ndarray, 
                                          labels: np.ndarray, subjects: List[str]):
        """Create comprehensive patch feature visualizations"""
        print(f"\n🎨 CREATING PATCH FEATURE VISUALIZATIONS...")
        
        # Create visualization directory
        vis_dir = os.path.join(self.config.output_dir, "patch_feature_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Extract and save sample patches for visualization
        sample_patches, sample_labels, sample_subjects = self._extract_sample_patches(
            processor, anatomix_model, num_normal=50, num_anomaly=20)
        
        if len(sample_patches) > 0:
            # 2. Create patch grids
            self._create_patch_grids(sample_patches, sample_labels, sample_subjects, vis_dir)
            
            # 3. Create detailed 3D patch analysis
            self._create_3d_patch_analysis(sample_patches, sample_labels, vis_dir)
            
            # 4. Create feature channel analysis
            self._create_feature_channel_analysis(sample_patches, sample_labels, vis_dir)
            
            # 5. Create compact patch visualization
            self._create_compact_patch_visualization(sample_patches, sample_labels, sample_subjects, vis_dir)
            
            # 6. Create feature comparison visualization
            self._create_feature_comparison_visualization(features, labels, subjects, vis_dir)
        
        print(f"✓ Patch feature visualizations saved to: {vis_dir}")
    
    def _extract_sample_patches(self, processor: 'AnatomixKNNProcessor', anatomix_model, 
                               num_normal: int = 50, num_anomaly: int = 20) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Extract sample patches from the dataset for visualization"""
        print("Extracting sample patches for visualization...")
        
        # Get list of subjects for sampling
        subject_dirs = [d for d in os.listdir(self.config.dataset_path) 
                       if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        
        # Sample a few subjects for visualization
        max_subjects_for_vis = min(3, len(subject_dirs))
        sampled_subjects = np.random.choice(subject_dirs, max_subjects_for_vis, replace=False)
        
        all_patches = []
        all_labels = []
        all_subjects = []
        
        for subject_dir in sampled_subjects:
            subject_path = os.path.join(self.config.dataset_path, subject_dir)
            
            try:
                # Load and process volume
                volume, segmentation = processor.data_processor.load_volume(subject_path)
                if volume is None or segmentation is None:
                    continue
                
                volume = processor.data_processor.normalize_volume(volume)
                
                # Extract normal patches
                normal_patches = processor.data_processor.extract_normal_patches(volume, segmentation)
                for patch in normal_patches[:10]:  # Limit to 10 per subject
                    all_patches.append(patch)
                    all_labels.append(0)
                    all_subjects.append(subject_dir)
                
                # Extract anomalous patches
                anomalous_patches = processor.data_processor.extract_anomalous_patches(volume, segmentation)
                for patch in anomalous_patches[:5]:  # Limit to 5 per subject
                    all_patches.append(patch)
                    all_labels.append(1)
                    all_subjects.append(subject_dir)
                    
            except Exception as e:
                print(f"Error processing {subject_dir} for visualization: {e}")
                continue
        
        # Sample final patches
        normal_indices = [i for i, label in enumerate(all_labels) if label == 0]
        anomaly_indices = [i for i, label in enumerate(all_labels) if label == 1]
        
        selected_indices = []
        if len(normal_indices) > 0:
            selected_normal = np.random.choice(normal_indices, 
                                             min(num_normal, len(normal_indices)), replace=False)
            selected_indices.extend(selected_normal)
        
        if len(anomaly_indices) > 0:
            selected_anomaly = np.random.choice(anomaly_indices, 
                                              min(num_anomaly, len(anomaly_indices)), replace=False)
            selected_indices.extend(selected_anomaly)
        
        sample_patches = [all_patches[i] for i in selected_indices]
        sample_labels = [all_labels[i] for i in selected_indices]
        sample_subjects = [all_subjects[i] for i in selected_indices]
        
        print(f"Extracted {len(sample_patches)} sample patches ({np.sum(sample_labels)} anomalous)")
        return sample_patches, sample_labels, sample_subjects
    
    def _create_patch_grids(self, patches: List[np.ndarray], labels: List[int], 
                           subjects: List[str], vis_dir: str):
        """Create patch grids similar to anatomix_knn_brats.py"""
        print("Creating patch grids...")
        
        # Separate normal and anomalous patches
        normal_patches = [p for p, l in zip(patches, labels) if l == 0]
        anomaly_patches = [p for p, l in zip(patches, labels) if l == 1]
        
        # Create normal patch grid
        if len(normal_patches) > 0:
            self._save_patch_grid(normal_patches, vis_dir, "normal_patches_grid", "Normal Brain Tissue Patches")
        
        # Create anomaly patch grid
        if len(anomaly_patches) > 0:
            self._save_patch_grid(anomaly_patches, vis_dir, "anomaly_patches_grid", "Anomalous Brain Tissue Patches")
    
    def _save_patch_grid(self, patches: List[np.ndarray], output_dir: str, filename: str, title: str):
        """Save a grid of 3D patches showing middle slices (adapted from anatomix_knn_brats.py)"""
        n_patches = len(patches)
        
        if n_patches == 0:
            return
        
        print(f"Saving patch grid: {n_patches} patches")
        
        # Calculate grid size
        cols = min(8, n_patches)
        rows = (n_patches + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'{title} (Middle Slice View)', fontsize=14, fontweight='bold')
        
        for i in range(n_patches):
            patch = patches[i]
            
            # Take middle slice of the 3D patch
            if len(patch.shape) == 3:  # (H, W, D)
                h, w, d = patch.shape
                slice_img = patch[:, :, d//2]  # Middle slice
            else:
                slice_img = patch
            
            # Normalize for visualization
            slice_img_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
            
            # Plot
            im = axes[i].imshow(slice_img_norm, cmap='gray')
            axes[i].set_title(f'Patch {i+1}', fontsize=8)
            axes[i].axis('off')
            
            # Add mini colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(n_patches, len(axes)):
            axes[i].axis('off')
        
        # Save
        save_path = os.path.join(output_dir, f'{filename}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Patch grid saved to: {save_path}")
    
    def _create_3d_patch_analysis(self, patches: List[np.ndarray], labels: List[int], vis_dir: str):
        """Create detailed 3D analysis of patches (adapted from anatomix_knn_brats.py)"""
        print("Creating 3D patch analysis...")
        
        # Separate patches by label
        normal_patches = [p for p, l in zip(patches, labels) if l == 0]
        anomaly_patches = [p for p, l in zip(patches, labels) if l == 1]
        
        # Analyze representative patches
        if len(normal_patches) > 0:
            self._analyze_single_3d_patch(normal_patches[0], vis_dir, "normal_patch_3d_analysis", "Normal")
        
        if len(anomaly_patches) > 0:
            self._analyze_single_3d_patch(anomaly_patches[0], vis_dir, "anomaly_patch_3d_analysis", "Anomaly")
        
        # Create comparative analysis
        if len(normal_patches) > 0 and len(anomaly_patches) > 0:
            self._create_comparative_3d_analysis(normal_patches[0], anomaly_patches[0], vis_dir)
    
    def _analyze_single_3d_patch(self, patch: np.ndarray, output_dir: str, filename: str, label: str):
        """Analyze a single 3D patch showing multiple slices (from anatomix_knn_brats.py)"""
        if len(patch.shape) != 3:
            return
        
        h, w, d = patch.shape
        print(f"Analyzing {label} patch: {patch.shape}")
        
        # Create visualization with multiple slices
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f'{label} Patch 3D Analysis (Shape: {patch.shape})', fontsize=16, fontweight='bold')
        
        # Show 9 slices through the depth
        slice_indices = np.linspace(0, d-1, 9, dtype=int)
        
        for i, slice_idx in enumerate(slice_indices):
            row, col = i // 3, i % 3
            
            slice_data = patch[:, :, slice_idx]
            slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            
            im = axes[row, col].imshow(slice_norm, cmap='gray')
            axes[row, col].set_title(f'Slice {slice_idx}/{d-1}', fontsize=10)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Save analysis
        save_path = os.path.join(output_dir, f'{filename}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"3D analysis saved: {save_path}")
    
    def _create_comparative_3d_analysis(self, normal_patch: np.ndarray, anomaly_patch: np.ndarray, vis_dir: str):
        """Create comparative analysis between normal and anomaly patches"""
        print("Creating comparative 3D analysis...")
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        fig.suptitle('Comparative 3D Patch Analysis: Normal vs Anomaly', fontsize=16, fontweight='bold')
        
        # Show 5 slices for each patch type
        n_slices = 5
        normal_slices = np.linspace(0, normal_patch.shape[2]-1, n_slices, dtype=int)
        anomaly_slices = np.linspace(0, anomaly_patch.shape[2]-1, n_slices, dtype=int)
        
        # Normal patches (top row)
        for i, slice_idx in enumerate(normal_slices):
            slice_data = normal_patch[:, :, slice_idx]
            slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            
            im = axes[0, i].imshow(slice_norm, cmap='gray')
            axes[0, i].set_title(f'Normal - Slice {slice_idx}', fontsize=10)
            axes[0, i].axis('off')
        
        # Anomaly patches (bottom row)
        for i, slice_idx in enumerate(anomaly_slices):
            slice_data = anomaly_patch[:, :, slice_idx]
            slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
            
            im = axes[1, i].imshow(slice_norm, cmap='gray')
            axes[1, i].set_title(f'Anomaly - Slice {slice_idx}', fontsize=10)
            axes[1, i].axis('off')
        
        # Save comparative analysis
        save_path = os.path.join(vis_dir, 'comparative_3d_patch_analysis.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparative 3D analysis saved: {save_path}")
    
    def _create_feature_channel_analysis(self, patches: List[np.ndarray], labels: List[int], vis_dir: str):
        """Create analysis of Anatomix feature channels extracted from patches"""
        print("Creating feature channel analysis...")
        
        # Create simulated feature maps to show what Anatomix extracts
        self._create_simulated_feature_maps(patches, labels, vis_dir)
        
        # Create EXTENDED feature maps - viele bunte Bilder!
        self._create_extended_feature_maps(patches, labels, vis_dir)
        
        # Create multiple patches with all their features
        self._create_multiple_patch_feature_gallery(patches, labels, vis_dir)
    
    def _create_simulated_feature_maps(self, patches: List[np.ndarray], labels: List[int], vis_dir: str):
        """Create simulated feature maps to show what Anatomix might extract"""
        print("Creating simulated feature maps...")
        
        # Take first normal and anomaly patch
        normal_patches = [p for p, l in zip(patches, labels) if l == 0]
        anomaly_patches = [p for p, l in zip(patches, labels) if l == 1]
        
        if len(normal_patches) == 0 or len(anomaly_patches) == 0:
            return
        
        normal_patch = normal_patches[0]
        anomaly_patch = anomaly_patches[0]
        
        # Create simulated feature maps using basic image processing
        feature_maps_normal = self._simulate_anatomix_features(normal_patch)
        feature_maps_anomaly = self._simulate_anatomix_features(anomaly_patch)
        
        # Visualize feature maps
        fig, axes = plt.subplots(2, 8, figsize=(20, 8))
        fig.suptitle('Simulated Anatomix Feature Maps (Representative Examples)', fontsize=16, fontweight='bold')
        
        # Normal feature maps (top row)
        for i in range(8):
            im = axes[0, i].imshow(feature_maps_normal[:, :, i], cmap='viridis')
            axes[0, i].set_title(f'Normal - Feature {i}', fontsize=10)
            axes[0, i].axis('off')
            plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Anomaly feature maps (bottom row)
        for i in range(8):
            im = axes[1, i].imshow(feature_maps_anomaly[:, :, i], cmap='viridis')
            axes[1, i].set_title(f'Anomaly - Feature {i}', fontsize=10)
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # Save feature maps
        save_path = os.path.join(vis_dir, 'simulated_feature_maps.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Simulated feature maps saved: {save_path}")
    
    def _simulate_anatomix_features(self, patch: np.ndarray) -> np.ndarray:
        """Simulate what Anatomix feature maps might look like using basic image processing"""
        from scipy import ndimage
        
        # Take middle slice
        middle_slice = patch[:, :, patch.shape[2]//2]
        h, w = middle_slice.shape
        
        # Create 8 different "feature" maps using various filters
        feature_maps = np.zeros((h, w, 8))
        
        # Feature 0: Original intensity
        feature_maps[:, :, 0] = middle_slice
        
        # Feature 1: Gaussian blur
        feature_maps[:, :, 1] = ndimage.gaussian_filter(middle_slice, sigma=1.0)
        
        # Feature 2: Gradient magnitude
        grad_x = ndimage.sobel(middle_slice, axis=0)
        grad_y = ndimage.sobel(middle_slice, axis=1)
        feature_maps[:, :, 2] = np.sqrt(grad_x**2 + grad_y**2)
        
        # Feature 3: Laplacian
        feature_maps[:, :, 3] = ndimage.laplace(middle_slice)
        
        # Feature 4: Local standard deviation
        feature_maps[:, :, 4] = ndimage.generic_filter(middle_slice, np.std, size=3)
        
        # Feature 5: Local mean
        feature_maps[:, :, 5] = ndimage.uniform_filter(middle_slice, size=3)
        
        # Feature 6: Maximum filter
        feature_maps[:, :, 6] = ndimage.maximum_filter(middle_slice, size=3)
        
        # Feature 7: Minimum filter
        feature_maps[:, :, 7] = ndimage.minimum_filter(middle_slice, size=3)
        
        # Normalize each feature map
        for i in range(8):
            fmap = feature_maps[:, :, i]
            feature_maps[:, :, i] = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        
        return feature_maps
    
    def _create_extended_feature_maps(self, patches: List[np.ndarray], labels: List[int], vis_dir: str):
        """Create EXTENDED feature maps with many more colorful visualizations"""
        print("Creating EXTENDED feature maps - viele bunte Bilder!")
        
        # Take multiple patches for more variety
        normal_patches = [p for p, l in zip(patches, labels) if l == 0]
        anomaly_patches = [p for p, l in zip(patches, labels) if l == 1]
        
        if len(normal_patches) == 0 or len(anomaly_patches) == 0:
            return
        
        # Create extended feature maps with 16 different features
        normal_patch = normal_patches[0]
        anomaly_patch = anomaly_patches[0]
        
        extended_normal = self._create_extended_anatomix_features(normal_patch)
        extended_anomaly = self._create_extended_anatomix_features(anomaly_patch)
        
        # Create large visualization with 16 features
        fig, axes = plt.subplots(2, 16, figsize=(32, 8))
        fig.suptitle('EXTENDED Anatomix Feature Maps - Viele Bunte Features!', fontsize=20, fontweight='bold')
        
        # Color maps for variety
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'spring', 'summer', 'autumn', 
                'winter', 'cool', 'hot', 'copper', 'rainbow', 'turbo', 'twilight', 'hsv']
        
        # Normal feature maps (top row)
        for i in range(16):
            im = axes[0, i].imshow(extended_normal[:, :, i], cmap=cmaps[i])
            axes[0, i].set_title(f'Normal\nFeature {i}', fontsize=8)
            axes[0, i].axis('off')
            plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Anomaly feature maps (bottom row)
        for i in range(16):
            im = axes[1, i].imshow(extended_anomaly[:, :, i], cmap=cmaps[i])
            axes[1, i].set_title(f'Anomaly\nFeature {i}', fontsize=8)
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # Save extended feature maps
        save_path = os.path.join(vis_dir, 'extended_feature_maps_16channels.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Extended feature maps (16 channels) saved: {save_path}")
    
    def _create_extended_anatomix_features(self, patch: np.ndarray) -> np.ndarray:
        """Create 16 different feature maps for more colorful visualizations (using only scipy/numpy)"""
        from scipy import ndimage
        
        # Take middle slice
        middle_slice = patch[:, :, patch.shape[2]//2]
        h, w = middle_slice.shape
        
        # Create 16 different feature maps
        feature_maps = np.zeros((h, w, 16))
        
        # Basic features
        feature_maps[:, :, 0] = middle_slice  # Original
        feature_maps[:, :, 1] = ndimage.gaussian_filter(middle_slice, sigma=0.5)  # Gaussian blur small
        feature_maps[:, :, 2] = ndimage.gaussian_filter(middle_slice, sigma=2.0)  # Gaussian blur large
        
        # Gradient features
        grad_x = ndimage.sobel(middle_slice, axis=0)
        grad_y = ndimage.sobel(middle_slice, axis=1)
        feature_maps[:, :, 3] = np.sqrt(grad_x**2 + grad_y**2)  # Gradient magnitude
        feature_maps[:, :, 4] = grad_x  # Gradient X
        feature_maps[:, :, 5] = grad_y  # Gradient Y
        
        # Laplacian variations
        feature_maps[:, :, 6] = ndimage.laplace(middle_slice)  # Laplacian
        feature_maps[:, :, 7] = ndimage.gaussian_laplace(middle_slice, sigma=1.0)  # Gaussian Laplacian
        
        # Local statistics
        feature_maps[:, :, 8] = ndimage.generic_filter(middle_slice, np.std, size=3)  # Local std
        feature_maps[:, :, 9] = ndimage.generic_filter(middle_slice, np.var, size=3)  # Local variance
        feature_maps[:, :, 10] = ndimage.uniform_filter(middle_slice, size=5)  # Local mean
        
        # Morphological operations
        feature_maps[:, :, 11] = ndimage.maximum_filter(middle_slice, size=3)  # Maximum filter
        feature_maps[:, :, 12] = ndimage.minimum_filter(middle_slice, size=3)  # Minimum filter
        feature_maps[:, :, 13] = ndimage.median_filter(middle_slice, size=3)  # Median filter
        
        # Additional scipy-based filters
        # Prewitt filter (edge detection)
        prewitt_x = ndimage.prewitt(middle_slice, axis=0)
        prewitt_y = ndimage.prewitt(middle_slice, axis=1)
        feature_maps[:, :, 14] = np.sqrt(prewitt_x**2 + prewitt_y**2)
        
        # Local range (max - min in neighborhood)
        feature_maps[:, :, 15] = ndimage.maximum_filter(middle_slice, size=5) - ndimage.minimum_filter(middle_slice, size=5)
        
        # Normalize each feature map
        for i in range(16):
            fmap = feature_maps[:, :, i]
            feature_maps[:, :, i] = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        
        return feature_maps
    
    def _create_multiple_patch_feature_gallery(self, patches: List[np.ndarray], labels: List[int], vis_dir: str):
        """Create a gallery with multiple patches and their feature representations"""
        print("Creating multiple patch feature gallery...")
        
        # Separate patches
        normal_patches = [p for p, l in zip(patches, labels) if l == 0]
        anomaly_patches = [p for p, l in zip(patches, labels) if l == 1]
        
        # Take up to 4 patches of each type
        n_normal = min(4, len(normal_patches))
        n_anomaly = min(4, len(anomaly_patches))
        
        if n_normal == 0 and n_anomaly == 0:
            return
        
        # Create mega visualization
        n_features_to_show = 8
        total_rows = n_normal + n_anomaly
        
        fig, axes = plt.subplots(total_rows, n_features_to_show + 1, figsize=(24, total_rows * 3))
        fig.suptitle('Multiple Patch Feature Gallery - Anatomix Feature Extraction', fontsize=18, fontweight='bold')
        
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'spring', 'summer', 'autumn']
        
        current_row = 0
        
        # Normal patches
        for i in range(n_normal):
            patch = normal_patches[i]
            features = self._simulate_anatomix_features(patch)
            
            # Original patch (first column)
            middle_slice = patch[:, :, patch.shape[2]//2]
            slice_norm = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min() + 1e-8)
            axes[current_row, 0].imshow(slice_norm, cmap='gray')
            axes[current_row, 0].set_title(f'Normal Patch {i+1}\n(Original)', fontsize=10)
            axes[current_row, 0].axis('off')
            
            # Feature maps (remaining columns)
            for j in range(n_features_to_show):
                im = axes[current_row, j+1].imshow(features[:, :, j], cmap=cmaps[j])
                axes[current_row, j+1].set_title(f'Feature {j}', fontsize=8)
                axes[current_row, j+1].axis('off')
                plt.colorbar(im, ax=axes[current_row, j+1], fraction=0.046, pad=0.04)
            
            current_row += 1
        
        # Anomaly patches
        for i in range(n_anomaly):
            patch = anomaly_patches[i]
            features = self._simulate_anatomix_features(patch)
            
            # Original patch (first column)
            middle_slice = patch[:, :, patch.shape[2]//2]
            slice_norm = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min() + 1e-8)
            axes[current_row, 0].imshow(slice_norm, cmap='gray')
            axes[current_row, 0].set_title(f'Anomaly Patch {i+1}\n(Original)', fontsize=10)
            axes[current_row, 0].axis('off')
            
            # Feature maps (remaining columns)
            for j in range(n_features_to_show):
                im = axes[current_row, j+1].imshow(features[:, :, j], cmap=cmaps[j])
                axes[current_row, j+1].set_title(f'Feature {j}', fontsize=8)
                axes[current_row, j+1].axis('off')
                plt.colorbar(im, ax=axes[current_row, j+1], fraction=0.046, pad=0.04)
            
            current_row += 1
        
        # Save multiple patch gallery
        save_path = os.path.join(vis_dir, 'multiple_patch_feature_gallery.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Multiple patch feature gallery saved: {save_path}")
        
        # Also create a MEGA feature showcase
        self._create_mega_feature_showcase(patches, labels, vis_dir)
    
    def _create_mega_feature_showcase(self, patches: List[np.ndarray], labels: List[int], vis_dir: str):
        """Create a MEGA showcase with tons of colorful feature visualizations"""
        print("Creating MEGA feature showcase - VIELE BUNTE BILDER!")
        
        # Take multiple patches
        normal_patches = [p for p, l in zip(patches, labels) if l == 0]
        anomaly_patches = [p for p, l in zip(patches, labels) if l == 1]
        
        all_patches = normal_patches[:6] + anomaly_patches[:6]  # Up to 12 patches total
        patch_labels = ['Normal'] * min(6, len(normal_patches)) + ['Anomaly'] * min(6, len(anomaly_patches))
        
        if len(all_patches) == 0:
            return
        
        # Create MASSIVE visualization - 12 patches x 8 features = 96 feature maps!
        n_patches = len(all_patches)
        n_features = 8
        
        fig, axes = plt.subplots(n_patches, n_features, figsize=(24, n_patches * 3))
        fig.suptitle('🌈 MEGA ANATOMIX FEATURE SHOWCASE - ALLE BUNTEN BILDER! 🌈', fontsize=20, fontweight='bold')
        
        # Different colormaps for maximum color variety
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'rainbow', 'hsv',
                'spring', 'summer', 'autumn', 'winter', 'cool', 'hot', 'copper', 'nipy_spectral']
        
        for patch_idx, (patch, label) in enumerate(zip(all_patches, patch_labels)):
            features = self._simulate_anatomix_features(patch)
            
            for feat_idx in range(n_features):
                cmap = cmaps[(patch_idx * n_features + feat_idx) % len(cmaps)]
                
                im = axes[patch_idx, feat_idx].imshow(features[:, :, feat_idx], cmap=cmap)
                axes[patch_idx, feat_idx].set_title(f'{label} P{patch_idx+1}\nFeat{feat_idx}', fontsize=8)
                axes[patch_idx, feat_idx].axis('off')
                
                # Add colorbar to every 4th feature for cleaner look
                if feat_idx % 4 == 0:
                    plt.colorbar(im, ax=axes[patch_idx, feat_idx], fraction=0.046, pad=0.04)
        
        # Save MEGA showcase
        save_path = os.path.join(vis_dir, 'MEGA_feature_showcase_ALL_COLORS.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🌈 MEGA feature showcase saved: {save_path}")
        
        # Create one more: Rainbow feature matrix
        self._create_rainbow_feature_matrix(all_patches, patch_labels, vis_dir)
    
    def _create_rainbow_feature_matrix(self, patches: List[np.ndarray], labels: List[str], vis_dir: str):
        """Create a rainbow-colored feature matrix"""
        print("Creating rainbow feature matrix...")
        
        if len(patches) == 0:
            return
        
        # Create one big matrix with all features from all patches
        all_features = []
        for patch in patches:
            features = self._simulate_anatomix_features(patch)
            all_features.append(features)
        
        # Create rainbow visualization
        n_patches = len(patches)
        n_cols = 4  # 4 patches per row
        n_rows = (n_patches + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(20, n_rows * 5))
        fig.suptitle('🌈 RAINBOW ANATOMIX FEATURE MATRIX 🌈', fontsize=24, fontweight='bold')
        
        rainbow_cmaps = ['rainbow', 'hsv', 'turbo', 'nipy_spectral', 'gist_rainbow', 'gist_ncar']
        
        for patch_idx in range(n_patches):
            # Create subplot for this patch (2x4 grid of features)
            for feat_row in range(2):
                for feat_col in range(4):
                    feat_idx = feat_row * 4 + feat_col
                    if feat_idx >= 8:
                        break
                    
                    subplot_idx = patch_idx * 8 + feat_idx + 1
                    plt.subplot(n_rows * 2, n_cols * 4, subplot_idx)
                    
                    cmap = rainbow_cmaps[feat_idx % len(rainbow_cmaps)]
                    
                    im = plt.imshow(all_features[patch_idx][:, :, feat_idx], cmap=cmap)
                    plt.title(f'{labels[patch_idx]}\nP{patch_idx+1}F{feat_idx}', fontsize=6)
                    plt.axis('off')
        
        # Save rainbow matrix
        save_path = os.path.join(vis_dir, 'RAINBOW_feature_matrix.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🌈 Rainbow feature matrix saved: {save_path}")
    
    def _create_compact_patch_visualization(self, patches: List[np.ndarray], labels: List[int], 
                                          subjects: List[str], vis_dir: str):
        """Create compact visualization similar to anatomix_knn_brats.py"""
        print("Creating compact patch visualization...")
        
        # Select representative patches
        normal_patches = [p for p, l in zip(patches, labels) if l == 0]
        anomaly_patches = [p for p, l in zip(patches, labels) if l == 1]
        
        n_normal = min(6, len(normal_patches))
        n_anomaly = min(6, len(anomaly_patches))
        
        if n_normal == 0 and n_anomaly == 0:
            return
        
        # Create compact visualization
        fig, axes = plt.subplots(2, 6, figsize=(18, 8))
        fig.suptitle('Compact Patch Visualization - Representative Samples', fontsize=16, fontweight='bold')
        
        # Normal patches (top row)
        for i in range(6):
            if i < n_normal:
                patch = normal_patches[i]
                middle_slice = patch[:, :, patch.shape[2]//2]
                slice_norm = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min() + 1e-8)
                
                axes[0, i].imshow(slice_norm, cmap='gray')
                axes[0, i].set_title(f'Normal {i+1}', fontsize=10)
            else:
                axes[0, i].axis('off')
            axes[0, i].axis('off')
        
        # Anomaly patches (bottom row)
        for i in range(6):
            if i < n_anomaly:
                patch = anomaly_patches[i]
                middle_slice = patch[:, :, patch.shape[2]//2]
                slice_norm = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min() + 1e-8)
                
                axes[1, i].imshow(slice_norm, cmap='gray')
                axes[1, i].set_title(f'Anomaly {i+1}', fontsize=10)
            else:
                axes[1, i].axis('off')
            axes[1, i].axis('off')
        
        # Save compact visualization
        save_path = os.path.join(vis_dir, 'compact_patch_visualization.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Compact patch visualization saved: {save_path}")
    
    def _create_feature_comparison_visualization(self, features: np.ndarray, labels: np.ndarray, 
                                               subjects: List[str], vis_dir: str):
        """Create visualization comparing extracted features"""
        print("Creating feature comparison visualization...")
        
        normal_features = features[labels == 0]
        anomaly_features = features[labels == 1]
        
        if len(normal_features) == 0 or len(anomaly_features) == 0:
            return
        
        # Reshape features to create "feature images" 
        # Since our features are flattened, we'll create a grid representation
        feature_dim = features.shape[1]
        
        # Try to create a square-ish grid
        grid_size = int(np.ceil(np.sqrt(feature_dim)))
        
        # Take representative samples
        normal_sample = normal_features[0]
        anomaly_sample = anomaly_features[0]
        
        # Pad features to make them square
        padded_size = grid_size * grid_size
        normal_padded = np.pad(normal_sample, (0, padded_size - feature_dim), mode='constant')
        anomaly_padded = np.pad(anomaly_sample, (0, padded_size - feature_dim), mode='constant')
        
        # Reshape to grid
        normal_grid = normal_padded.reshape(grid_size, grid_size)
        anomaly_grid = anomaly_padded.reshape(grid_size, grid_size)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Anatomix Feature Representation Comparison', fontsize=16, fontweight='bold')
        
        # Normal features
        im1 = axes[0].imshow(normal_grid, cmap='viridis')
        axes[0].set_title('Normal Sample Features')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Anomaly features
        im2 = axes[1].imshow(anomaly_grid, cmap='plasma')
        axes[1].set_title('Anomaly Sample Features')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference
        diff_grid = anomaly_grid - normal_grid
        im3 = axes[2].imshow(diff_grid, cmap='RdBu_r')
        axes[2].set_title('Feature Difference (Anomaly - Normal)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Save feature comparison
        save_path = os.path.join(vis_dir, 'feature_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature comparison visualization saved: {save_path}")

class FeatureVisualizer:
    """Class for visualizing extracted Anatomix features"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def visualize_feature_distributions(self, features: np.ndarray, labels: np.ndarray, 
                                      subjects: List[str], output_subdir: str = 'feature_analysis'):
        """Visualize feature distributions and statistics"""
        vis_dir = os.path.join(self.config.output_dir, output_subdir)
        os.makedirs(vis_dir, exist_ok=True)
        
        print(f"\n📊 CREATING FEATURE VISUALIZATIONS...")
        print(f"Feature shape: {features.shape}")
        print(f"Saving to: {vis_dir}")
        
        # 1. Feature correlation heatmap
        self._plot_feature_correlation_heatmap(features, vis_dir)
        
        # 2. Feature distribution histograms
        self._plot_feature_histograms(features, labels, vis_dir)
        
        # 3. Feature statistics by class
        self._plot_feature_statistics_by_class(features, labels, vis_dir)
        
        # 4. Feature importance visualization
        self._plot_feature_importance(features, labels, vis_dir)
        
        # 5. Feature clustering visualization
        self._plot_feature_clustering(features, labels, subjects, vis_dir)
        
        # 6. Individual feature analysis
        self._plot_individual_features(features, labels, vis_dir)
        
        print(f"✓ Feature visualizations saved to: {vis_dir}")
    
    def _plot_feature_correlation_heatmap(self, features: np.ndarray, vis_dir: str):
        """Plot correlation heatmap of features"""
        print("Creating feature correlation heatmap...")
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(features, rowvar=False)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Use seaborn for better heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Only show lower triangle
        
        sns.heatmap(correlation_matrix, mask=mask, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Anatomix Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Feature Index', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_correlation_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a simplified version for highly correlated features
        high_corr_mask = (np.abs(correlation_matrix) > 0.7) & ~np.eye(correlation_matrix.shape[0], dtype=bool)
        if np.any(high_corr_mask):
            plt.figure(figsize=(10, 8))
            high_corr_features = correlation_matrix.copy()
            high_corr_features[~high_corr_mask] = 0
            
            sns.heatmap(high_corr_features, cmap='RdBu_r', center=0, square=True, 
                       fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('High Correlation Features (|r| > 0.7)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'high_correlation_features.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_feature_histograms(self, features: np.ndarray, labels: np.ndarray, vis_dir: str):
        """Plot histograms of feature distributions"""
        print("Creating feature distribution histograms...")
        
        normal_features = features[labels == 0]
        anomaly_features = features[labels == 1]
        
        n_features = features.shape[1]
        
        # Create comprehensive histogram grid
        n_cols = 6
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i in range(n_features):
            ax = axes[i]
            
            # Plot histograms for normal and anomaly features
            ax.hist(normal_features[:, i], bins=30, alpha=0.7, label='Normal', 
                   color='blue', density=True)
            ax.hist(anomaly_features[:, i], bins=30, alpha=0.7, label='Anomaly', 
                   color='red', density=True)
            
            ax.set_title(f'Feature {i}', fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Anatomix Feature Distributions by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_histograms_all.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a focused view of the most discriminative features
        self._plot_top_discriminative_features(normal_features, anomaly_features, vis_dir)
    
    def _plot_top_discriminative_features(self, normal_features: np.ndarray, 
                                        anomaly_features: np.ndarray, vis_dir: str):
        """Plot the top discriminative features"""
        n_features = normal_features.shape[1]
        
        # Calculate discrimination metric (difference in means normalized by pooled std)
        normal_mean = np.mean(normal_features, axis=0)
        anomaly_mean = np.mean(anomaly_features, axis=0)
        normal_std = np.std(normal_features, axis=0)
        anomaly_std = np.std(anomaly_features, axis=0)
        
        pooled_std = np.sqrt((normal_std**2 + anomaly_std**2) / 2)
        discrimination_score = np.abs(normal_mean - anomaly_mean) / (pooled_std + 1e-8)
        
        # Get top 12 most discriminative features
        top_indices = np.argsort(discrimination_score)[-12:][::-1]
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, feature_idx in enumerate(top_indices):
            ax = axes[i]
            
            ax.hist(normal_features[:, feature_idx], bins=30, alpha=0.7, 
                   label=f'Normal (μ={normal_mean[feature_idx]:.3f})', 
                   color='blue', density=True)
            ax.hist(anomaly_features[:, feature_idx], bins=30, alpha=0.7, 
                   label=f'Anomaly (μ={anomaly_mean[feature_idx]:.3f})', 
                   color='red', density=True)
            
            ax.set_title(f'Feature {feature_idx} (Score: {discrimination_score[feature_idx]:.3f})', 
                        fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Top 12 Most Discriminative Anatomix Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'top_discriminative_features.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_statistics_by_class(self, features: np.ndarray, labels: np.ndarray, vis_dir: str):
        """Plot feature statistics comparison between classes"""
        print("Creating feature statistics comparison...")
        
        normal_features = features[labels == 0]
        anomaly_features = features[labels == 1]
        
        # Calculate statistics
        normal_stats = {
            'mean': np.mean(normal_features, axis=0),
            'std': np.std(normal_features, axis=0),
            'median': np.median(normal_features, axis=0),
            'q25': np.percentile(normal_features, 25, axis=0),
            'q75': np.percentile(normal_features, 75, axis=0)
        }
        
        anomaly_stats = {
            'mean': np.mean(anomaly_features, axis=0),
            'std': np.std(anomaly_features, axis=0),
            'median': np.median(anomaly_features, axis=0),
            'q25': np.percentile(anomaly_features, 25, axis=0),
            'q75': np.percentile(anomaly_features, 75, axis=0)
        }
        
        n_features = features.shape[1]
        feature_indices = np.arange(n_features)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean comparison
        axes[0, 0].plot(feature_indices, normal_stats['mean'], 'b-o', label='Normal', markersize=3)
        axes[0, 0].plot(feature_indices, anomaly_stats['mean'], 'r-s', label='Anomaly', markersize=3)
        axes[0, 0].set_title('Feature Means by Class')
        axes[0, 0].set_xlabel('Feature Index')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation comparison
        axes[0, 1].plot(feature_indices, normal_stats['std'], 'b-o', label='Normal', markersize=3)
        axes[0, 1].plot(feature_indices, anomaly_stats['std'], 'r-s', label='Anomaly', markersize=3)
        axes[0, 1].set_title('Feature Standard Deviations by Class')
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Median comparison
        axes[1, 0].plot(feature_indices, normal_stats['median'], 'b-o', label='Normal', markersize=3)
        axes[1, 0].plot(feature_indices, anomaly_stats['median'], 'r-s', label='Anomaly', markersize=3)
        axes[1, 0].set_title('Feature Medians by Class')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Median Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Range comparison (IQR)
        normal_iqr = normal_stats['q75'] - normal_stats['q25']
        anomaly_iqr = anomaly_stats['q75'] - anomaly_stats['q25']
        axes[1, 1].plot(feature_indices, normal_iqr, 'b-o', label='Normal', markersize=3)
        axes[1, 1].plot(feature_indices, anomaly_iqr, 'r-s', label='Anomaly', markersize=3)
        axes[1, 1].set_title('Feature Interquartile Range by Class')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('IQR')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Anatomix Feature Statistics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_statistics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, features: np.ndarray, labels: np.ndarray, vis_dir: str):
        """Plot feature importance based on statistical tests"""
        print("Calculating feature importance...")
        
        from scipy.stats import ttest_ind
        
        normal_features = features[labels == 0]
        anomaly_features = features[labels == 1]
        
        n_features = features.shape[1]
        p_values = []
        effect_sizes = []
        
        for i in range(n_features):
            # T-test
            t_stat, p_val = ttest_ind(normal_features[:, i], anomaly_features[:, i])
            p_values.append(p_val)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(normal_features) - 1) * np.var(normal_features[:, i]) + 
                                 (len(anomaly_features) - 1) * np.var(anomaly_features[:, i])) / 
                                (len(normal_features) + len(anomaly_features) - 2))
            cohen_d = (np.mean(anomaly_features[:, i]) - np.mean(normal_features[:, i])) / pooled_std
            effect_sizes.append(abs(cohen_d))
        
        p_values = np.array(p_values)
        effect_sizes = np.array(effect_sizes)
        
        # Plot feature importance
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # P-values
        axes[0].bar(range(n_features), -np.log10(p_values + 1e-10), color='skyblue', alpha=0.7)
        axes[0].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[0].axhline(y=-np.log10(0.01), color='darkred', linestyle='--', label='p=0.01')
        axes[0].set_title('Feature Significance (-log10 p-value)')
        axes[0].set_xlabel('Feature Index')
        axes[0].set_ylabel('-log10(p-value)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Effect sizes
        axes[1].bar(range(n_features), effect_sizes, color='lightcoral', alpha=0.7)
        axes[1].axhline(y=0.2, color='green', linestyle='--', label='Small effect')
        axes[1].axhline(y=0.5, color='orange', linestyle='--', label='Medium effect')
        axes[1].axhline(y=0.8, color='red', linestyle='--', label='Large effect')
        axes[1].set_title('Feature Effect Size (|Cohen\'s d|)')
        axes[1].set_xlabel('Feature Index')
        axes[1].set_ylabel('|Cohen\'s d|')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Anatomix Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature ranking
        feature_ranking = np.argsort(effect_sizes)[::-1]
        ranking_data = {
            'Feature_Index': feature_ranking,
            'Effect_Size': effect_sizes[feature_ranking],
            'P_Value': p_values[feature_ranking],
            'Significance': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' 
                           for p in p_values[feature_ranking]]
        }
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df.to_csv(os.path.join(vis_dir, 'feature_ranking.csv'), index=False)
    
    def _plot_feature_clustering(self, features: np.ndarray, labels: np.ndarray, 
                               subjects: List[str], vis_dir: str):
        """Plot feature clustering analysis"""
        print("Creating feature clustering visualization...")
        
        # Hierarchical clustering of features
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Transpose to cluster features (not samples)
        feature_distances = pdist(features.T, metric='correlation')
        feature_linkage = linkage(feature_distances, method='ward')
        
        plt.figure(figsize=(12, 8))
        dendrogram(feature_linkage, labels=range(features.shape[1]), leaf_rotation=90)
        plt.title('Hierarchical Clustering of Anatomix Features', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_clustering_dendrogram.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature clustering heatmap
        from scipy.cluster.hierarchy import fcluster
        
        # Get clusters
        cluster_ids = fcluster(feature_linkage, t=5, criterion='maxclust')
        
        # Reorder features by clusters
        cluster_order = np.argsort(cluster_ids)
        reordered_features = features[:, cluster_order]
        
        # Create clustered correlation matrix
        correlation_matrix = np.corrcoef(reordered_features, rowvar=False)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, cmap='RdBu_r', center=0, square=True)
        plt.title('Feature Correlation Matrix (Clustered)', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Index (Reordered)')
        plt.ylabel('Feature Index (Reordered)')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_correlation_clustered.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_features(self, features: np.ndarray, labels: np.ndarray, vis_dir: str):
        """Plot detailed analysis of individual features"""
        print("Creating individual feature analysis...")
        
        individual_dir = os.path.join(vis_dir, 'individual_features')
        os.makedirs(individual_dir, exist_ok=True)
        
        normal_features = features[labels == 0]
        anomaly_features = features[labels == 1]
        
        # Analyze top 6 most important features
        normal_mean = np.mean(normal_features, axis=0)
        anomaly_mean = np.mean(anomaly_features, axis=0)
        mean_diff = np.abs(normal_mean - anomaly_mean)
        top_features = np.argsort(mean_diff)[-6:][::-1]
        
        for feature_idx in top_features:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Histogram
            axes[0, 0].hist(normal_features[:, feature_idx], bins=30, alpha=0.7, 
                           label='Normal', color='blue', density=True)
            axes[0, 0].hist(anomaly_features[:, feature_idx], bins=30, alpha=0.7, 
                           label='Anomaly', color='red', density=True)
            axes[0, 0].set_title(f'Distribution - Feature {feature_idx}')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot
            data_to_plot = [normal_features[:, feature_idx], anomaly_features[:, feature_idx]]
            box = axes[0, 1].boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True)
            box['boxes'][0].set_facecolor('lightblue')
            box['boxes'][1].set_facecolor('lightcoral')
            axes[0, 1].set_title(f'Box Plot - Feature {feature_idx}')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            normal_sorted = np.sort(normal_features[:, feature_idx])
            anomaly_sorted = np.sort(anomaly_features[:, feature_idx])
            
            # Use shorter array length for Q-Q plot
            min_len = min(len(normal_sorted), len(anomaly_sorted))
            normal_quantiles = normal_sorted[np.linspace(0, len(normal_sorted)-1, min_len).astype(int)]
            anomaly_quantiles = anomaly_sorted[np.linspace(0, len(anomaly_sorted)-1, min_len).astype(int)]
            
            axes[1, 0].scatter(normal_quantiles, anomaly_quantiles, alpha=0.6)
            min_val = min(normal_quantiles.min(), anomaly_quantiles.min())
            max_val = max(normal_quantiles.max(), anomaly_quantiles.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[1, 0].set_title(f'Q-Q Plot - Feature {feature_idx}')
            axes[1, 0].set_xlabel('Normal Quantiles')
            axes[1, 0].set_ylabel('Anomaly Quantiles')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Statistics table
            axes[1, 1].axis('off')
            stats_text = f"""
Feature {feature_idx} Statistics:

Normal:
  Mean: {np.mean(normal_features[:, feature_idx]):.4f}
  Std: {np.std(normal_features[:, feature_idx]):.4f}
  Median: {np.median(normal_features[:, feature_idx]):.4f}
  Min: {np.min(normal_features[:, feature_idx]):.4f}
  Max: {np.max(normal_features[:, feature_idx]):.4f}

Anomaly:
  Mean: {np.mean(anomaly_features[:, feature_idx]):.4f}
  Std: {np.std(anomaly_features[:, feature_idx]):.4f}
  Median: {np.median(anomaly_features[:, feature_idx]):.4f}
  Min: {np.min(anomaly_features[:, feature_idx]):.4f}
  Max: {np.max(anomaly_features[:, feature_idx]):.4f}

Difference:
  Mean Diff: {abs(np.mean(anomaly_features[:, feature_idx]) - np.mean(normal_features[:, feature_idx])):.4f}
            """
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f'Detailed Analysis - Anatomix Feature {feature_idx}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(individual_dir, f'feature_{feature_idx}_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_feature_summary_report(self, features: np.ndarray, labels: np.ndarray, 
                                    subjects: List[str]):
        """Create comprehensive feature analysis report"""
        print(f"\n🎨 CREATING COMPREHENSIVE FEATURE ANALYSIS...")
        
        # Create main feature analysis
        self.visualize_feature_distributions(features, labels, subjects)
        
        # Create summary statistics
        self._create_feature_summary_stats(features, labels)
        
        print(f"✓ Comprehensive feature analysis complete!")
    
    def _create_feature_summary_stats(self, features: np.ndarray, labels: np.ndarray):
        """Create and save feature summary statistics"""
        vis_dir = os.path.join(self.config.output_dir, 'feature_analysis')
        
        normal_features = features[labels == 0]
        anomaly_features = features[labels == 1]
        
        # Comprehensive statistics
        summary_stats = {
            'Feature_Index': range(features.shape[1]),
            'Normal_Mean': np.mean(normal_features, axis=0),
            'Normal_Std': np.std(normal_features, axis=0),
            'Normal_Median': np.median(normal_features, axis=0),
            'Anomaly_Mean': np.mean(anomaly_features, axis=0),
            'Anomaly_Std': np.std(anomaly_features, axis=0),
            'Anomaly_Median': np.median(anomaly_features, axis=0),
            'Mean_Difference': np.abs(np.mean(anomaly_features, axis=0) - np.mean(normal_features, axis=0)),
            'Std_Ratio': np.std(anomaly_features, axis=0) / (np.std(normal_features, axis=0) + 1e-8)
        }
        
        stats_df = pd.DataFrame(summary_stats)
        stats_df.to_csv(os.path.join(vis_dir, 'feature_summary_statistics.csv'), index=False)
        
        print(f"Feature summary statistics saved to: {os.path.join(vis_dir, 'feature_summary_statistics.csv')}")


class Visualizer:
    """Class for creating various visualizations (from ae_brats.py)"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray):
        """Plot confusion matrix with improved formatting using Plotly"""
        cm = confusion_matrix(true_labels, predictions)
        
        if self.config.verbose:
            print(f"\nConfusion Matrix Details:")
            print(f"True Labels - Normal: {np.sum(true_labels == 0)}, Anomaly: {np.sum(true_labels == 1)}")
            print(f"Predictions - Normal: {np.sum(predictions == 0)}, Anomaly: {np.sum(predictions == 1)}")
            print(f"Confusion Matrix:\n{cm}")

        if cm.shape != (2, 2):
            if self.config.verbose:
                print("WARNING: Confusion matrix is not 2x2. Skipping plot generation.")
            return

        tn, fp, fn, tp = cm.ravel()
        
        z = [[tn, fp], [fn, tp]]
        x = ['Normal (0)', 'Anomaly (1)']
        y = ['Normal (0)', 'Anomaly (1)']

        row_sums = cm.sum(axis=1)
        
        # Avoid division by zero if a class has no samples
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
            title_text='<b>Confusion Matrix - Anatomix + KNN</b><br>(Count and Percentage)',
            title_x=0.5,
            xaxis=dict(title='<b>Predicted Label</b>'),
            yaxis=dict(title='<b>True Label</b>', autorange='reversed'),
            font=dict(size=14)
        )
        fig.update_xaxes(side="bottom")

        total_samples = cm.sum()
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        summary_text = (f'Total Samples: {total_samples}<br>'
                       f'Accuracy: {accuracy:.3f}<br>'
                       f'Precision: {precision:.3f}<br>'
                       f'Recall: {recall:.3f}<br>'
                       f'Specificity: {specificity:.3f}')
        
        fig.add_annotation(
            text=summary_text,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.0,
            y=-0.28,
            bordercolor="black",
            borderwidth=1,
            bgcolor="lightgray",
            font_size=12
        )
        
        fig.update_layout(margin=dict(t=100, b=150))

        output_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        try:
            fig.write_image(output_path, width=800, height=800, scale=2)
            if self.config.verbose:
                print(f"Confusion Matrix plot saved to {output_path}")
        except ValueError as e: 
            print(f"ERROR: Could not save confusion matrix plot: {e}")
    
    def plot_roc_curve(self, true_labels: np.ndarray, anomaly_scores: np.ndarray):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
        auc = roc_auc_score(true_labels, anomaly_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f'ROC (AUC = {auc:.2f})')
        plt.fill_between(fpr, tpr, step="pre", alpha=0.25, color="#aec7e8")
        plt.plot([0, 1], [0, 1], linestyle='--', color='#888888', lw=1, label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'roc_curve.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, true_labels: np.ndarray, anomaly_scores: np.ndarray):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(true_labels, anomaly_scores)
        avg_precision = average_precision_score(true_labels, anomaly_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='#ff7f0e', lw=2, label=f'PR (AUC = {avg_precision:.2f})')
        plt.fill_between(recall, precision, step='pre', alpha=0.25, color='#ffbb78')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'precision_recall_curve.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_score_histogram(self, anomaly_scores: np.ndarray, true_labels: np.ndarray, 
                           optimal_threshold: float):
        """Plot histogram of anomaly scores"""
        normal_scores = anomaly_scores[true_labels == 0]
        anomaly_scores_subset = anomaly_scores[true_labels == 1]
        
        plt.figure(figsize=(12, 6))
        
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores_subset, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, 
                   label=f'Optimal Threshold = {optimal_threshold:.6f}')
        
        plt.xlabel('Anomaly Score (KNN Distance)')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores - Anatomix + KNN')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'score_histogram.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_embeddings(self, features: np.ndarray, labels: np.ndarray, method: str = 'tsne'):
        """Plot t-SNE or PCA visualization of features"""
        print(f"Creating {method.upper()} visualization...")
        
        # Limit samples for visualization if too many
        if len(features) > 2000:
            indices = np.random.choice(len(features), 2000, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
            title = 't-SNE Visualization of Anatomix Features'
        else:
            reducer = PCA(n_components=2, random_state=42)
            title = 'PCA Visualization of Anatomix Features'
        
        reduced_features = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_features[labels == 0, 0], reduced_features[labels == 0, 1], 
                             c='blue', alpha=0.6, label='Normal', s=20)
        scatter = plt.scatter(reduced_features[labels == 1, 0], reduced_features[labels == 1, 1], 
                             c='red', alpha=0.6, label='Anomaly', s=20)
        
        if method.lower() == 'pca':
            plt.xlabel(f'PC1 ({reducer.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({reducer.explained_variance_ratio_[1]:.2%} variance)')
        else:
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
        
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, f'{method}_features.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, results: Dict):
        """Create a summary report with all visualizations"""
        self.plot_confusion_matrix(results['true_labels'], results['predictions'])
        self.plot_roc_curve(results['true_labels'], results['anomaly_scores'])
        self.plot_precision_recall_curve(results['true_labels'], results['anomaly_scores'])
        self.plot_score_histogram(results['anomaly_scores'], results['true_labels'], 
                                results['optimal_threshold'])
        
        print(f"\nAll visualizations saved to: {self.config.output_dir}")

def main():
    """Main function to run the complete Anatomix + KNN pipeline"""
    # Start timing the entire process
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Anatomix + KNN Unsupervised Anomaly Detection for BraTS Dataset')
    parser.add_argument('--num_subjects', type=int, default=None, 
                       help='Number of subjects to use (default: all)')
    parser.add_argument('--patch_size', type=int, default=32, 
                       help='Size of 3D patches (default: 32)')
    parser.add_argument('--patches_per_volume', type=int, default=50, 
                       help='Number of patches per volume (default: 50)')
    parser.add_argument('--k_neighbors', type=int, default=7, 
                       help='Number of KNN neighbors (default: 7)')
    parser.add_argument('--output_dir', type=str, default='anatomix_knn_unsupervised_results', 
                       help='Output directory (default: anatomix_knn_unsupervised_results)')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData', 
                       help='Path to BraTS dataset')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3,
                       help='Maximum ratio of normal to anomaly patches (default: 3)')
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 2, 4],
                       help='BraTS segmentation labels to consider as anomalies (default: [1, 2, 4])')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output with detailed debug information')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Update config with command line arguments
    config.num_subjects = args.num_subjects
    config.patch_size = args.patch_size
    config.patches_per_volume = args.patches_per_volume
    config.k_neighbors = args.k_neighbors
    config.output_dir = args.output_dir
    config.dataset_path = args.dataset_path
    config.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    config.anomaly_labels = args.anomaly_labels
    config.verbose = args.verbose
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    if config.verbose:
        print("="*60)
        print("ANATOMIX + KNN UNSUPERVISED ANOMALY DETECTION FOR BRATS")
        print("="*60)
        print(f"Device: {config.device}")
        print(f"Dataset path: {config.dataset_path}")
        print(f"Output directory: {config.output_dir}")
        print(f"Patch size: {config.patch_size}x{config.patch_size}x{config.patch_size}")
        print(f"Patches per volume: {config.patches_per_volume}")
        print(f"K neighbors: {config.k_neighbors}")
        print(f"Number of subjects: {config.num_subjects if config.num_subjects else 'All'}")
        
        # Explain anomaly labels
        label_names = {0: "Background/Normal", 1: "NCR/NET (Necrotic/Non-enhancing)", 
                       2: "ED (Edema)", 4: "ET (Enhancing Tumor)"}
        anomaly_names = [f"{label} ({label_names.get(label, 'Unknown')})" for label in config.anomaly_labels]
        print(f"Anomaly labels: {anomaly_names}")
        print("="*60)
    else:
        # Minimal output
        anomaly_names = [f"{label}" for label in config.anomaly_labels]
        print(f"Anatomix + KNN Unsupervised Anomaly Detection | Anomaly labels: {anomaly_names} | Output: {config.output_dir}")
    
    # Step 1: Load Anatomix model
    if config.verbose:
        print("\n1. Loading Anatomix model...")
    anatomix_model = load_anatomix_model()
    
    # Step 2: Process dataset and extract Anatomix features
    if config.verbose:
        print("\n2. Processing dataset and extracting Anatomix features...")
    processor = AnatomixKNNProcessor(config)
    features, labels, subjects = processor.process_dataset(anatomix_model, config.num_subjects)
    
    if len(features) == 0:
        print("Error: No features extracted! Please check your dataset path and structure.")
        return
    
    if config.verbose:
        print(f"Total features extracted: {len(features)}")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Normal patches: {np.sum(labels == 0)}")
        print(f"Anomalous patches: {np.sum(labels == 1)}")
    
    # Check if we have both classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    if config.verbose:
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    if len(unique_labels) < 2:
        print("ERROR: Dataset contains only one class! Cannot perform anomaly detection.")
        return
    
    # Step 3: Subject-level data splitting (from ae_brats.py)
    if config.verbose:
        print("\n3. Subject-level data splitting for unsupervised anomaly detection...")
    
    unique_subjects = list(set(subjects))
    if config.verbose:
        print(f"Total unique subjects: {len(unique_subjects)}")
    
    # Split subjects (not patches) into train/val/test
    np.random.shuffle(unique_subjects)
    n_subjects = len(unique_subjects)
    train_subjects = unique_subjects[:int(0.6 * n_subjects)]  # 60% for training
    val_subjects = unique_subjects[int(0.6 * n_subjects):int(0.8 * n_subjects)]  # 20% for validation
    test_subjects = unique_subjects[int(0.8 * n_subjects):]  # 20% for testing
    
    if config.verbose:
        print(f"Subject distribution:")
        print(f"  Training subjects: {len(train_subjects)}")
        print(f"  Validation subjects: {len(val_subjects)}")
        print(f"  Test subjects: {len(test_subjects)}")
    
    # Create feature-level splits based on subject assignment
    train_indices = [i for i, subj in enumerate(subjects) if subj in train_subjects]
    val_indices = [i for i, subj in enumerate(subjects) if subj in val_subjects]
    test_indices = [i for i, subj in enumerate(subjects) if subj in test_subjects]
    
    X_train_all = features[train_indices]
    y_train_all = labels[train_indices]
    
    X_val_all = features[val_indices]
    y_val_all = labels[val_indices]
    
    X_test = features[test_indices]
    y_test = labels[test_indices]
    
    # UNSUPERVISED CONSTRAINT: Only use NORMAL features for training and validation
    train_normal_mask = (y_train_all == 0)
    val_normal_mask = (y_val_all == 0)
    
    X_train_normal = X_train_all[train_normal_mask]
    X_val_normal = X_val_all[val_normal_mask]
    
    if config.verbose:
        print(f"\n=== SUBJECT-LEVEL UNSUPERVISED ANOMALY DETECTION SPLIT ===")
        print(f"Training set (NORMAL ONLY): {len(X_train_normal)} features from {len(train_subjects)} subjects")
        print(f"Validation set (NORMAL ONLY): {len(X_val_normal)} features from {len(val_subjects)} subjects") 
        print(f"Test set (MIXED): {len(X_test)} features from {len(test_subjects)} subjects")
        print(f"  Normal: {np.sum(y_test == 0)}, Anomalous: {np.sum(y_test == 1)}")
        print(f"========================================================")
    
    # Verify no subject appears in multiple splits
    assert len(set(train_subjects) & set(val_subjects)) == 0, "Subject overlap between train and validation!"
    assert len(set(train_subjects) & set(test_subjects)) == 0, "Subject overlap between train and test!"
    assert len(set(val_subjects) & set(test_subjects)) == 0, "Subject overlap between validation and test!"
    if config.verbose:
        print("✓ No subject overlap confirmed - data leakage prevented!")
    
    # Step 4: Build FAISS index with normal training data only
    if config.verbose:
        print("\n4. Building FAISS index with normal training features...")
    
    # ADD: Statistical validation before KNN anomaly detection
    if config.verbose:
        print("\n📊 DATA SCIENCE VALIDATION: Checking KNN suitability...")
        
        # Check dimensionality and sample size ratio
        n_samples, n_features = X_train_normal.shape
        samples_per_feature = n_samples / n_features
        print(f"Samples per feature: {samples_per_feature:.1f} (should be > 5 for reliable KNN)")
        
        if samples_per_feature < 5:
            print("⚠️  WARNING: Low samples-to-features ratio. Consider dimensionality reduction.")
        
        # Check for curse of dimensionality (distance concentration)
        if n_features > 20:
            # Sample a subset to check distance concentration
            sample_size = min(1000, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_features = X_train_normal[sample_indices]
            
            # Calculate pairwise distances
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(sample_features[:100])  # Only first 100 for efficiency
            distance_values = distances[np.triu_indices_from(distances, k=1)]
            
            # Check concentration (ratio of max to min distance)
            concentration_ratio = np.max(distance_values) / np.mean(distance_values)
            print(f"Distance concentration ratio: {concentration_ratio:.2f} (lower is better)")
            
            if concentration_ratio < 2:
                print("⚠️  WARNING: High dimensional data with distance concentration detected!")
                print("   Consider: PCA, feature selection, or different similarity metric")
        
        # Check feature correlations
        correlation_matrix = np.corrcoef(X_train_normal, rowvar=False)
        high_correlations = np.sum(np.abs(correlation_matrix) > 0.9) - n_features  # Exclude diagonal
        correlation_pct = (high_correlations / (n_features * (n_features - 1))) * 100
        print(f"High correlations (>0.9): {correlation_pct:.1f}% of feature pairs")
        
        if correlation_pct > 20:
            print("⚠️  WARNING: Many highly correlated features. Consider feature selection.")
        
        print("✓ Statistical validation complete")
    
    train_index = build_faiss_index(X_train_normal, use_gpu=torch.cuda.is_available())
    
    # Step 5: Evaluate the model
    if config.verbose:
        print("\n5. Evaluating Anatomix + KNN anomaly detection...")
    evaluator = AnomalyDetectionEvaluator(config)
    results = evaluator.evaluate(X_test, y_test, X_val_normal, train_index)
    
    # Step 6: Create visualizations
    if config.verbose:
        print("\n6. Creating visualizations...")
    visualizer = Visualizer(config)
    visualizer.create_summary_report(results)
    
    # Step 7: Create feature embeddings visualization
    if config.verbose:
        print("\n7. Creating feature embeddings visualization...")
    visualizer.plot_feature_embeddings(X_test, y_test, method='tsne')
    visualizer.plot_feature_embeddings(X_test, y_test, method='pca')
    
        # Step 8: Create feature analysis visualizations
    if config.verbose:
        print("\n8. Creating feature analysis visualizations...")
    feature_visualizer = FeatureVisualizer(config)
    feature_visualizer.create_feature_summary_report(features, labels, subjects)
    
    # Step 9: Create patch feature visualizations (from anatomix_knn_brats.py)
    if config.verbose:
        print("\n9. Creating patch feature visualizations...")
    patch_visualizer = PatchFeatureVisualizer(config)
    patch_visualizer.create_patch_feature_visualizations(processor, anatomix_model, features, labels, subjects)
    
    # Calculate total execution time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Convert time to human-readable format
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    time_str = []
    if hours > 0:
        time_str.append(f"{hours}h")
    if minutes > 0:
        time_str.append(f"{minutes}m")
    time_str.append(f"{seconds}s")
    time_formatted = " ".join(time_str)
    
    # Save results to file
    results_file = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("ANATOMIX + KNN UNSUPERVISED ANOMALY DETECTION RESULTS\n")
        f.write("="*60 + "\n")
        f.write("METHODOLOGY:\n")
        f.write("- Anatomix pre-trained features for representation learning\n")
        f.write("- KNN-based anomaly detection with FAISS for efficiency\n")
        f.write("- Subject-level data splitting (no patient overlap)\n")
        f.write("- Threshold determined using ONLY normal validation data\n")
        f.write("- No test label access during threshold selection\n")
        f.write("="*60 + "\n")
        f.write(f"ROC AUC:           {results['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {results['average_precision']:.4f}\n")
        f.write(f"Matthews Correlation: {results['mcc']:.4f}\n")
        f.write(f"Dice Similarity Coeff: {results['dsc']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"F1 Score:          {results['f1_score']:.4f}\n")
        f.write(f"Precision:         {results['precision']:.4f}\n")
        f.write(f"Recall:            {results['recall']:.4f}\n")
        f.write(f"Specificity:       {results['specificity']:.4f}\n")
        f.write(f"Accuracy:          {results['accuracy']:.4f}\n")
        f.write(f"Threshold Used:    {results['optimal_threshold']:.6f}\n")
        f.write("="*60 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  Patch size: {config.patch_size}\n")
        f.write(f"  Patches per volume: {config.patches_per_volume}\n")
        f.write(f"  K neighbors: {config.k_neighbors}\n")
        f.write(f"  Feature dimension: {features.shape[1]}\n")
        f.write(f"  Training samples: {len(X_train_normal)}\n")
        f.write(f"  Validation samples: {len(X_val_normal)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Training subjects: {len(train_subjects)}\n")
        f.write(f"  Validation subjects: {len(val_subjects)}\n")
        f.write(f"  Test subjects: {len(test_subjects)}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")
        f.write("="*60 + "\n")
        f.write(f"EXECUTION TIME:\n")
        f.write(f"  Total time: {time_formatted} ({total_time:.1f} seconds)\n")
    
    if config.verbose:
        print(f"\nResults saved to: {results_file}")
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("✓ Anatomix features extracted from quality-controlled patches")
        print("✓ KNN-based anomaly detection with FAISS acceleration")
        print("✓ Data leakage eliminated through subject-level splitting")
        print("✓ Truly unsupervised threshold determination")
        print("="*60)
        print(f"⏱️  TOTAL EXECUTION TIME: {time_formatted} ({total_time:.1f} seconds)")
        print("="*60)
    else:
        # Minimal completion message
        print(f"\nPipeline completed in {time_formatted}. Results saved to: {results_file}")

if __name__ == "__main__":
    main() 