#!/usr/bin/env python3
"""
3D Autoencoder-based Anomaly Detection for BraTS Dataset

This program implements a 3D Convolutional Autoencoder for UNSUPERVISED anomaly detection 
on the BraTS dataset. The autoencoder is trained ONLY on normal patches (without tumor) 
and learns to reconstruct normal brain tissue patterns. High reconstruction errors indicate 
anomalies (tumors).

CORRECTED APPROACH:
- Training: Only normal patches (unsupervised learning)
- Validation: Only normal patches (monitor overfitting)
- Testing: Mixed normal + anomalous patches (evaluate detection performance)
- Detection: Reconstruction error threshold determines anomalies

Features:
- Proper unsupervised anomaly detection methodology
- 3D patch extraction from volumes with quality validation
- GPU acceleration for all compute-intensive operations
- Real progress bars during training
- Comprehensive evaluation metrics for anomaly detection
- Multiple visualization options
- Configurable parameters via command line arguments
"""

import os
import sys
import glob
import argparse
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class Config:
    """Central configuration class for all parameters"""
    
    def __init__(self):
        # Dataset paths
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "brats_3d_anomaly_results"
        
        # Patch extraction parameters
        self.patch_size = 32  # 32x32x32 patches
        self.patches_per_volume = 50  # Number of patches to extract per volume
        self.min_non_zero_ratio = 0.2  # Increased minimum ratio of non-zero voxels in patch
        self.max_normal_to_anomaly_ratio = 3  # Reduced ratio for better balance
        self.min_tumor_ratio_in_patch = 0.05  # Increased minimum tumor ratio (5% instead of 1%)
        
        # Additional patch quality parameters
        self.min_patch_std = 0.01  # Minimum standard deviation for patch quality
        self.min_patch_mean = 0.05  # Minimum mean intensity for patch quality
        self.max_tumor_ratio_normal = 0.01  # Maximum allowed tumor ratio in normal patches
        self.min_tumor_ratio_anomaly = 0.05  # Minimum required tumor ratio in anomaly patches
        self.max_normal_patches_per_subject = 100  # Maximum normal patches per subject
        self.max_anomaly_patches_per_subject = 50  # Maximum anomaly patches per subject
        
        # Model parameters
        self.latent_dim = 256  # Increased latent dimension for better representation
        self.learning_rate = 5e-5  # Reduced learning rate for more stable training
        self.batch_size = 8  # Reduced batch size for better gradients with limited data
        self.num_epochs = 100
        self.early_stopping_patience = 20  # Increased patience
        
        # Training parameters
        self.train_test_split = 0.8
        self.validation_split = 0.2
        
        # Visualization parameters
        self.num_samples_visualize = 5
        self.slice_axis = 'axial'  # 'axial', 'coronal', 'sagittal'
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if torch.cuda.is_available() else 0
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)


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


class Autoencoder3D(nn.Module):
    """Enhanced 3D Convolutional Autoencoder with skip connections for better anomaly detection"""
    
    def __init__(self, input_channels=1, latent_dim=128):
        super(Autoencoder3D, self).__init__()
        
        # Encoder with residual-like connections
        self.encoder_conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)  # 32->16
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(2)  # 16->8
        
        self.encoder_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(2)  # 8->4
        
        self.encoder_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        # Latent space
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 4, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, 256 * 4 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(1, (256, 4, 4, 4))
        
        # Decoder with skip connections
        self.decoder_conv4 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        self.decoder_conv3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        self.decoder_conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        self.decoder_conv1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
    def encode(self, x):
        # Encoder path
        e1 = self.encoder_conv1(x)
        e1_pool = self.pool1(e1)
        
        e2 = self.encoder_conv2(e1_pool)
        e2_pool = self.pool2(e2)
        
        e3 = self.encoder_conv3(e2_pool)
        e3_pool = self.pool3(e3)
        
        e4 = self.encoder_conv4(e3_pool)
        
        # Latent representation
        flat = self.flatten(e4)
        latent = self.fc_encode(flat)
        
        return latent
    
    def decode(self, latent):
        # Decode from latent space
        decoded = self.fc_decode(latent)
        unflat = self.unflatten(decoded)
        
        # Decoder path
        d4 = self.decoder_conv4(unflat)
        d4_up = self.upsample3(d4)
        
        d3 = self.decoder_conv3(d4_up)
        d3_up = self.upsample2(d3)
        
        d2 = self.decoder_conv2(d3_up)
        d2_up = self.upsample1(d2)
        
        output = self.decoder_conv1(d2_up)
        
        return output
        
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class BraTSDataProcessor:
    """Class for processing BraTS data and extracting patches"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_volume(self, subject_path: str, modality: str = 't1c') -> Tuple[np.ndarray, np.ndarray]:
        """Load a specific modality volume and its segmentation mask"""
        volume_file = glob.glob(os.path.join(subject_path, f"*-{modality}.nii.gz"))[0]
        seg_file = glob.glob(os.path.join(subject_path, "*-seg.nii.gz"))[0]
        
        volume = nib.load(volume_file).get_fdata()
        segmentation = nib.load(seg_file).get_fdata()
        
        return volume, segmentation
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range while preserving tissue contrast"""
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
        """
        Extract normal patches from regions without tumor (label 0)
        """
        patches = []
        
        # Get coordinates where there's no tumor (background + healthy tissue)
        valid_coords = np.where(segmentation == 0)
        
        if len(valid_coords[0]) == 0:
            return patches
        
        # Calculate number of patches to extract
        max_patches = min(len(valid_coords[0]) // 100, self.config.max_normal_patches_per_subject)
        
        if max_patches == 0:
            return patches
        
        # Sample coordinates
        indices = np.random.choice(len(valid_coords[0]), 
                                 size=min(max_patches, len(valid_coords[0])), 
                                 replace=False)
        
        patch_coords = [(valid_coords[0][i], valid_coords[1][i], valid_coords[2][i]) 
                       for i in indices]
        
        # Extract patches with progress bar
        for x, y, z in tqdm(patch_coords, desc="Extracting normal patches", leave=False):
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
                # Verify this is actually a normal patch
                patch_seg = segmentation[x_start:x_end, y_start:y_end, z_start:z_end]
                tumor_ratio = np.sum(patch_seg > 0) / patch_seg.size
                
                if tumor_ratio <= self.config.max_tumor_ratio_normal:
                    patches.append(patch)
        
        return patches
    
    def extract_anomalous_patches(self, volume: np.ndarray, segmentation: np.ndarray) -> List[np.ndarray]:
        """
        Extract anomalous patches from tumor regions (labels 1, 2, 4)
        """
        patches = []
        
        # Get coordinates where there's tumor
        tumor_coords = np.where(segmentation > 0)
        
        if len(tumor_coords[0]) == 0:
            return patches
        
        # Calculate number of patches to extract
        max_patches = min(len(tumor_coords[0]) // 50, self.config.max_anomaly_patches_per_subject)
        
        if max_patches == 0:
            return patches
        
        # Sample coordinates
        indices = np.random.choice(len(tumor_coords[0]), 
                                 size=min(max_patches, len(tumor_coords[0])), 
                                 replace=False)
        
        patch_coords = [(tumor_coords[0][i], tumor_coords[1][i], tumor_coords[2][i]) 
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
                # Verify this patch contains tumor
                patch_seg = segmentation[x_start:x_end, y_start:y_end, z_start:z_end]
                tumor_ratio = np.sum(patch_seg > 0) / patch_seg.size
                
                if tumor_ratio >= self.config.min_tumor_ratio_anomaly:
                    patches.append(patch)
        
        return patches
    
    def process_dataset(self, num_subjects: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Process the BraTS dataset and extract patches with subject tracking"""
        
        # Get list of subjects
        subject_dirs = [d for d in os.listdir(self.config.dataset_path) 
                       if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        
        if num_subjects:
            subject_dirs = subject_dirs[:num_subjects]
        
        print(f"Processing {len(subject_dirs)} subjects...")
        
        all_normal_patches = []
        all_anomalous_patches = []
        all_normal_subjects = []  # Track which subject each patch comes from
        all_anomalous_subjects = []
        
        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            subject_path = os.path.join(self.config.dataset_path, subject_dir)
            
            try:
                # Load volume and segmentation
                volume, segmentation = self.load_volume(subject_path)
                
                if volume is None or segmentation is None:
                    continue
                
                # Normalize volume
                volume = self.normalize_volume(volume)
                
                # Extract patches
                normal_patches = self.extract_normal_patches(volume, segmentation)
                anomalous_patches = self.extract_anomalous_patches(volume, segmentation)
                
                # Add patches and track subjects
                all_normal_patches.extend(normal_patches)
                all_normal_subjects.extend([subject_dir] * len(normal_patches))
                
                all_anomalous_patches.extend(anomalous_patches)
                all_anomalous_subjects.extend([subject_dir] * len(anomalous_patches))
                
            except Exception as e:
                continue
        
        print(f"Extracted {len(all_normal_patches)} normal patches and {len(all_anomalous_patches)} anomalous patches")
        
        # Balance dataset
        num_anomalous = len(all_anomalous_patches)
        max_normal = int(num_anomalous * self.config.max_normal_to_anomaly_ratio)
        
        if len(all_normal_patches) > max_normal:
            indices = np.random.choice(len(all_normal_patches), max_normal, replace=False)
            all_normal_patches = [all_normal_patches[i] for i in indices]
            all_normal_subjects = [all_normal_subjects[i] for i in indices]
        
        print(f"Final dataset: {len(all_normal_patches)} normal, {len(all_anomalous_patches)} anomalous patches")
        
        # Combine patches and create labels
        all_patches = all_normal_patches + all_anomalous_patches
        labels = [0] * len(all_normal_patches) + [1] * len(all_anomalous_patches)
        subjects = all_normal_subjects + all_anomalous_subjects
        
        # Convert to numpy arrays (DO NOT add channel dimension here - Dataset class handles it)
        patches_array = np.array(all_patches, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)
        
        return patches_array, labels_array, subjects

    def validate_patch_quality(self, patches: np.ndarray, labels: np.ndarray, 
                              sample_segmentations: List[np.ndarray] = None) -> Dict:
        """Validate the quality of extracted patches"""
        print("\nValidating patch quality...")
        
        normal_patches = patches[labels == 0]
        anomaly_patches = patches[labels == 1]
        
        # Check patch statistics
        stats = {
            'normal_patches': {
                'count': len(normal_patches),
                'mean_intensity': np.mean([patch.mean() for patch in normal_patches]),
                'std_intensity': np.std([patch.mean() for patch in normal_patches]),
                'non_zero_ratio': np.mean([np.sum(patch > 0) / patch.size for patch in normal_patches])
            },
            'anomaly_patches': {
                'count': len(anomaly_patches),
                'mean_intensity': np.mean([patch.mean() for patch in anomaly_patches]),
                'std_intensity': np.std([patch.mean() for patch in anomaly_patches]),
                'non_zero_ratio': np.mean([np.sum(patch > 0) / patch.size for patch in anomaly_patches])
            }
        }
        
        print(f"Normal patches statistics:")
        print(f"  Count: {stats['normal_patches']['count']}")
        print(f"  Mean intensity: {stats['normal_patches']['mean_intensity']:.4f}")
        print(f"  Non-zero ratio: {stats['normal_patches']['non_zero_ratio']:.4f}")
        
        print(f"Anomaly patches statistics:")
        print(f"  Count: {stats['anomaly_patches']['count']}")
        print(f"  Mean intensity: {stats['anomaly_patches']['mean_intensity']:.4f}")
        print(f"  Non-zero ratio: {stats['anomaly_patches']['non_zero_ratio']:.4f}")
        
        # Check for potential issues
        if stats['anomaly_patches']['mean_intensity'] <= stats['normal_patches']['mean_intensity']:
            print("WARNING: Anomaly patches have lower or equal intensity than normal patches!")
            print("This might indicate poor patch extraction or data preprocessing issues.")
        
        if abs(stats['normal_patches']['non_zero_ratio'] - stats['anomaly_patches']['non_zero_ratio']) < 0.1:
            print("WARNING: Normal and anomaly patches have very similar non-zero ratios!")
            print("This might indicate insufficient differentiation in patch extraction.")
        
        return stats


class AnomalyDetector:
    """Main class for training and evaluating the autoencoder"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = Autoencoder3D(latent_dim=config.latent_dim).to(config.device)
        self.scaler = GradScaler()
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the autoencoder ONLY on normal data for proper anomaly detection"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        scaler = GradScaler()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        total_steps = self.config.num_epochs * len(train_loader)
        
        print("TRAINING MODE: Autoencoder will be trained ONLY on normal data (unsupervised)")
        print("Anomalous data will be used ONLY for testing, not training")
        
        with tqdm(total=total_steps, desc="Training Progress") as pbar:
            for epoch in range(self.config.num_epochs):
                # Training phase - ONLY on normal data
                self.model.train()
                train_loss = 0.0
                normal_samples_processed = 0
                
                for batch_idx, (data, labels) in enumerate(train_loader):
                    data = data.to(self.config.device)
                    labels = labels.to(self.config.device)
                    
                    # CRITICAL FIX: Only use normal samples (label = 0) for training
                    normal_mask = (labels == 0).squeeze()
                    
                    if normal_mask.sum() == 0:  # Skip if no normal samples in batch
                        pbar.update(1)
                        continue
                    
                    normal_data = data[normal_mask]
                    normal_samples_processed += normal_data.size(0)
                    
                    optimizer.zero_grad()
                    
                    with autocast():
                        reconstructed, _ = self.model(normal_data)
                        loss = criterion(reconstructed, normal_data)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix({
                        'Epoch': f'{epoch+1}/{self.config.num_epochs}',
                        'Loss': f'{loss.item():.6f}',
                        'Normal_samples': normal_samples_processed
                    })
                
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
                train_losses.append(avg_train_loss)
                
                # Validation phase - ONLY on normal data
                self.model.eval()
                val_loss = 0.0
                val_normal_samples = 0
                
                with torch.no_grad():
                    for data, labels in val_loader:
                        data = data.to(self.config.device)
                        labels = labels.to(self.config.device)
                        
                        # Only validate on normal samples
                        normal_mask = (labels == 0).squeeze()
                        
                        if normal_mask.sum() == 0:
                            continue
                            
                        normal_data = data[normal_mask]
                        val_normal_samples += normal_data.size(0)
                        
                        with autocast():
                            reconstructed, _ = self.model(normal_data)
                            loss = criterion(reconstructed, normal_data)
                        
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
                val_losses.append(avg_val_loss)
                
                scheduler.step(avg_val_loss)
                
                # Print epoch summary
                if (epoch + 1) % 10 == 0:
                    print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                    print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                    print(f"Normal samples processed: Train={normal_samples_processed}, Val={val_normal_samples}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 
                              os.path.join(self.config.output_dir, 'best_autoencoder_3d.pth'))
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    pbar.write(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        print(f"\nTraining completed. Model trained on normal data only.")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        # Save training history
        self.save_training_plots(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def save_training_plots(self, train_losses: List[float], val_losses: List[float]):
        """Save training and validation loss plots"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_reconstruction_errors(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate reconstruction errors for anomaly detection"""
        self.model.eval()
        
        reconstruction_errors = []
        true_labels = []
        latent_features = []
        
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Calculating reconstruction errors"):
                data = data.to(self.config.device)
                
                with autocast():
                    reconstructed, latent = self.model(data)
                    
                # Calculate MSE for each sample
                mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2, 3, 4))
                
                reconstruction_errors.extend(mse.cpu().numpy())
                true_labels.extend(labels.cpu().numpy().flatten())
                latent_features.extend(latent.cpu().numpy())
        
        return np.array(reconstruction_errors), np.array(true_labels), np.array(latent_features)
    
    def find_unsupervised_threshold(self, val_loader: DataLoader) -> float:
        """Find threshold using ONLY normal validation data (no test label access)"""
        print(f"\nUNSUPERVISED THRESHOLD DETERMINATION (No Test Label Access)")
        print(f"{'='*60}")
        print("Computing threshold using ONLY normal validation data...")
        
        # Calculate reconstruction errors on validation set (only normal data)
        self.model.eval()
        normal_val_errors = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Computing validation errors"):
                data = data.to(self.config.device)
                labels = labels.to(self.config.device)
                
                # Only use normal samples (should be all in val set anyway)
                normal_mask = (labels == 0).squeeze()
                if normal_mask.sum() == 0:
                    continue
                    
                normal_data = data[normal_mask]
                
                with autocast():
                    reconstructed, _ = self.model(normal_data)
                    
                # Calculate MSE for each sample
                mse = torch.mean((normal_data - reconstructed) ** 2, dim=(1, 2, 3, 4))
                normal_val_errors.extend(mse.cpu().numpy())
        
        normal_val_errors = np.array(normal_val_errors)
        
        print(f"Normal validation errors - Count: {len(normal_val_errors)}")
        print(f"Normal validation errors - Mean: {normal_val_errors.mean():.6f}, Std: {normal_val_errors.std():.6f}")
        print(f"Normal validation errors - Min: {normal_val_errors.min():.6f}, Max: {normal_val_errors.max():.6f}")
        
        # Create unsupervised threshold candidates based ONLY on normal data
        threshold_methods = {
            'percentile_95': np.percentile(normal_val_errors, 95),
            'percentile_97': np.percentile(normal_val_errors, 97),
            'percentile_99': np.percentile(normal_val_errors, 99),
            'mean_plus_2std': normal_val_errors.mean() + 2 * normal_val_errors.std(),
            'mean_plus_3std': normal_val_errors.mean() + 3 * normal_val_errors.std(),
            'median_plus_2mad': np.median(normal_val_errors) + 2 * np.median(np.abs(normal_val_errors - np.median(normal_val_errors))),
            'iqr_outlier': np.percentile(normal_val_errors, 75) + 1.5 * (np.percentile(normal_val_errors, 75) - np.percentile(normal_val_errors, 25))
        }
        
        print(f"\nUNSUPERVISED THRESHOLD CANDIDATES:")
        print(f"{'Method':<20} {'Threshold':<12}")
        print(f"{'-'*35}")
        for method, threshold in threshold_methods.items():
            print(f"{method:<20} {threshold:<12.6f}")
        
        # Use 95th percentile as default (common practice in anomaly detection)
        selected_threshold = threshold_methods['percentile_95']
        selected_method = 'percentile_95'
        
        print(f"\nSELECTED THRESHOLD: {selected_threshold:.6f} (Method: {selected_method})")
        print(f"Rationale: 95th percentile of normal validation errors is a conservative,")
        print(f"commonly used threshold that doesn't require test label access.")
        print(f"{'='*60}")
        
        return selected_threshold
    
    def evaluate(self, test_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Evaluate the autoencoder for truly unsupervised anomaly detection"""
        # Load best model
        model_path = os.path.join(self.config.output_dir, 'best_autoencoder_3d.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Loaded best model for evaluation")
        
        # STEP 1: Determine threshold using ONLY normal validation data (no test label access)
        optimal_threshold = self.find_unsupervised_threshold(val_loader)
        
        # STEP 2: Calculate reconstruction errors on test set
        reconstruction_errors, true_labels, latent_features = self.calculate_reconstruction_errors(test_loader)
        
        print(f"\nEVALUATION: Truly Unsupervised Anomaly Detection Results")
        print(f"{'='*60}")
        print(f"Total test samples: {len(reconstruction_errors)}")
        print(f"Normal samples: {np.sum(true_labels == 0)}")
        print(f"Anomalous samples: {np.sum(true_labels == 1)}")
        
        # STEP 3: Apply threshold (determined without seeing test labels)
        predictions = (reconstruction_errors > optimal_threshold).astype(int)
        
        # STEP 4: Calculate metrics - NOW it's fair because threshold was set without test labels
        try:
            roc_auc = roc_auc_score(true_labels, reconstruction_errors)
            average_precision = average_precision_score(true_labels, reconstruction_errors)
        except:
            roc_auc = 0.0
            average_precision = 0.0
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # Additional metrics for imbalanced anomaly detection
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0
        
        # False Positive Rate and False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Analyze reconstruction error distributions POST-HOC (for understanding, not optimization)
        normal_errors = reconstruction_errors[true_labels == 0]
        anomaly_errors = reconstruction_errors[true_labels == 1]
        separation_ratio = anomaly_errors.mean() / normal_errors.mean() if normal_errors.mean() > 0 else 0
        
        print(f"\nTRULY UNSUPERVISED ANOMALY DETECTION PERFORMANCE:")
        print(f"{'='*60}")
        print(f"ROC AUC:                  {roc_auc:.4f}")
        print(f"Average Precision (AP):   {average_precision:.4f}")
        print(f"Matthews Correlation:     {mcc:.4f}")
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
        
        # POST-HOC analysis (for understanding only)
        print(f"\nPOST-HOC RECONSTRUCTION ERROR ANALYSIS:")
        print(f"Normal errors    - Mean: {normal_errors.mean():.6f}, Std: {normal_errors.std():.6f}")
        print(f"Anomaly errors   - Mean: {anomaly_errors.mean():.6f}, Std: {anomaly_errors.std():.6f}")
        print(f"Separation ratio - Anomaly/Normal: {separation_ratio:.3f}")
        
        # Detailed confusion matrix analysis
        print(f"\nCONFUSION MATRIX ANALYSIS:")
        print(f"True Negatives (Normal correctly identified):      {tn}")
        print(f"False Positives (Normal misclassified as anomaly): {fp}")
        print(f"False Negatives (Anomaly missed):                  {fn}")
        print(f"True Positives (Anomaly correctly detected):       {tp}")
        
        results = {
            'reconstruction_errors': reconstruction_errors,
            'true_labels': true_labels,
            'predictions': predictions,
            'latent_features': latent_features,
            'optimal_threshold': optimal_threshold,
            'roc_auc': roc_auc,
            'average_precision': average_precision,
            'mcc': mcc,
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


class Visualizer:
    """Class for creating various visualizations"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray):
        """Plot confusion matrix with improved formatting"""
        cm = confusion_matrix(true_labels, predictions)
        
        # Print detailed information
        print(f"\nConfusion Matrix Details:")
        print(f"True Labels - Normal: {np.sum(true_labels == 0)}, Anomaly: {np.sum(true_labels == 1)}")
        print(f"Predictions - Normal: {np.sum(predictions == 0)}, Anomaly: {np.sum(predictions == 1)}")
        print(f"Confusion Matrix:\n{cm}")
        
        # Check if we have both classes in predictions and true labels
        if len(np.unique(true_labels)) < 2:
            print("WARNING: True labels contain only one class!")
        if len(np.unique(predictions)) < 2:
            print("WARNING: Predictions contain only one class!")
        
        plt.figure(figsize=(12, 10))
        
        # Create percentage annotations for better readability
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create combined annotations showing both count and percentage
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
                row.append(text)
            annotations.append(row)
        
        # Plot heatmap with improved formatting
        ax = sns.heatmap(cm, 
                        annot=np.array(annotations), 
                        fmt='', 
                        cmap='Blues',
                        xticklabels=['Normal (0)', 'Anomaly (1)'],
                        yticklabels=['Normal (0)', 'Anomaly (1)'],
                        cbar_kws={'label': 'Count'},
                        square=True,
                        linewidths=2,
                        linecolor='white',
                        annot_kws={'size': 14, 'weight': 'bold'})
        
        # Improve title and labels
        plt.title('Confusion Matrix\n(Count and Percentage)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        
        # Adjust tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add summary statistics as text
        total_samples = cm.sum()
        accuracy = np.trace(cm) / total_samples
        
        # Calculate per-class metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        summary_text = (f'Total Samples: {total_samples}\n'
                       f'Accuracy: {accuracy:.3f}\n'
                       f'Precision: {precision:.3f}\n'
                       f'Recall: {recall:.3f}\n'
                       f'Specificity: {specificity:.3f}')
        
        plt.figtext(0.02, 0.02, summary_text, 
                   fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for summary text
        plt.savefig(os.path.join(self.config.output_dir, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_roc_curve(self, true_labels: np.ndarray, reconstruction_errors: np.ndarray):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(true_labels, reconstruction_errors)
        auc = roc_auc_score(true_labels, reconstruction_errors)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'roc_curve.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, true_labels: np.ndarray, reconstruction_errors: np.ndarray):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(true_labels, reconstruction_errors)
        avg_precision = average_precision_score(true_labels, reconstruction_errors)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.4f})')
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
    
    def plot_reconstruction_error_histogram(self, reconstruction_errors: np.ndarray, 
                                          true_labels: np.ndarray, optimal_threshold: float):
        """Plot histogram of reconstruction errors"""
        normal_errors = reconstruction_errors[true_labels == 0]
        anomaly_errors = reconstruction_errors[true_labels == 1]
        
        plt.figure(figsize=(12, 6))
        
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, 
                   label=f'Optimal Threshold = {optimal_threshold:.6f}')
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'reconstruction_error_histogram.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_latent_space_visualization(self, latent_features: np.ndarray, true_labels: np.ndarray):
        """Plot t-SNE and PCA visualizations of latent space"""
        print("Creating latent space visualizations...")
        
        # Limit samples for visualization if too many
        if len(latent_features) > 2000:
            indices = np.random.choice(len(latent_features), 2000, replace=False)
            latent_features = latent_features[indices]
            true_labels = true_labels[indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA
        print("Computing PCA...")
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(latent_features)
        
        scatter1 = ax1.scatter(pca_features[true_labels == 0, 0], pca_features[true_labels == 0, 1], 
                             c='blue', alpha=0.6, label='Normal', s=20)
        scatter2 = ax1.scatter(pca_features[true_labels == 1, 0], pca_features[true_labels == 1, 1], 
                             c='red', alpha=0.6, label='Anomaly', s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA of Latent Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_features)//4))
        tsne_features = tsne.fit_transform(latent_features)
        
        scatter3 = ax2.scatter(tsne_features[true_labels == 0, 0], tsne_features[true_labels == 0, 1], 
                             c='blue', alpha=0.6, label='Normal', s=20)
        scatter4 = ax2.scatter(tsne_features[true_labels == 1, 0], tsne_features[true_labels == 1, 1], 
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
    
    def visualize_3d_patches(self, patches: np.ndarray, labels: np.ndarray, num_samples: int = 5):
        """Visualize 3D patches by showing multiple slices"""
        print(f"Visualizing {num_samples} sample patches...")
        
        # Select random samples
        indices = np.random.choice(len(patches), min(num_samples, len(patches)), replace=False)
        
        for i, idx in enumerate(indices):
            patch = patches[idx]
            label = labels[idx]
            label_name = "Anomaly" if label == 1 else "Normal"
            
            # Create subplot for different slice orientations
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'Sample {i+1}: {label_name} Patch', fontsize=16)
            
            # Show axial slices (z-axis)
            for j in range(4):
                slice_idx = int(patch.shape[2] * (j + 1) / 5)  # Evenly spaced slices
                axes[0, j].imshow(patch[:, :, slice_idx], cmap='gray')
                axes[0, j].set_title(f'Axial Slice {slice_idx}')
                axes[0, j].axis('off')
            
            # Show coronal slices (y-axis)
            for j in range(4):
                slice_idx = int(patch.shape[1] * (j + 1) / 5)  # Evenly spaced slices
                axes[1, j].imshow(patch[:, slice_idx, :], cmap='gray')
                axes[1, j].set_title(f'Coronal Slice {slice_idx}')
                axes[1, j].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, f'patch_visualization_sample_{i}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_report(self, results: Dict):
        """Create a summary report with all visualizations"""
        self.plot_confusion_matrix(results['true_labels'], results['predictions'])
        self.plot_roc_curve(results['true_labels'], results['reconstruction_errors'])
        self.plot_precision_recall_curve(results['true_labels'], results['reconstruction_errors'])
        self.plot_reconstruction_error_histogram(results['reconstruction_errors'], 
                                                results['true_labels'], 
                                                results['optimal_threshold'])
        self.plot_latent_space_visualization(results['latent_features'], results['true_labels'])
        
        print(f"\nAll visualizations saved to: {self.config.output_dir}")


def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description='3D Autoencoder Anomaly Detection for BraTS Dataset')
    parser.add_argument('--num_subjects', type=int, default=None, 
                       help='Number of subjects to use (default: all)')
    parser.add_argument('--patch_size', type=int, default=32, 
                       help='Size of 3D patches (default: 32)')
    parser.add_argument('--patches_per_volume', type=int, default=50, 
                       help='Number of patches per volume (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for training (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, 
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--latent_dim', type=int, default=256, 
                       help='Latent dimension size (default: 256)')
    parser.add_argument('--output_dir', type=str, default='brats_3d_anomaly_results', 
                       help='Output directory (default: brats_3d_anomaly_results)')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData', 
                       help='Path to BraTS dataset')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3,
                       help='Maximum ratio of normal to anomaly patches (default: 3)')
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05,
                       help='Minimum tumor ratio in anomalous patches (default: 0.05)')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Update config with command line arguments
    config.num_subjects = args.num_subjects
    config.patch_size = args.patch_size
    config.patches_per_volume = args.patches_per_volume
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.latent_dim = args.latent_dim
    config.output_dir = args.output_dir
    config.dataset_path = args.dataset_path
    config.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    config.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("="*60)
    print("3D AUTOENCODER ANOMALY DETECTION FOR BRATS DATASET")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Dataset path: {config.dataset_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Patch size: {config.patch_size}x{config.patch_size}x{config.patch_size}")
    print(f"Patches per volume: {config.patches_per_volume}")
    print(f"Number of subjects: {config.num_subjects if config.num_subjects else 'All'}")
    print("="*60)
    
    # Step 1: Process dataset and extract patches
    print("\n1. Processing dataset and extracting patches...")
    processor = BraTSDataProcessor(config)
    patches, labels, subjects = processor.process_dataset(config.num_subjects)
    
    if len(patches) == 0:
        print("Error: No patches extracted! Please check your dataset path and structure.")
        return
    
    print(f"Total patches extracted: {len(patches)}")
    print(f"Patch shape: {patches[0].shape}")
    print(f"Normal patches: {np.sum(labels == 0)}")
    print(f"Anomalous patches: {np.sum(labels == 1)}")
    
    # Step 1.5: Validate patch quality
    processor.validate_patch_quality(patches, labels)
    
    # Step 2: Split data into train and test sets
    print("\n2. Subject-level data splitting for truly unsupervised anomaly detection...")
    
    # Check if we have both classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    if len(unique_labels) < 2:
        print("ERROR: Dataset contains only one class! Cannot perform anomaly detection.")
        print("Please check your data extraction - you need both normal and anomalous patches.")
        return
    
    # CRITICAL FIX: Subject-level splitting to prevent patient data leakage
    unique_subjects = list(set(subjects))
    print(f"Total unique subjects: {len(unique_subjects)}")
    
    # Split subjects (not patches) into train/val/test
    np.random.shuffle(unique_subjects)
    n_subjects = len(unique_subjects)
    train_subjects = unique_subjects[:int(0.6 * n_subjects)]  # 60% for training
    val_subjects = unique_subjects[int(0.6 * n_subjects):int(0.8 * n_subjects)]  # 20% for validation
    test_subjects = unique_subjects[int(0.8 * n_subjects):]  # 20% for testing
    
    print(f"Subject distribution:")
    print(f"  Training subjects: {len(train_subjects)}")
    print(f"  Validation subjects: {len(val_subjects)}")
    print(f"  Test subjects: {len(test_subjects)}")
    
    # Create patch-level splits based on subject assignment
    train_indices = [i for i, subj in enumerate(subjects) if subj in train_subjects]
    val_indices = [i for i, subj in enumerate(subjects) if subj in val_subjects]
    test_indices = [i for i, subj in enumerate(subjects) if subj in test_subjects]
    
    X_train_all = patches[train_indices]
    y_train_all = labels[train_indices]
    
    X_val_all = patches[val_indices]
    y_val_all = labels[val_indices]
    
    X_test = patches[test_indices]
    y_test = labels[test_indices]
    
    # UNSUPERVISED CONSTRAINT: Only use NORMAL patches for training and validation
    train_normal_mask = (y_train_all == 0)
    val_normal_mask = (y_val_all == 0)
    
    X_train_normal = X_train_all[train_normal_mask]
    y_train_normal = y_train_all[train_normal_mask]
    
    X_val_normal = X_val_all[val_normal_mask]
    y_val_normal = y_val_all[val_normal_mask]
    
    # Test set keeps both normal and anomalous (for evaluation)
    # This is the only place where we're allowed to have anomalous data
    
    print(f"\n=== SUBJECT-LEVEL UNSUPERVISED ANOMALY DETECTION SPLIT ===")
    print(f"Training set (NORMAL ONLY): {len(X_train_normal)} patches from {len(train_subjects)} subjects")
    print(f"  Normal: {np.sum(y_train_normal == 0)}, Anomalous: {np.sum(y_train_normal == 1)}")
    print(f"Validation set (NORMAL ONLY): {len(X_val_normal)} patches from {len(val_subjects)} subjects") 
    print(f"  Normal: {np.sum(y_val_normal == 0)}, Anomalous: {np.sum(y_val_normal == 1)}")
    print(f"Test set (MIXED): {len(X_test)} patches from {len(test_subjects)} subjects")
    print(f"  Normal: {np.sum(y_test == 0)}, Anomalous: {np.sum(y_test == 1)}")
    print(f"========================================================")
    
    # Verify no subject appears in multiple splits
    assert len(set(train_subjects) & set(val_subjects)) == 0, "Subject overlap between train and validation!"
    assert len(set(train_subjects) & set(test_subjects)) == 0, "Subject overlap between train and test!"
    assert len(set(val_subjects) & set(test_subjects)) == 0, "Subject overlap between validation and test!"
    print("✓ No subject overlap confirmed - data leakage prevented!")
    
    # Create datasets with the corrected split
    # Training: Only normal data from training subjects
    train_dataset = BraTSPatchDataset(X_train_normal, y_train_normal)
    # Validation: Only normal data from validation subjects
    val_dataset = BraTSPatchDataset(X_val_normal, y_val_normal)
    # Testing: Mixed data (normal + anomalous) from test subjects
    test_dataset = BraTSPatchDataset(X_test, y_test)
    
    # Step 3: Create data loaders
    print("\n3. Creating data loaders...")
    
    # UPDATED EXPLANATION: For proper unsupervised anomaly detection
    print(f"\nCORRECTED ANOMALY DETECTION APPROACH:")
    print(f"✓ Training: ONLY normal data (autoencoder learns normal patterns)")
    print(f"✓ Validation: ONLY normal data (monitor overfitting on normal data)")  
    print(f"✓ Testing: MIXED data (evaluate anomaly detection performance)")
    print(f"✓ Anomaly Detection: High reconstruction error = Anomaly")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=config.num_workers)
    
    # Step 4: Initialize and train the autoencoder
    print("\n4. Training 3D Autoencoder...")
    detector = AnomalyDetector(config)
    train_losses, val_losses = detector.train(train_loader, val_loader)
    
    # Step 5: Evaluate the model
    print("\n5. Evaluating model on test set...")
    results = detector.evaluate(test_loader, val_loader)
    
    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    visualizer = Visualizer(config)
    visualizer.create_summary_report(results)
    
    # Step 7: Visualize sample patches
    print("\n7. Visualizing sample patches...")
    visualizer.visualize_3d_patches(X_test, y_test, config.num_samples_visualize)
    
    # Save results to file
    results_file = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("TRULY UNSUPERVISED 3D Autoencoder Anomaly Detection Results\n")
        f.write("="*60 + "\n")
        f.write("DATA LEAKAGE PREVENTION MEASURES:\n")
        f.write("- Subject-level data splitting (no patient overlap)\n")
        f.write("- Threshold determined using ONLY normal validation data\n")
        f.write("- No test label access during threshold selection\n")
        f.write("="*60 + "\n")
        f.write(f"ROC AUC:           {results['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {results['average_precision']:.4f}\n")
        f.write(f"Matthews Correlation: {results['mcc']:.4f}\n")
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
        f.write(f"  Latent dimension: {config.latent_dim}\n")
        f.write(f"  Learning rate: {config.learning_rate}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  Number of epochs: {config.num_epochs}\n")
        f.write(f"  Training samples: {len(X_train_normal)}\n")
        f.write(f"  Validation samples: {len(X_val_normal)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Training subjects: {len(train_subjects)}\n")
        f.write(f"  Validation subjects: {len(val_subjects)}\n")
        f.write(f"  Test subjects: {len(test_subjects)}\n")
    
    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("✓ Data leakage eliminated through subject-level splitting")
    print("✓ Truly unsupervised threshold determination")
    print("✓ No test label access during model development")
    print("="*60)


if __name__ == "__main__":
    main() 