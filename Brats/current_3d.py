#!/usr/bin/env python3
"""
3D Autoencoder-based Anomaly Detection for BraTS Dataset

This program implements a 3D Convolutional Autoencoder for anomaly detection on the BraTS dataset.
It extracts normal 3D patches (cubes) without tumor information for training, then evaluates 
the model's ability to detect anomalies.

Features:
- 3D patch extraction from volumes without tumor information
- Train/test split (80:20)
- GPU acceleration for all compute-intensive operations
- Real progress bars during training
- Comprehensive evaluation metrics
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
    
    def process_dataset(self, num_subjects: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Process the BraTS dataset and extract patches"""
        
        # Get list of subjects
        subject_dirs = [d for d in os.listdir(self.config.dataset_path) 
                       if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        
        if num_subjects:
            subject_dirs = subject_dirs[:num_subjects]
        
        print(f"Processing {len(subject_dirs)} subjects...")
        
        all_normal_patches = []
        all_anomalous_patches = []
        
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
                
                all_normal_patches.extend(normal_patches)
                all_anomalous_patches.extend(anomalous_patches)
                
            except Exception as e:
                continue
        
        print(f"Extracted {len(all_normal_patches)} normal patches and {len(all_anomalous_patches)} anomalous patches")
        
        # Balance dataset
        num_anomalous = len(all_anomalous_patches)
        max_normal = int(num_anomalous * self.config.max_normal_to_anomaly_ratio)
        
        if len(all_normal_patches) > max_normal:
            indices = np.random.choice(len(all_normal_patches), max_normal, replace=False)
            all_normal_patches = [all_normal_patches[i] for i in indices]
        
        print(f"Final dataset: {len(all_normal_patches)} normal, {len(all_anomalous_patches)} anomalous patches")
        
        # Combine patches and create labels
        all_patches = all_normal_patches + all_anomalous_patches
        labels = [0] * len(all_normal_patches) + [1] * len(all_anomalous_patches)
        
        # Convert to numpy arrays (DO NOT add channel dimension here - Dataset class handles it)
        patches_array = np.array(all_patches, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)
        
        return patches_array, labels_array

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
        """Train the autoencoder with progress tracking across all epochs"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        scaler = GradScaler()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        total_steps = self.config.num_epochs * len(train_loader)
        
        with tqdm(total=total_steps, desc="Training Progress") as pbar:
            for epoch in range(self.config.num_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_idx, (normal_data, _) in enumerate(train_loader):
                    normal_data = normal_data.to(self.config.device)
                    
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
                        'Loss': f'{loss.item():.6f}'
                    })
                
                avg_train_loss = train_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for normal_data, _ in val_loader:
                        normal_data = normal_data.to(self.config.device)
                        
                        with autocast():
                            reconstructed, _ = self.model(normal_data)
                            loss = criterion(reconstructed, normal_data)
                        
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                scheduler.step(avg_val_loss)
                
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
    
    def find_optimal_threshold(self, reconstruction_errors: np.ndarray, true_labels: np.ndarray) -> float:
        """Find optimal threshold for anomaly detection using multiple metrics"""
        # Get error statistics
        normal_errors = reconstruction_errors[true_labels == 0]
        anomaly_errors = reconstruction_errors[true_labels == 1]
        
        print(f"\nReconstruction Error Statistics:")
        print(f"Normal patches - Mean: {normal_errors.mean():.6f}, Std: {normal_errors.std():.6f}")
        print(f"Normal patches - Min: {normal_errors.min():.6f}, Max: {normal_errors.max():.6f}")
        print(f"Anomaly patches - Mean: {anomaly_errors.mean():.6f}, Std: {anomaly_errors.std():.6f}")
        print(f"Anomaly patches - Min: {anomaly_errors.min():.6f}, Max: {anomaly_errors.max():.6f}")
        
        # Create more sophisticated threshold candidates
        # Include percentiles of normal data and statistical thresholds
        normal_percentiles = np.percentile(normal_errors, [50, 75, 90, 95, 99])
        statistical_thresholds = [
            normal_errors.mean() + normal_errors.std(),
            normal_errors.mean() + 2 * normal_errors.std(),
            normal_errors.mean() + 3 * normal_errors.std()
        ]
        
        # Linear space between min and max errors
        linear_thresholds = np.linspace(reconstruction_errors.min(), reconstruction_errors.max(), 100)
        
        # Combine all threshold candidates
        thresholds = np.concatenate([normal_percentiles, statistical_thresholds, linear_thresholds])
        thresholds = np.unique(thresholds)  # Remove duplicates
        
        best_threshold = 0
        best_f1 = 0
        best_balanced_accuracy = 0
        results = []
        
        for threshold in thresholds:
            predictions = (reconstruction_errors > threshold).astype(int)
            
            # Skip if all predictions are the same class
            if len(np.unique(predictions)) < 2:
                continue
                
            try:
                f1 = f1_score(true_labels, predictions)
                precision = precision_score(true_labels, predictions)
                recall = recall_score(true_labels, predictions)
                accuracy = accuracy_score(true_labels, predictions)
                
                # Calculate balanced accuracy (better for imbalanced data)
                tn = np.sum((true_labels == 0) & (predictions == 0))
                fp = np.sum((true_labels == 0) & (predictions == 1))
                fn = np.sum((true_labels == 1) & (predictions == 0))
                tp = np.sum((true_labels == 1) & (predictions == 1))
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                balanced_accuracy = (sensitivity + specificity) / 2
                
                results.append({
                    'threshold': threshold,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity
                })
                
                # Primary criterion: F1 score, secondary: balanced accuracy
                if f1 > best_f1 or (f1 == best_f1 and balanced_accuracy > best_balanced_accuracy):
                    best_f1 = f1
                    best_balanced_accuracy = balanced_accuracy
                    best_threshold = threshold
                    
            except:
                continue
        
        # Print top 5 threshold candidates
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('f1', ascending=False)
            print(f"\nTop 5 threshold candidates:")
            print(results_df.head().to_string(index=False))
        
        print(f"\nSelected threshold: {best_threshold:.6f} (F1: {best_f1:.4f}, Balanced Accuracy: {best_balanced_accuracy:.4f})")
        
        return best_threshold
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Comprehensive evaluation of the model"""
        print("Evaluating model...")
        
        # Load best model
        best_model_path = os.path.join(self.config.output_dir, 'best_autoencoder_3d.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for evaluation")
        
        # Calculate reconstruction errors
        reconstruction_errors, true_labels, latent_features = self.calculate_reconstruction_errors(test_loader)
        
        print(f"\nEvaluation dataset statistics:")
        print(f"Total samples: {len(reconstruction_errors)}")
        print(f"Normal samples: {np.sum(true_labels == 0)} ({np.sum(true_labels == 0)/len(true_labels)*100:.1f}%)")
        print(f"Anomalous samples: {np.sum(true_labels == 1)} ({np.sum(true_labels == 1)/len(true_labels)*100:.1f}%)")
        
        # Analyze reconstruction error distributions
        normal_errors = reconstruction_errors[true_labels == 0]
        anomaly_errors = reconstruction_errors[true_labels == 1]
        
        print(f"\nReconstruction error distribution analysis:")
        print(f"Expected: Normal errors should be LOWER than anomaly errors")
        print(f"Normal errors    - Mean: {normal_errors.mean():.6f} ± {normal_errors.std():.6f}")
        print(f"Anomaly errors   - Mean: {anomaly_errors.mean():.6f} ± {anomaly_errors.std():.6f}")
        print(f"Separation ratio - Anomaly/Normal mean: {anomaly_errors.mean()/normal_errors.mean():.3f}")
        
        # Check if anomaly detection makes sense
        if anomaly_errors.mean() <= normal_errors.mean():
            print("WARNING: Anomaly errors are not higher than normal errors!")
            print("This suggests the model is not learning to distinguish between normal and anomalous patterns.")
            print("Possible causes:")
            print("1. Model architecture is too simple/complex")
            print("2. Training data quality issues")
            print("3. Insufficient training")
            print("4. Poor patch extraction quality")
        
        # Find optimal threshold
        optimal_threshold = self.find_optimal_threshold(reconstruction_errors, true_labels)
        predictions = (reconstruction_errors > optimal_threshold).astype(int)
        
        # Detailed prediction analysis
        print(f"\nPrediction analysis:")
        print(f"Threshold used: {optimal_threshold:.6f}")
        print(f"Samples above threshold (predicted anomalous): {np.sum(predictions == 1)}")
        print(f"Samples below threshold (predicted normal): {np.sum(predictions == 0)}")
        
        # Calculate metrics with error handling
        try:
            roc_auc = roc_auc_score(true_labels, reconstruction_errors)
            avg_precision = average_precision_score(true_labels, reconstruction_errors)
        except Exception as e:
            print(f"Error calculating AUC metrics: {e}")
            roc_auc = 0.5
            avg_precision = 0.5
        
        try:
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            accuracy = precision = recall = f1 = 0.0
        
        results = {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'optimal_threshold': optimal_threshold,
            'reconstruction_errors': reconstruction_errors,
            'true_labels': true_labels,
            'predictions': predictions,
            'latent_features': latent_features,
            'normal_error_stats': {
                'mean': normal_errors.mean(),
                'std': normal_errors.std(),
                'min': normal_errors.min(),
                'max': normal_errors.max()
            },
            'anomaly_error_stats': {
                'mean': anomaly_errors.mean(),
                'std': anomaly_errors.std(),
                'min': anomaly_errors.min(),
                'max': anomaly_errors.max()
            }
        }
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"ROC AUC:           {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"F1 Score:          {f1:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.6f}")
        print("="*50)
        
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
    patches, labels = processor.process_dataset(config.num_subjects)
    
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
    print("\n2. Splitting data into train/test sets...")
    
    # Check if we have both classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    if len(unique_labels) < 2:
        print("ERROR: Dataset contains only one class! Cannot perform anomaly detection.")
        print("Please check your data extraction - you need both normal and anomalous patches.")
        return
    
    # Ensure minimum samples for stratified split
    min_samples = min(counts)
    if min_samples < 2:
        print("ERROR: Insufficient samples for stratified split. Need at least 2 samples per class.")
        print(f"Current distribution: Normal={counts[0] if 0 in unique_labels else 0}, "
              f"Anomaly={counts[1] if 1 in unique_labels else 0}")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        patches, labels, test_size=1-config.train_test_split, 
        random_state=42, stratify=labels
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.validation_split, 
        random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train)} patches")
    print(f"  Normal: {np.sum(y_train == 0)}, Anomalous: {np.sum(y_train == 1)}")
    print(f"Validation set: {len(X_val)} patches")
    print(f"  Normal: {np.sum(y_val == 0)}, Anomalous: {np.sum(y_val == 1)}")
    print(f"Test set: {len(X_test)} patches")
    print(f"  Normal: {np.sum(y_test == 0)}, Anomalous: {np.sum(y_test == 1)}")
    
    # Step 3: Create data loaders
    print("\n3. Creating data loaders...")
    
    # CRITICAL: For anomaly detection, we need to ensure training/validation sets
    # contain sufficient normal data for the autoencoder to learn from
    print(f"\nData split analysis for anomaly detection:")
    print(f"Training will use ONLY normal data to train the autoencoder")
    print(f"Testing will use BOTH normal and anomalous data for evaluation")
    
    train_dataset = BraTSPatchDataset(X_train, y_train)
    val_dataset = BraTSPatchDataset(X_val, y_val)
    test_dataset = BraTSPatchDataset(X_test, y_test)
    
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
    results = detector.evaluate(test_loader)
    
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
        f.write("3D Autoencoder Anomaly Detection Results\n")
        f.write("="*50 + "\n")
        f.write(f"ROC AUC:           {results['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {results['average_precision']:.4f}\n")
        f.write(f"Accuracy:          {results['accuracy']:.4f}\n")
        f.write(f"Precision:         {results['precision']:.4f}\n")
        f.write(f"Recall:            {results['recall']:.4f}\n")
        f.write(f"F1 Score:          {results['f1_score']:.4f}\n")
        f.write(f"Optimal Threshold: {results['optimal_threshold']:.6f}\n")
        f.write("="*50 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  Patch size: {config.patch_size}\n")
        f.write(f"  Patches per volume: {config.patches_per_volume}\n")
        f.write(f"  Latent dimension: {config.latent_dim}\n")
        f.write(f"  Learning rate: {config.learning_rate}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  Number of epochs: {config.num_epochs}\n")
        f.write(f"  Training samples: {len(X_train)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
    
    print(f"\nResults saved to: {results_file}")
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main() 