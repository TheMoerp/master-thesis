#!/usr/bin/env python3
"""
3D Autoencoder for Anomaly Detection on BraTS Dataset
=====================================================

This program implements a 3D convolutional autoencoder for anomaly detection
on the BraTS dataset. It extracts normal patches (without tumor information)
for training and evaluates the model's ability to detect anomalies.

Features:
- 3D patch extraction from BraTS volumes
- GPU acceleration for all compute-intensive operations
- Real progress bars (no step-style)
- Comprehensive evaluation metrics
- Multiple visualization options
- Configurable parameters via command line arguments

Author: AI Assistant
"""

import argparse
import os
import sys
import logging
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import time

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ae_brats2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for all parameters"""
    
    def __init__(self, args):
        # Dataset parameters
        self.dataset_path = args.dataset_path
        self.num_subjects = args.num_subjects
        self.patch_size = args.patch_size
        self.patches_per_volume = args.patches_per_volume
        self.train_ratio = args.train_ratio
        
        # Model parameters
        self.latent_dim = args.latent_dim
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        
        # Training parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = args.num_workers
        self.seed = args.seed
        
        # Output parameters
        self.output_dir = args.output_dir
        self.save_model = args.save_model
        self.visualize = args.visualize
        
        # Anomaly detection parameters
        self.anomaly_threshold_percentile = args.anomaly_threshold_percentile


class BraTSDataset(Dataset):
    """Dataset class for BraTS 3D patches"""
    
    def __init__(self, patches: np.ndarray, labels: np.ndarray = None, transform=None):
        """
        Args:
            patches: Array of 3D patches [N, C, D, H, W]
            labels: Array of labels (0=normal, 1=anomaly)
            transform: Optional transform to be applied
        """
        self.patches = torch.FloatTensor(patches)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        if self.transform:
            patch = self.transform(patch)
            
        if self.labels is not None:
            return patch, self.labels[idx]
        return patch


class Autoencoder3D(nn.Module):
    """3D Convolutional Autoencoder"""
    
    def __init__(self, input_channels=4, latent_dim=128):
        super(Autoencoder3D, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Second conv block
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Third conv block
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Fourth conv block
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        
        # Latent space
        self.fc_encode = nn.Linear(256 * 2 * 2 * 2, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2 * 2)
        
        # Decoder
        self.decoder = nn.Sequential(
            # First deconv block
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Second deconv block
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Third deconv block
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Conv3d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        """Encode input to latent space"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_encode(x)
    
    def decode(self, z):
        """Decode from latent space"""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 2, 2, 2)
        return self.decoder(x)
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        z = self.encode(x)
        return self.decode(z), z


class BraTSDataProcessor:
    """Class for processing BraTS data and extracting patches"""
    
    def __init__(self, config: Config):
        self.config = config
        self.subjects = []
        self.patches = []
        self.patch_labels = []
        
    def load_subjects(self) -> List[str]:
        """Load list of available subjects"""
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
        subjects = []
        for subject_dir in dataset_path.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('BraTS-GLI-'):
                subjects.append(subject_dir.name)
                
        subjects.sort()
        logger.info(f"Found {len(subjects)} subjects in dataset")
        
        if self.config.num_subjects > 0:
            subjects = subjects[:self.config.num_subjects]
            logger.info(f"Using {len(subjects)} subjects as requested")
            
        return subjects
    
    def load_volume(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load all modalities and segmentation for a subject"""
        subject_path = Path(self.config.dataset_path) / subject_id
        
        # Load all modalities
        modalities = ['t1n', 't1c', 't2w', 't2f']
        volumes = []
        
        for modality in modalities:
            file_path = subject_path / f"{subject_id}-{modality}.nii.gz"
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            nii = nib.load(str(file_path))
            volume = nii.get_fdata().astype(np.float32)
            
            # Normalize volume
            volume = self.normalize_volume(volume)
            volumes.append(volume)
        
        # Stack modalities
        multi_modal_volume = np.stack(volumes, axis=0)  # Shape: [4, H, W, D]
        
        # Load segmentation
        seg_path = subject_path / f"{subject_id}-seg.nii.gz"
        if seg_path.exists():
            seg_nii = nib.load(str(seg_path))
            segmentation = seg_nii.get_fdata().astype(np.uint8)
        else:
            logger.warning(f"No segmentation found for {subject_id}")
            segmentation = np.zeros(multi_modal_volume.shape[1:], dtype=np.uint8)
            
        return multi_modal_volume, segmentation
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range"""
        # Remove background (assuming 0 is background)
        mask = volume > 0
        if mask.sum() > 0:
            volume_masked = volume[mask]
            # Normalize to 0-1 using percentiles to handle outliers
            p1, p99 = np.percentile(volume_masked, [1, 99])
            volume = np.clip(volume, p1, p99)
            volume = (volume - p1) / (p99 - p1)
        
        return volume
    
    def extract_normal_patches(self, volume: np.ndarray, segmentation: np.ndarray) -> List[np.ndarray]:
        """Extract patches that don't contain tumor information"""
        patches = []
        patch_size = self.config.patch_size
        
        # Get volume dimensions
        _, h, w, d = volume.shape
        
        # Calculate valid patch positions
        max_attempts = self.config.patches_per_volume * 10  # Try more to find normal patches
        attempts = 0
        
        while len(patches) < self.config.patches_per_volume and attempts < max_attempts:
            # Random patch position
            x = random.randint(0, max(0, h - patch_size))
            y = random.randint(0, max(0, w - patch_size))
            z = random.randint(0, max(0, d - patch_size))
            
            # Extract patch from segmentation
            seg_patch = segmentation[x:x+patch_size, y:y+patch_size, z:z+patch_size]
            
            # Check if patch contains tumor (any non-zero value in segmentation)
            if seg_patch.sum() == 0:  # No tumor in this patch
                # Extract corresponding volume patch
                volume_patch = volume[:, x:x+patch_size, y:y+patch_size, z:z+patch_size]
                
                # Check if patch has enough non-zero voxels (not just background)
                if np.mean(volume_patch > 0) > 0.1:  # At least 10% non-background
                    patches.append(volume_patch)
            
            attempts += 1
        
        if len(patches) < self.config.patches_per_volume:
            logger.warning(f"Could only extract {len(patches)} normal patches "
                         f"(requested {self.config.patches_per_volume})")
        
        return patches
    
    def extract_anomaly_patches(self, volume: np.ndarray, segmentation: np.ndarray) -> List[np.ndarray]:
        """Extract patches that contain tumor information"""
        patches = []
        patch_size = self.config.patch_size
        
        # Get volume dimensions
        _, h, w, d = volume.shape
        
        # Find tumor locations
        tumor_coords = np.where(segmentation > 0)
        if len(tumor_coords[0]) == 0:
            return patches
        
        # Extract patches around tumor locations
        max_attempts = self.config.patches_per_volume * 5
        attempts = 0
        
        while len(patches) < self.config.patches_per_volume and attempts < max_attempts:
            # Random tumor voxel
            idx = random.randint(0, len(tumor_coords[0]) - 1)
            center_x, center_y, center_z = (tumor_coords[0][idx], 
                                          tumor_coords[1][idx], 
                                          tumor_coords[2][idx])
            
            # Calculate patch boundaries
            x = max(0, min(center_x - patch_size//2, h - patch_size))
            y = max(0, min(center_y - patch_size//2, w - patch_size))
            z = max(0, min(center_z - patch_size//2, d - patch_size))
            
            # Extract patch
            seg_patch = segmentation[x:x+patch_size, y:y+patch_size, z:z+patch_size]
            
            # Check if patch contains tumor
            if seg_patch.sum() > 0:
                volume_patch = volume[:, x:x+patch_size, y:y+patch_size, z:z+patch_size]
                patches.append(volume_patch)
            
            attempts += 1
        
        return patches
    
    def process_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Process entire dataset and extract patches"""
        subjects = self.load_subjects()
        
        normal_patches = []
        anomaly_patches = []
        
        logger.info("Extracting patches from volumes...")
        
        for subject_id in tqdm(subjects, desc="Processing subjects"):
            try:
                volume, segmentation = self.load_volume(subject_id)
                
                # Extract normal patches (for training)
                normal_patches_subject = self.extract_normal_patches(volume, segmentation)
                normal_patches.extend(normal_patches_subject)
                
                # Extract anomaly patches (for testing)
                anomaly_patches_subject = self.extract_anomaly_patches(volume, segmentation)
                anomaly_patches.extend(anomaly_patches_subject)
                
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}: {e}")
                continue
        
        logger.info(f"Extracted {len(normal_patches)} normal patches")
        logger.info(f"Extracted {len(anomaly_patches)} anomaly patches")
        
        # Convert to numpy arrays
        if normal_patches:
            normal_patches = np.array(normal_patches)
        else:
            normal_patches = np.empty((0, 4, self.config.patch_size, 
                                     self.config.patch_size, self.config.patch_size))
            
        if anomaly_patches:
            anomaly_patches = np.array(anomaly_patches)
        else:
            anomaly_patches = np.empty((0, 4, self.config.patch_size, 
                                      self.config.patch_size, self.config.patch_size))
        
        # Combine patches and create labels
        all_patches = np.concatenate([normal_patches, anomaly_patches], axis=0)
        labels = np.concatenate([
            np.zeros(len(normal_patches)),  # Normal = 0
            np.ones(len(anomaly_patches))   # Anomaly = 1
        ])
        
        return all_patches, labels


class Trainer:
    """Training class for the autoencoder"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = config.device
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            if isinstance(batch, tuple):
                data, _ = batch
            else:
                data = batch
                
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(data)
            loss = self.criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.6f}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            
            for batch in progress_bar:
                if isinstance(batch, tuple):
                    data, _ = batch
                else:
                    data = batch
                    
                data = data.to(self.device)
                
                # Forward pass
                reconstructed, _ = self.model(data)
                loss = self.criterion(reconstructed, data)
                
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'Val Loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Full training loop"""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.config.save_model:
                        self.save_model('best_autoencoder_brats.pth')
                else:
                    patience_counter += 1
                
                epoch_time = time.time() - start_time
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                          f"Train Loss: {train_loss:.6f} - "
                          f"Val Loss: {val_loss:.6f} - "
                          f"Time: {epoch_time:.2f}s")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                epoch_time = time.time() - start_time
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                          f"Train Loss: {train_loss:.6f} - "
                          f"Time: {epoch_time:.2f}s")
        
        logger.info("Training completed!")
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        filepath = Path(self.config.output_dir) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        filepath = Path(self.config.output_dir) / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {filepath}")


class AnomalyDetector:
    """Anomaly detection using reconstruction error"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.eval()
        
    def compute_reconstruction_errors(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute reconstruction errors for all samples"""
        reconstruction_errors = []
        latent_features = []
        true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Computing reconstruction errors")
            
            for batch in progress_bar:
                if isinstance(batch, tuple):
                    data, labels = batch
                    true_labels.extend(labels.cpu().numpy())
                else:
                    data = batch
                    
                data = data.to(self.device)
                
                # Forward pass
                reconstructed, latent = self.model(data)
                
                # Compute reconstruction error (MSE per sample)
                mse = F.mse_loss(reconstructed, data, reduction='none')
                mse = mse.view(mse.size(0), -1).mean(dim=1)
                
                reconstruction_errors.extend(mse.cpu().numpy())
                latent_features.extend(latent.cpu().numpy())
        
        return (np.array(reconstruction_errors), 
                np.array(latent_features), 
                np.array(true_labels) if true_labels else None)
    
    def find_optimal_threshold(self, reconstruction_errors: np.ndarray, 
                             true_labels: np.ndarray) -> Tuple[float, Dict]:
        """Find optimal threshold for anomaly detection"""
        if true_labels is None:
            # Use percentile-based threshold
            threshold = np.percentile(reconstruction_errors, 
                                    self.config.anomaly_threshold_percentile)
            return threshold, {}
        
        # Find threshold that maximizes F1 score
        thresholds = np.linspace(reconstruction_errors.min(), 
                               reconstruction_errors.max(), 100)
        
        best_threshold = thresholds[0]
        best_f1 = 0
        best_metrics = {}
        
        for threshold in thresholds:
            predictions = (reconstruction_errors > threshold).astype(int)
            
            if len(np.unique(predictions)) > 1:  # Avoid division by zero
                f1 = f1_score(true_labels, predictions)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'accuracy': accuracy_score(true_labels, predictions),
                        'precision': precision_score(true_labels, predictions),
                        'recall': recall_score(true_labels, predictions),
                        'f1': f1
                    }
        
        return best_threshold, best_metrics
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Comprehensive evaluation of anomaly detection"""
        logger.info("Starting anomaly detection evaluation...")
        
        # Compute reconstruction errors
        reconstruction_errors, latent_features, true_labels = \
            self.compute_reconstruction_errors(data_loader)
        
        results = {
            'reconstruction_errors': reconstruction_errors,
            'latent_features': latent_features,
            'true_labels': true_labels
        }
        
        if true_labels is not None:
            # Find optimal threshold
            optimal_threshold, threshold_metrics = \
                self.find_optimal_threshold(reconstruction_errors, true_labels)
            
            # Make predictions with optimal threshold
            predictions = (reconstruction_errors > optimal_threshold).astype(int)
            
            # Compute metrics
            metrics = {
                'roc_auc': roc_auc_score(true_labels, reconstruction_errors),
                'average_precision': average_precision_score(true_labels, reconstruction_errors),
                'optimal_threshold': optimal_threshold,
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions),
                'recall': recall_score(true_labels, predictions),
                'f1_score': f1_score(true_labels, predictions),
                'confusion_matrix': confusion_matrix(true_labels, predictions)
            }
            
            results.update(metrics)
            
            # Log results
            logger.info("Anomaly Detection Results:")
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Average Precision: {metrics['average_precision']:.4f}")
            logger.info(f"Optimal Threshold: {metrics['optimal_threshold']:.6f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return results


class Visualizer:
    """Visualization utilities"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, train_losses: List[float], 
                           val_losses: List[float] = None):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('Training Progress', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, true_labels: np.ndarray, scores: np.ndarray, auc_score: float):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(true_labels, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, true_labels: np.ndarray, scores: np.ndarray, ap_score: float):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(true_labels, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, 
                label=f'PR Curve (AP = {ap_score:.3f})')
        
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_reconstruction_error_histogram(self, reconstruction_errors: np.ndarray, 
                                          true_labels: np.ndarray = None, 
                                          threshold: float = None):
        """Plot histogram of reconstruction errors"""
        plt.figure(figsize=(10, 6))
        
        if true_labels is not None:
            normal_errors = reconstruction_errors[true_labels == 0]
            anomaly_errors = reconstruction_errors[true_labels == 1]
            
            plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', 
                    color='blue', density=True)
            plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', 
                    color='red', density=True)
        else:
            plt.hist(reconstruction_errors, bins=50, alpha=0.7, 
                    color='blue', density=True)
        
        if threshold is not None:
            plt.axvline(threshold, color='black', linestyle='--', linewidth=2,
                       label=f'Threshold = {threshold:.4f}')
        
        plt.title('Reconstruction Error Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Reconstruction Error', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'reconstruction_error_histogram.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_latent_space_projection(self, latent_features: np.ndarray, 
                                   true_labels: np.ndarray = None, 
                                   method: str = 'tsne'):
        """Plot 2D projection of latent space"""
        if latent_features.shape[0] < 50:
            logger.warning("Too few samples for meaningful latent space visualization")
            return
        
        # Reduce dimensionality
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_features)//4))
            projection = reducer.fit_transform(latent_features)
            title = 't-SNE Projection of Latent Space'
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
            projection = reducer.fit_transform(latent_features)
            title = 'PCA Projection of Latent Space'
        
        plt.figure(figsize=(10, 8))
        
        if true_labels is not None:
            scatter = plt.scatter(projection[true_labels == 0, 0], 
                                projection[true_labels == 0, 1],
                                c='blue', alpha=0.6, label='Normal', s=50)
            scatter = plt.scatter(projection[true_labels == 1, 0], 
                                projection[true_labels == 1, 1],
                                c='red', alpha=0.6, label='Anomaly', s=50)
            plt.legend(fontsize=12)
        else:
            plt.scatter(projection[:, 0], projection[:, 1], 
                       c='blue', alpha=0.6, s=50)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'latent_space_{method.lower()}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_3d_patches(self, patches: np.ndarray, labels: np.ndarray = None, 
                           num_samples: int = 5):
        """Visualize 3D patches by showing middle slices"""
        if len(patches) == 0:
            return
            
        num_samples = min(num_samples, len(patches))
        indices = np.random.choice(len(patches), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        modality_names = ['T1n', 'T1c', 'T2w', 'T2-FLAIR']
        
        for i, idx in enumerate(indices):
            patch = patches[idx]  # Shape: [4, D, H, W]
            label_text = f"Label: {'Anomaly' if labels is not None and labels[idx] == 1 else 'Normal'}"
            
            for j, modality in enumerate(modality_names):
                # Show middle slice
                middle_slice = patch[j, :, :, patch.shape[3]//2]
                
                axes[i, j].imshow(middle_slice, cmap='gray')
                axes[i, j].set_title(f'{modality}\n{label_text if j == 0 else ""}')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_patches.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_plot(self, results: Dict):
        """Create a comprehensive summary plot"""
        if 'true_labels' not in results or results['true_labels'] is None:
            return
            
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Confusion Matrix
        plt.subplot(2, 4, 1)
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        
        # 2. ROC Curve
        plt.subplot(2, 4, 2)
        fpr, tpr, _ = roc_curve(results['true_labels'], results['reconstruction_errors'])
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {results["roc_auc"]:.3f}')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        plt.subplot(2, 4, 3)
        precision, recall, _ = precision_recall_curve(results['true_labels'], 
                                                    results['reconstruction_errors'])
        plt.plot(recall, precision, 'b-', linewidth=2, 
                label=f'AP = {results["average_precision"]:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Reconstruction Error Histogram
        plt.subplot(2, 4, 4)
        normal_errors = results['reconstruction_errors'][results['true_labels'] == 0]
        anomaly_errors = results['reconstruction_errors'][results['true_labels'] == 1]
        plt.hist(normal_errors, bins=30, alpha=0.7, label='Normal', density=True)
        plt.hist(anomaly_errors, bins=30, alpha=0.7, label='Anomaly', density=True)
        plt.axvline(results['optimal_threshold'], color='black', linestyle='--',
                   label=f'Threshold = {results["optimal_threshold"]:.4f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Error Distribution')
        plt.legend()
        
        # 5-8. Metrics Summary
        plt.subplot(2, 4, (5, 8))
        metrics_text = f"""
        ANOMALY DETECTION RESULTS
        
        ROC AUC: {results['roc_auc']:.4f}
        Average Precision: {results['average_precision']:.4f}
        
        Optimal Threshold: {results['optimal_threshold']:.6f}
        
        Accuracy: {results['accuracy']:.4f}
        Precision: {results['precision']:.4f}
        Recall: {results['recall']:.4f}
        F1 Score: {results['f1_score']:.4f}
        
        Normal Samples: {np.sum(results['true_labels'] == 0)}
        Anomaly Samples: {np.sum(results['true_labels'] == 1)}
        """
        
        plt.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_detection_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='3D Autoencoder for Anomaly Detection on BraTS Dataset'
    )
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, 
                       default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData',
                       help='Path to BraTS dataset')
    parser.add_argument('--num_subjects', type=int, default=0,
                       help='Number of subjects to use (0 = all)')
    parser.add_argument('--patch_size', type=int, default=32,
                       help='Size of 3D patches')
    parser.add_argument('--patches_per_volume', type=int, default=20,
                       help='Number of patches to extract per volume')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data for training')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension of autoencoder')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    
    # Training parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='ae_brats2_results',
                       help='Output directory')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualizations')
    
    # Anomaly detection parameters
    parser.add_argument('--anomaly_threshold_percentile', type=float, default=95,
                       help='Percentile for anomaly threshold when no labels available')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    config = Config(args)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Create output directory
    Path(config.output_dir).mkdir(exist_ok=True)
    
    logger.info("Starting BraTS Anomaly Detection with 3D Autoencoder")
    logger.info(f"Using device: {config.device}")
    logger.info(f"Configuration: {vars(config)}")
    
    # Process dataset
    processor = BraTSDataProcessor(config)
    patches, labels = processor.process_dataset()
    
    if len(patches) == 0:
        logger.error("No patches extracted! Check dataset path and parameters.")
        return
    
    logger.info(f"Dataset shape: {patches.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    
    # Split dataset
    dataset = BraTSDataset(patches, labels)
    
    # For training, we only use normal patches (label = 0)
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]
    
    # Split normal patches for training/validation
    train_size = int(len(normal_indices) * config.train_ratio)
    train_indices = normal_indices[:train_size]
    val_indices = normal_indices[train_size:]
    
    # Create datasets
    train_patches = patches[train_indices]
    val_patches = patches[val_indices]
    
    train_dataset = BraTSDataset(train_patches)
    val_dataset = BraTSDataset(val_patches)
    
    # Test dataset includes both normal and anomaly patches
    test_indices = np.concatenate([val_indices, anomaly_indices])
    test_patches = patches[test_indices]
    test_labels = labels[test_indices]
    test_dataset = BraTSDataset(test_patches, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)} (Normal: {np.sum(test_labels == 0)}, Anomaly: {np.sum(test_labels == 1)})")
    
    # Create model
    model = Autoencoder3D(input_channels=4, latent_dim=config.latent_dim)
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Create visualizer
    if config.visualize:
        visualizer = Visualizer(config)
        
        # Plot training curves
        visualizer.plot_training_curves(trainer.train_losses, trainer.val_losses)
        
        # Visualize sample patches
        sample_indices = np.random.choice(len(patches), min(5, len(patches)), replace=False)
        visualizer.visualize_3d_patches(patches[sample_indices], labels[sample_indices])
    
    # Anomaly detection evaluation
    detector = AnomalyDetector(model, config)
    results = detector.evaluate(test_loader)
    
    # Create visualizations
    if config.visualize and results['true_labels'] is not None:
        visualizer.plot_confusion_matrix(results['confusion_matrix'])
        visualizer.plot_roc_curve(results['true_labels'], 
                                 results['reconstruction_errors'], 
                                 results['roc_auc'])
        visualizer.plot_precision_recall_curve(results['true_labels'], 
                                              results['reconstruction_errors'], 
                                              results['average_precision'])
        visualizer.plot_reconstruction_error_histogram(results['reconstruction_errors'], 
                                                      results['true_labels'], 
                                                      results['optimal_threshold'])
        
        # Latent space visualization
        if len(results['latent_features']) >= 50:
            visualizer.plot_latent_space_projection(results['latent_features'], 
                                                   results['true_labels'], 
                                                   method='tsne')
            visualizer.plot_latent_space_projection(results['latent_features'], 
                                                   results['true_labels'], 
                                                   method='pca')
        
        # Summary plot
        visualizer.create_summary_plot(results)
    
    logger.info("Analysis completed successfully!")
    logger.info(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main() 