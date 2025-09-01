#!/usr/bin/env python3
"""
3D VQ-VAE-based Anomaly Detection for BraTS Dataset

This program mirrors the pipeline of ae_brats.py (patch extraction, subject-level
splitting, unsupervised training on normal patches, thresholding on normal
validation errors, and evaluation on mixed test data), but replaces the model
with a 3D Vector-Quantized VAE (VQ-VAE).
"""

import os
import sys
import glob
import argparse
import random
import warnings
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.figure_factory as ff

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class Config:
    """Configuration mirroring ae_brats, with VQ-VAE additions."""

    def __init__(self):
        # Dataset paths
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "vqvae_brats2_results"

        # Patch extraction parameters
        self.patch_size = 32
        self.patches_per_volume = 50
        self.min_non_zero_ratio = 0.2
        self.max_normal_to_anomaly_ratio = 3
        self.min_tumor_ratio_in_patch = 0.05

        # Additional patch quality parameters
        self.min_patch_std = 0.01
        self.min_patch_mean = 0.05
        self.max_tumor_ratio_normal = 0.01
        self.min_tumor_ratio_anomaly = 0.05
        self.max_normal_patches_per_subject = 100
        self.max_anomaly_patches_per_subject = 50

        # Segmentation labels for anomaly detection (BraTS: 1,2,4 are tumor)
        self.anomaly_labels = [1, 2, 4]

        # Brain tissue quality parameters for normal patches
        self.min_brain_tissue_ratio = 0.3
        self.max_background_intensity = 0.1
        self.min_brain_mean_intensity = 0.1
        self.max_high_intensity_ratio = 0.7
        self.high_intensity_threshold = 0.9
        self.edge_margin = 8

        # Model parameters
        self.learning_rate = 5e-5
        self.batch_size = 8
        self.num_epochs = 100
        self.early_stopping_patience = 20

        # VQ-VAE specific parameters
        self.codebook_size = 512
        self.embedding_dim = 128
        self.commitment_beta = 0.25

        # Training parameters
        self.train_test_split = 0.8
        self.validation_split = 0.2

        # Visualization parameters
        self.slice_axis = 'axial'

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4 if torch.cuda.is_available() else 0

        os.makedirs(self.output_dir, exist_ok=True)


class BraTSPatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, transform=None):
        self.patches = patches
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        patch = torch.FloatTensor(patch).unsqueeze(0)
        label = torch.FloatTensor([label])
        if self.transform:
            patch = self.transform(patch)
        return patch, label


class VectorQuantizer(nn.Module):
    """
    Standard VQ layer (no EMA) for 3D feature maps.
    Input: z_e in shape (B, C, D, H, W)
    Output: z_q same shape, vq_loss, perplexity, indices
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: (B, C, D, H, W)
        b, c, d, h, w = z.shape
        # Move channel to last dim to compute distances
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        z_flat = z_perm.view(-1, c)  # (B*D*H*W, C)

        # Compute distances to embeddings
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x.e
        x_sq = torch.sum(z_flat ** 2, dim=1, keepdim=True)  # (N, 1)
        e_sq = torch.sum(self.embedding.weight ** 2, dim=1)  # (K)
        xe = torch.matmul(z_flat, self.embedding.weight.t())  # (N, K)
        distances = x_sq + e_sq.unsqueeze(0) - 2 * xe  # (N, K)

        encoding_indices = torch.argmin(distances, dim=1)  # (N)
        z_q_flat = self.embedding(encoding_indices)  # (N, C)
        z_q = z_q_flat.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # Losses
        # Note: stop gradients appropriately (straight-through estimator)
        embedding_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q, z.detach())
        vq_loss = embedding_loss + commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Perplexity
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(z_flat.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, vq_loss, perplexity, encoding_indices.view(b, d, h, w)


class VQVAE3D(nn.Module):
    def __init__(self, input_channels: int = 1, embedding_dim: int = 128, codebook_size: int = 512,
                 commitment_beta: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Encoder: 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.quantizer = VectorQuantizer(num_embeddings=codebook_size,
                                         embedding_dim=embedding_dim,
                                         commitment_cost=commitment_beta)

        # Decoder: 4 -> 8 -> 16 -> 32
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor):
        z_e = self.encode(x)
        z_q, vq_loss, perplexity, _ = self.quantizer(z_e)
        x_recon = self.decode(z_q)
        # Latent features for visualization: global average over spatial dims
        latent_features = torch.mean(z_q, dim=(2, 3, 4))
        return x_recon, latent_features, vq_loss, perplexity


class BraTSDataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def load_volume(self, subject_path: str, modality: str = 't1c') -> Tuple[np.ndarray, np.ndarray]:
        volume_file = glob.glob(os.path.join(subject_path, f"*-{modality}.nii.gz"))[0]
        seg_file = glob.glob(os.path.join(subject_path, "*-seg.nii.gz"))[0]
        volume = nib.load(volume_file).get_fdata()
        segmentation = nib.load(seg_file).get_fdata()
        return volume, segmentation

    def is_brain_tissue_patch(self, patch: np.ndarray) -> bool:
        brain_tissue_mask = patch > self.config.max_background_intensity
        brain_tissue_ratio = np.sum(brain_tissue_mask) / patch.size
        if brain_tissue_ratio < self.config.min_brain_tissue_ratio:
            return False
        brain_tissue_values = patch[brain_tissue_mask]
        if len(brain_tissue_values) == 0:
            return False
        mean_brain_intensity = np.mean(brain_tissue_values)
        if mean_brain_intensity < self.config.min_brain_mean_intensity:
            return False
        high_intensity_mask = patch > self.config.high_intensity_threshold
        high_intensity_ratio = np.sum(high_intensity_mask) / patch.size
        if high_intensity_ratio > self.config.max_high_intensity_ratio:
            return False
        if patch.std() < self.config.min_patch_std * 2:
            return False
        reasonable_intensity_mask = (patch > 0.05) & (patch < 0.95)
        reasonable_ratio = np.sum(reasonable_intensity_mask) / patch.size
        if reasonable_ratio < 0.5:
            return False
        return True

    def is_anomaly_segmentation(self, segmentation_patch: np.ndarray) -> bool:
        for label in self.config.anomaly_labels:
            if np.any(segmentation_patch == label):
                return True
        return False

    def get_anomaly_ratio_in_patch(self, segmentation_patch: np.ndarray) -> float:
        anomaly_mask = np.isin(segmentation_patch, self.config.anomaly_labels)
        return np.sum(anomaly_mask) / segmentation_patch.size

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        volume = volume.astype(np.float32)
        non_zero_mask = volume > 0
        if np.sum(non_zero_mask) == 0:
            return volume
        non_zero_values = volume[non_zero_mask]
        p1 = np.percentile(non_zero_values, 1)
        p99 = np.percentile(non_zero_values, 99)
        volume = np.clip(volume, p1, p99)
        volume[non_zero_mask] = (volume[non_zero_mask] - p1) / (p99 - p1)
        volume = np.clip(volume, 0, 1)
        return volume

    def extract_normal_patches(self, volume: np.ndarray, segmentation: np.ndarray) -> List[np.ndarray]:
        patches = []
        brain_mask = (volume > self.config.max_background_intensity) & (volume < 0.95)
        anomaly_mask = np.isin(segmentation, self.config.anomaly_labels)
        normal_tissue_coords = np.where(~anomaly_mask)
        brain_coords = np.where(brain_mask)
        normal_tissue_set = set(zip(normal_tissue_coords[0], normal_tissue_coords[1], normal_tissue_coords[2]))
        brain_set = set(zip(brain_coords[0], brain_coords[1], brain_coords[2]))
        valid_coords_set = normal_tissue_set.intersection(brain_set)
        if len(valid_coords_set) == 0:
            return patches
        valid_coords_list = list(valid_coords_set)
        edge_margin = self.config.edge_margin
        filtered_coords = []
        for x, y, z in valid_coords_list:
            if (x >= edge_margin and x < volume.shape[0] - edge_margin and
                y >= edge_margin and y < volume.shape[1] - edge_margin and
                z >= edge_margin and z < volume.shape[2] - edge_margin):
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
            if getattr(self.config, 'verbose', False):
                print("Warning: No valid coordinates found for normal patch extraction")
            return patches
        max_patches = min(len(filtered_coords) // 20, self.config.max_normal_patches_per_subject)
        max_patches = max(max_patches, 10)
        num_to_sample = min(max_patches * 5, len(filtered_coords))
        indices = np.random.choice(len(filtered_coords), size=num_to_sample, replace=False)
        patch_coords = [filtered_coords[i] for i in indices]
        patches_extracted = 0
        patches_rejected = 0
        for x, y, z in tqdm(patch_coords, desc="Extracting normal patches", leave=False):
            x_start = x - self.config.patch_size // 2
            x_end = x_start + self.config.patch_size
            y_start = y - self.config.patch_size // 2
            y_end = y_start + self.config.patch_size
            z_start = z - self.config.patch_size // 2
            z_end = z_start + self.config.patch_size
            patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
            patch_seg = segmentation[x_start:x_end, y_start:y_end, z_start:z_end]
            anomaly_ratio = self.get_anomaly_ratio_in_patch(patch_seg)
            if anomaly_ratio > self.config.max_tumor_ratio_normal:
                patches_rejected += 1
                continue
            if not self.is_brain_tissue_patch(patch):
                patches_rejected += 1
                continue
            if patch.std() < self.config.min_patch_std:
                patches_rejected += 1
                continue
            patches.append(patch)
            patches_extracted += 1
            if patches_extracted >= max_patches:
                break
        if getattr(self.config, 'verbose', False):
            print(f"Normal patch extraction: {patches_extracted} accepted, {patches_rejected} rejected")
        return patches

    def extract_anomalous_patches(self, volume: np.ndarray, segmentation: np.ndarray) -> List[np.ndarray]:
        patches = []
        anomaly_mask = np.isin(segmentation, self.config.anomaly_labels)
        anomaly_coords = np.where(anomaly_mask)
        if len(anomaly_coords[0]) == 0:
            return patches
        max_patches = min(len(anomaly_coords[0]) // 50, self.config.max_anomaly_patches_per_subject)
        if max_patches == 0:
            return patches
        indices = np.random.choice(len(anomaly_coords[0]), size=min(max_patches, len(anomaly_coords[0])), replace=False)
        patch_coords = [(anomaly_coords[0][i], anomaly_coords[1][i], anomaly_coords[2][i]) for i in indices]
        for x, y, z in tqdm(patch_coords, desc="Extracting anomaly patches", leave=False):
            x_start = max(0, x - self.config.patch_size // 2)
            x_end = min(volume.shape[0], x_start + self.config.patch_size)
            y_start = max(0, y - self.config.patch_size // 2)
            y_end = min(volume.shape[1], y_start + self.config.patch_size)
            z_start = max(0, z - self.config.patch_size // 2)
            z_end = min(volume.shape[2], z_start + self.config.patch_size)
            if x_end - x_start < self.config.patch_size:
                x_start = max(0, x_end - self.config.patch_size)
            if y_end - y_start < self.config.patch_size:
                y_start = max(0, y_end - self.config.patch_size)
            if z_end - z_start < self.config.patch_size:
                z_start = max(0, z_end - self.config.patch_size)
            if (x_end - x_start != self.config.patch_size or
                y_end - y_start != self.config.patch_size or
                z_end - z_start != self.config.patch_size):
                continue
            patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
            if patch.std() > self.config.min_patch_std and patch.mean() > self.config.min_patch_mean:
                patch_seg = segmentation[x_start:x_end, y_start:y_end, z_start:z_end]
                anomaly_ratio = self.get_anomaly_ratio_in_patch(patch_seg)
                if anomaly_ratio >= self.config.min_tumor_ratio_anomaly:
                    patches.append(patch)
        return patches

    def process_dataset(self, num_subjects: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        subject_dirs = [d for d in os.listdir(self.config.dataset_path)
                        if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        if num_subjects:
            subject_dirs = subject_dirs[:num_subjects]
        if getattr(self.config, 'verbose', False):
            print(f"Processing {len(subject_dirs)} subjects...")
        all_normal_patches = []
        all_anomalous_patches = []
        all_normal_subjects = []
        all_anomalous_subjects = []
        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            subject_path = os.path.join(self.config.dataset_path, subject_dir)
            try:
                volume, segmentation = self.load_volume(subject_path)
                if volume is None or segmentation is None:
                    continue
                volume = self.normalize_volume(volume)
                normal_patches = self.extract_normal_patches(volume, segmentation)
                anomalous_patches = self.extract_anomalous_patches(volume, segmentation)
                all_normal_patches.extend(normal_patches)
                all_normal_subjects.extend([subject_dir] * len(normal_patches))
                all_anomalous_patches.extend(anomalous_patches)
                all_anomalous_subjects.extend([subject_dir] * len(anomalous_patches))
            except Exception:
                continue
        if getattr(self.config, 'verbose', False):
            print(f"Extracted {len(all_normal_patches)} normal patches and {len(all_anomalous_patches)} anomalous patches")
        num_anomalous = len(all_anomalous_patches)
        max_normal = int(num_anomalous * self.config.max_normal_to_anomaly_ratio)
        if len(all_normal_patches) > max_normal and max_normal > 0:
            indices = np.random.choice(len(all_normal_patches), max_normal, replace=False)
            all_normal_patches = [all_normal_patches[i] for i in indices]
            all_normal_subjects = [all_normal_subjects[i] for i in indices]
        if getattr(self.config, 'verbose', False):
            print(f"Final dataset: {len(all_normal_patches)} normal, {len(all_anomalous_patches)} anomalous patches")
        all_patches = all_normal_patches + all_anomalous_patches
        labels = [0] * len(all_normal_patches) + [1] * len(all_anomalous_patches)
        subjects = all_normal_subjects + all_anomalous_subjects
        patches_array = np.array(all_patches, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)
        return patches_array, labels_array, subjects


class VQVAEAnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self.model = VQVAE3D(
            input_channels=1,
            embedding_dim=config.embedding_dim,
            codebook_size=config.codebook_size,
            commitment_beta=config.commitment_beta,
        ).to(config.device)
        self.scaler = GradScaler()

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        scaler = GradScaler()
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        total_steps = self.config.num_epochs * max(1, len(train_loader))
        if getattr(self.config, 'verbose', False):
            print("TRAINING MODE: VQ-VAE will be trained ONLY on normal data (unsupervised)")
        with tqdm(total=total_steps, desc="Training Progress") as pbar:
            for epoch in range(self.config.num_epochs):
                self.model.train()
                train_loss = 0.0
                normal_samples_processed = 0
                for batch_idx, (data, labels) in enumerate(train_loader):
                    data = data.to(self.config.device)
                    labels = labels.to(self.config.device)
                    normal_mask = (labels == 0).squeeze()
                    if normal_mask.sum() == 0:
                        pbar.update(1)
                        continue
                    normal_data = data[normal_mask]
                    normal_samples_processed += normal_data.size(0)
                    optimizer.zero_grad()
                    with autocast():
                        recon, _, vq_loss, _ = self.model(normal_data)
                        recon_loss = criterion(recon, normal_data)
                        loss = recon_loss + vq_loss
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
                avg_train_loss = train_loss / max(1, len(train_loader))
                train_losses.append(avg_train_loss)
                # Validation on normal data only
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, labels in val_loader:
                        data = data.to(self.config.device)
                        labels = labels.to(self.config.device)
                        normal_mask = (labels == 0).squeeze()
                        if normal_mask.sum() == 0:
                            continue
                        normal_data = data[normal_mask]
                        with autocast():
                            recon, _, vq_loss, _ = self.model(normal_data)
                            recon_loss = criterion(recon, normal_data)
                            loss = recon_loss + vq_loss
                        val_loss += loss.item()
                avg_val_loss = val_loss / max(1, len(val_loader))
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, 'best_vqvae_3d.pth'))
                else:
                    patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    pbar.write(f"Early stopping triggered after {epoch+1} epochs")
                    break
        if getattr(self.config, 'verbose', False):
            print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
        self.save_training_plots(train_losses, val_losses)
        return train_losses, val_losses

    def save_training_plots(self, train_losses: List[float], val_losses: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VQ-VAE Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_reconstruction_errors(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        reconstruction_errors = []
        true_labels = []
        latent_features = []
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Calculating reconstruction errors"):
                data = data.to(self.config.device)
                with autocast():
                    reconstructed, latent, _, _ = self.model(data)
                mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2, 3, 4))
                reconstruction_errors.extend(mse.cpu().numpy())
                true_labels.extend(labels.cpu().numpy().flatten())
                latent_features.extend(latent.cpu().numpy())
        return np.array(reconstruction_errors), np.array(true_labels), np.array(latent_features)

    def find_unsupervised_threshold(self, val_loader: DataLoader) -> float:
        if getattr(self.config, 'verbose', False):
            print(f"\nUNSUPERVISED THRESHOLD DETERMINATION (Normal validation only)")
        self.model.eval()
        normal_val_errors = []
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Computing validation errors"):
                data = data.to(self.config.device)
                labels = labels.to(self.config.device)
                normal_mask = (labels == 0).squeeze()
                if normal_mask.sum() == 0:
                    continue
                normal_data = data[normal_mask]
                with autocast():
                    reconstructed, _, _, _ = self.model(normal_data)
                mse = torch.mean((normal_data - reconstructed) ** 2, dim=(1, 2, 3, 4))
                normal_val_errors.extend(mse.cpu().numpy())
        normal_val_errors = np.array(normal_val_errors)
        threshold = np.percentile(normal_val_errors, 95)
        return threshold

    def evaluate(self, test_loader: DataLoader, val_loader: DataLoader) -> Dict:
        model_path = os.path.join(self.config.output_dir, 'best_vqvae_3d.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            if getattr(self.config, 'verbose', False):
                print("Loaded best VQ-VAE model for evaluation")
        optimal_threshold = self.find_unsupervised_threshold(val_loader)
        reconstruction_errors, true_labels, latent_features = self.calculate_reconstruction_errors(test_loader)
        print(f"\nEVALUATION: Truly Unsupervised Anomaly Detection Results")
        print("="*60)
        print(f"Total test samples: {len(reconstruction_errors)}")
        print(f"Normal samples: {np.sum(true_labels == 0)}")
        print(f"Anomalous samples: {np.sum(true_labels == 1)}")
        predictions = (reconstruction_errors > optimal_threshold).astype(int)
        try:
            roc_auc = roc_auc_score(true_labels, reconstruction_errors)
            average_precision = average_precision_score(true_labels, reconstruction_errors)
        except Exception:
            roc_auc = 0.0
            average_precision = 0.0
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0
        dsc = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        normal_errors = reconstruction_errors[true_labels == 0]
        anomaly_errors = reconstruction_errors[true_labels == 1]
        separation_ratio = anomaly_errors.mean() / normal_errors.mean() if normal_errors.mean() > 0 else 0
        print(f"\nTRULY UNSUPERVISED ANOMALY DETECTION PERFORMANCE:")
        print("="*60)
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
        print("="*60)
        print(f"\nPOST-HOC RECONSTRUCTION ERROR ANALYSIS:")
        print(f"Normal errors    - Mean: {normal_errors.mean():.6f}, Std: {normal_errors.std():.6f}")
        print(f"Anomaly errors   - Mean: {anomaly_errors.mean():.6f}, Std: {anomaly_errors.std():.6f}")
        print(f"Separation ratio - Anomaly/Normal: {separation_ratio:.3f}")
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


class Visualizer:
    def __init__(self, config: Config):
        self.config = config

    def plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray):
        cm = confusion_matrix(true_labels, predictions)
        if cm.shape != (2, 2):
            if getattr(self.config, 'verbose', False):
                print("WARNING: Confusion matrix is not 2x2. Skipping plot generation.")
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
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Blues', font_colors=['black', 'white'])
        fig.update_layout(
            title_text='<b>Confusion Matrix</b><br>(Count and Percentage)',
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
            xref='paper', yref='paper', x=0.0, y=-0.28,
            bordercolor="black", borderwidth=1, bgcolor="lightgray", font_size=12
        )
        fig.update_layout(margin=dict(t=100, b=150))
        output_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        try:
            fig.write_image(output_path, width=800, height=800, scale=2)
        except ValueError as e:
            print(f"ERROR: Could not save confusion matrix plot: {e}")
            print("Please install 'plotly' and 'kaleido' (`pip install plotly kaleido`).")

    def plot_roc_curve(self, true_labels: np.ndarray, reconstruction_errors: np.ndarray):
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
        plt.savefig(os.path.join(self.config.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curve(self, true_labels: np.ndarray, reconstruction_errors: np.ndarray):
        precision, recall, _ = precision_recall_curve(true_labels, reconstruction_errors)
        avg_precision = average_precision_score(true_labels, reconstruction_errors)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_reconstruction_error_histogram(self, reconstruction_errors: np.ndarray, true_labels: np.ndarray, optimal_threshold: float):
        normal_errors = reconstruction_errors[true_labels == 0]
        anomaly_errors = reconstruction_errors[true_labels == 1]
        plt.figure(figsize=(12, 6))
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {optimal_threshold:.6f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'reconstruction_error_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_latent_space_visualization(self, latent_features: np.ndarray, true_labels: np.ndarray):
        print("Creating latent space visualizations...")
        if len(latent_features) > 2000:
            indices = np.random.choice(len(latent_features), 2000, replace=False)
            latent_features = latent_features[indices]
            true_labels = true_labels[indices]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        print("Computing PCA...")
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(latent_features)
        ax1.scatter(pca_features[true_labels == 0, 0], pca_features[true_labels == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
        ax1.scatter(pca_features[true_labels == 1, 0], pca_features[true_labels == 1, 1], c='red', alpha=0.6, label='Anomaly', s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA of Latent Features')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_features)//4))
        tsne_features = tsne.fit_transform(latent_features)
        ax2.scatter(tsne_features[true_labels == 0, 0], tsne_features[true_labels == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
        ax2.scatter(tsne_features[true_labels == 1, 0], tsne_features[true_labels == 1, 1], c='red', alpha=0.6, label='Anomaly', s=20)
        ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2'); ax2.set_title('t-SNE of Latent Features')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'latent_space_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_report(self, results: Dict):
        self.plot_confusion_matrix(results['true_labels'], results['predictions'])
        self.plot_roc_curve(results['true_labels'], results['reconstruction_errors'])
        self.plot_precision_recall_curve(results['true_labels'], results['reconstruction_errors'])
        self.plot_reconstruction_error_histogram(results['reconstruction_errors'], results['true_labels'], results['optimal_threshold'])
        self.plot_latent_space_visualization(results['latent_features'], results['true_labels'])
        print(f"\nAll visualizations saved to: {self.config.output_dir}")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='3D VQ-VAE Anomaly Detection for BraTS Dataset')
    parser.add_argument('--num_subjects', type=int, default=None, help='Number of subjects to use (default: all)')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of 3D patches (default: 32)')
    parser.add_argument('--patches_per_volume', type=int, default=50, help='Number of patches per volume (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--embedding_dim', type=int, default=128, help='VQ-VAE embedding dimension (default: 128)')
    parser.add_argument('--codebook_size', type=int, default=512, help='Number of VQ codebook entries (default: 512)')
    parser.add_argument('--commitment_beta', type=float, default=0.25, help='VQ commitment cost beta (default: 0.25)')
    parser.add_argument('--output_dir', type=str, default='vqvae_brats2_results', help='Output directory (default: vqvae_brats2_results)')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData', help='Path to BraTS dataset')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3, help='Max ratio of normal to anomaly patches')
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05, help='Min tumor ratio in anomalous patches')
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 2, 4], help='BraTS labels considered anomalies (default: 1 2 4)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    config = Config()
    config.num_subjects = args.num_subjects
    config.patch_size = args.patch_size
    config.patches_per_volume = args.patches_per_volume
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.embedding_dim = args.embedding_dim
    config.codebook_size = args.codebook_size
    config.commitment_beta = args.commitment_beta
    config.output_dir = args.output_dir
    config.dataset_path = args.dataset_path
    config.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    config.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    config.anomaly_labels = args.anomaly_labels
    config.verbose = args.verbose
    os.makedirs(config.output_dir, exist_ok=True)

    if getattr(config, 'verbose', False):
        print("="*60)
        print("3D VQ-VAE ANOMALY DETECTION FOR BRATS DATASET")
        print("="*60)
        print(f"Device: {config.device}")
        print(f"Dataset path: {config.dataset_path}")
        print(f"Output directory: {config.output_dir}")
        print(f"Patch size: {config.patch_size}^3")
        print(f"Patches per volume: {config.patches_per_volume}")
        print(f"VQ: embedding_dim={config.embedding_dim}, codebook_size={config.codebook_size}, beta={config.commitment_beta}")
        label_names = {0: "Background/Normal", 1: "NCR/NET", 2: "ED", 4: "ET"}
        anomaly_names = [f"{label} ({label_names.get(label, 'Unknown')})" for label in config.anomaly_labels]
        print(f"Anomaly labels: {anomaly_names}")
        print("="*60)
    else:
        anomaly_names = [f"{label}" for label in config.anomaly_labels]
        print(f"3D VQ-VAE Anomaly Detection | Anomaly labels: {anomaly_names} | Output: {config.output_dir}")

    # Step 1: Process dataset and extract patches
    if getattr(config, 'verbose', False):
        print("\n1. Processing dataset and extracting patches...")
    processor = BraTSDataProcessor(config)
    patches, labels, subjects = processor.process_dataset(config.num_subjects)
    if len(patches) == 0:
        print("Error: No patches extracted! Please check your dataset path and structure.")
        return
    if getattr(config, 'verbose', False):
        print(f"Total patches extracted: {len(patches)}")
        print(f"Patch shape: {patches[0].shape}")
        print(f"Normal patches: {np.sum(labels == 0)}")
        print(f"Anomalous patches: {np.sum(labels == 1)}")

    # Step 2: Subject-level split and unsupervised constraints
    if getattr(config, 'verbose', False):
        print("\n2. Subject-level data splitting for truly unsupervised anomaly detection...")
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        print("ERROR: Dataset contains only one class! Cannot perform anomaly detection.")
        return
    unique_subjects = list(set(subjects))
    np.random.shuffle(unique_subjects)
    n_subjects = len(unique_subjects)
    train_subjects = unique_subjects[:int(0.6 * n_subjects)]
    val_subjects = unique_subjects[int(0.6 * n_subjects):int(0.8 * n_subjects)]
    test_subjects = unique_subjects[int(0.8 * n_subjects):]
    train_indices = [i for i, subj in enumerate(subjects) if subj in train_subjects]
    val_indices = [i for i, subj in enumerate(subjects) if subj in val_subjects]
    test_indices = [i for i, subj in enumerate(subjects) if subj in test_subjects]
    X_train_all = patches[train_indices]; y_train_all = labels[train_indices]
    X_val_all = patches[val_indices]; y_val_all = labels[val_indices]
    X_test = patches[test_indices]; y_test = labels[test_indices]
    train_normal_mask = (y_train_all == 0)
    val_normal_mask = (y_val_all == 0)
    X_train_normal = X_train_all[train_normal_mask]
    y_train_normal = y_train_all[train_normal_mask]
    X_val_normal = X_val_all[val_normal_mask]
    y_val_normal = y_val_all[val_normal_mask]
    if getattr(config, 'verbose', False):
        print(f"Training set (NORMAL ONLY): {len(X_train_normal)} patches from {len(train_subjects)} subjects")
        print(f"Validation set (NORMAL ONLY): {len(X_val_normal)} patches from {len(val_subjects)} subjects")
        print(f"Test set (MIXED): {len(X_test)} patches from {len(test_subjects)} subjects")
    assert len(set(train_subjects) & set(val_subjects)) == 0
    assert len(set(train_subjects) & set(test_subjects)) == 0
    assert len(set(val_subjects) & set(test_subjects)) == 0
    if getattr(config, 'verbose', False):
        print("✓ No subject overlap confirmed - data leakage prevented!")

    train_dataset = BraTSPatchDataset(X_train_normal, y_train_normal)
    val_dataset = BraTSPatchDataset(X_val_normal, y_val_normal)
    test_dataset = BraTSPatchDataset(X_test, y_test)

    if getattr(config, 'verbose', False):
        print("\n3. Creating data loaders...")
        print("\nCORRECTED ANOMALY DETECTION APPROACH:")
        print("✓ Training: ONLY normal data (learn normal patterns)")
        print("✓ Validation: ONLY normal data (monitor overfitting)")
        print("✓ Testing: MIXED data (evaluate detection)")
        print("✓ Detection: High reconstruction error = Anomaly")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    if getattr(config, 'verbose', False):
        print("\n4. Training 3D VQ-VAE...")
    detector = VQVAEAnomalyDetector(config)
    train_losses, val_losses = detector.train(train_loader, val_loader)

    if getattr(config, 'verbose', False):
        print("\n5. Evaluating model on test set...")
    results = detector.evaluate(test_loader, val_loader)

    if getattr(config, 'verbose', False):
        print("\n6. Creating visualizations...")
    visualizer = Visualizer(config)
    visualizer.create_summary_report(results)

    # Optional: Visualize sample patches (reuse test set)
    # Note: visualization of patches (many images) omitted here to keep parity minimal

    end_time = time.time()
    total_time = end_time - start_time
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
    results_file = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("TRULY UNSUPERVISED 3D VQ-VAE Anomaly Detection Results\n")
        f.write("="*60 + "\n")
        f.write("DATA LEAKAGE PREVENTION MEASURES:\n")
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
        f.write(f"  VQ embedding dim: {config.embedding_dim}\n")
        f.write(f"  Codebook size: {config.codebook_size}\n")
        f.write(f"  Commitment beta: {config.commitment_beta}\n")
        f.write(f"  Learning rate: {config.learning_rate}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  Number of epochs: {config.num_epochs}\n")
        f.write(f"  Training samples: {len(train_dataset)}\n")
        f.write(f"  Validation samples: {len(val_dataset)}\n")
        f.write(f"  Test samples: {len(test_dataset)}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")
        f.write("="*60 + "\n")
        f.write(f"EXECUTION TIME:\n")
        f.write(f"  Total time: {time_formatted} ({total_time:.1f} seconds)\n")
    if getattr(config, 'verbose', False):
        print(f"\nPipeline completed in {time_formatted}. Results saved to: {results_file}")
    else:
        print(f"\nPipeline completed in {time_formatted}. Results saved to: {results_file}")


if __name__ == "__main__":
    main()


