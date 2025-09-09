#!/usr/bin/env python3
"""
3D Transformer-GAN-based Anomaly Detection for BraTS Dataset

This mirrors the pipeline used in ae_brats.py and vqvae_brats2.py:
- Subject-level splitting (to avoid leakage)
- Train ONLY on normal patches (unsupervised)
- Determine threshold using ONLY normal validation data
- Evaluate on mixed test data

Model:
- Generator: Latent Transformer decoder that maps a compact latent sequence to a 3D volume via conv upsampling
- Discriminator: 3D PatchGAN classifier

CLI adds: --tumor_core_only to restrict anomalies to tumor core (labels [1, 4])
"""

import os
import glob
import argparse
import random
import warnings
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

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

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class Config:
    def __init__(self):
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "transformer_gan_brats_results"

        # Patch extraction
        self.patch_size = 32
        self.patches_per_volume = 50
        self.min_non_zero_ratio = 0.2
        self.max_normal_to_anomaly_ratio = 3
        self.min_tumor_ratio_in_patch = 0.05

        # Patch quality
        self.min_patch_std = 0.01
        self.min_patch_mean = 0.05
        self.max_tumor_ratio_normal = 0.01
        self.min_tumor_ratio_anomaly = 0.05
        self.max_normal_patches_per_subject = 100
        self.max_anomaly_patches_per_subject = 50

        # Segmentation labels (BraTS tumor: 1,2,4)
        self.anomaly_labels = [1, 2, 4]

        # Brain tissue quality
        self.min_brain_tissue_ratio = 0.3
        self.max_background_intensity = 0.1
        self.min_brain_mean_intensity = 0.1
        self.max_high_intensity_ratio = 0.7
        self.high_intensity_threshold = 0.9
        self.edge_margin = 8

        # Model params
        self.learning_rate = 2e-4
        self.batch_size = 8
        self.num_epochs = 100
        self.early_stopping_patience = 20
        self.rec_weight = 50.0

        # Transformer generator params
        self.latent_tokens = 64
        self.token_dim = 256
        self.num_transformer_layers = 6
        self.num_heads = 8
        self.mlp_ratio = 4

        # Device
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
            return patches
        max_patches = min(len(filtered_coords) // 20, self.config.max_normal_patches_per_subject)
        max_patches = max(max_patches, 10)
        num_to_sample = min(max_patches * 5, len(filtered_coords))
        indices = np.random.choice(len(filtered_coords), size=num_to_sample, replace=False)
        patch_coords = [filtered_coords[i] for i in indices]
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
                continue
            if not self.is_brain_tissue_patch(patch):
                continue
            if patch.std() < self.config.min_patch_std:
                continue
            patches.append(patch)
            if len(patches) >= max_patches:
                break
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
        all_normal_patches = []
        all_anomalous_patches = []
        all_normal_subjects = []
        all_anomalous_subjects = []
        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            subject_path = os.path.join(self.config.dataset_path, subject_dir)
            try:
                volume, segmentation = self.load_volume(subject_path)
                volume = self.normalize_volume(volume)
                normal_patches = self.extract_normal_patches(volume, segmentation)
                anomalous_patches = self.extract_anomalous_patches(volume, segmentation)
                all_normal_patches.extend(normal_patches)
                all_normal_subjects.extend([subject_dir] * len(normal_patches))
                all_anomalous_patches.extend(anomalous_patches)
                all_anomalous_subjects.extend([subject_dir] * len(anomalous_patches))
            except Exception:
                continue
        num_anomalous = len(all_anomalous_patches)
        max_normal = int(num_anomalous * self.config.max_normal_to_anomaly_ratio)
        if len(all_normal_patches) > max_normal and max_normal > 0:
            indices = np.random.choice(len(all_normal_patches), max_normal, replace=False)
            all_normal_patches = [all_normal_patches[i] for i in indices]
            all_normal_subjects = [all_normal_subjects[i] for i in indices]
        all_patches = all_normal_patches + all_anomalous_patches
        labels = [0] * len(all_normal_patches) + [1] * len(all_anomalous_patches)
        subjects = all_normal_subjects + all_anomalous_subjects
        return np.array(all_patches, dtype=np.float32), np.array(labels, dtype=np.int64), subjects


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerGenerator3D(nn.Module):
    def __init__(self, patch_size: int, latent_tokens: int, token_dim: int,
                 num_layers: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        # Encoder with retained features for skip connections
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
        )  # -> (32,16,16,16)
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )  # -> (64,8,8,8)
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
        )  # -> (128,4,4,4)
        self.enc4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.GELU(),
        )  # -> (256,4,4,4)
        # Transformer bottleneck over 4x4x4 tokens (64 tokens of dim 256)
        self.token_dim = token_dim
        self.pos = PositionalEncoding(256, max_len=64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=num_heads,
                                                   dim_feedforward=256 * mlp_ratio,
                                                   activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Decoder with U-Net-like skip connections: 4 -> 8 -> 16 -> 32
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
        )
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,32,32,32)
        x1 = self.enc1(x)  # (B,32,16,16,16)
        x2 = self.enc2(x1)  # (B,64,8,8,8)
        x3 = self.enc3(x2)  # (B,128,4,4,4)
        feat = self.enc4(x3)  # (B,256,4,4,4)
        b, c, d, h, w = feat.shape
        tokens = feat.view(b, c, d * h * w).permute(0, 2, 1).contiguous()  # (B,64,256)
        tokens = self.pos(tokens)
        tokens = self.transformer(tokens)
        feat2 = tokens.permute(0, 2, 1).contiguous().view(b, c, d, h, w)
        # Decode with skip connections
        y = self.up1(feat2)  # (B,128,8,8,8)
        y = torch.cat([y, x2], dim=1)  # (B,128+64,8,8,8)
        y = self.dec1(y)  # (B,128,8,8,8)
        y = self.up2(y)  # (B,64,16,16,16)
        y = torch.cat([y, x1], dim=1)  # (B,64+32,16,16,16)
        y = self.dec2(y)  # (B,64,16,16,16)
        y = self.up3(y)  # (B,32,32,32,32)
        out = self.out_conv(y)  # (B,1,32,32,32)
        return out


class PatchDiscriminator3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, 4, 2, 1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerGANAnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self.G = TransformerGenerator3D(
            patch_size=config.patch_size,
            latent_tokens=config.latent_tokens,
            token_dim=config.token_dim,
            num_layers=config.num_transformer_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
        ).to(config.device)
        self.D = PatchDiscriminator3D().to(config.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        self.scaler = GradScaler()

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        best_val = float('inf')
        patience = 0
        total_steps = self.config.num_epochs * max(1, len(train_loader))
        with tqdm(total=total_steps, desc="Training Progress") as pbar:
            for epoch in range(self.config.num_epochs):
                self.G.train(); self.D.train()
                epoch_loss = 0.0
                for real, labels in train_loader:
                    real = real.to(self.config.device)
                    labels = labels.to(self.config.device)
                    normal_mask = (labels.view(-1) == 0)
                    if normal_mask.sum() == 0:
                        pbar.update(1); continue
                    real_n = real[normal_mask]
                    bsz = real_n.size(0)

                    # Train D
                    self.opt_D.zero_grad(set_to_none=True)
                    with autocast():
                        fake = self.G(real_n)
                        d_real = self.D(torch.cat([real_n, real_n], dim=1))
                        d_fake = self.D(torch.cat([real_n, fake.detach()], dim=1))
                        valid = torch.ones_like(d_real)
                        fake_lbl = torch.zeros_like(d_fake)
                        loss_d = bce(d_real, valid) + bce(d_fake, fake_lbl)
                    self.scaler.scale(loss_d).backward()
                    self.scaler.step(self.opt_D)

                    # Train G
                    self.opt_G.zero_grad(set_to_none=True)
                    with autocast():
                        d_fake_for_g = self.D(torch.cat([real_n, fake], dim=1))
                        adv = bce(d_fake_for_g, valid)
                        rec = mse(fake, real_n)
                        loss_g = adv + self.config.rec_weight * rec
                    self.scaler.scale(loss_g).backward()
                    self.scaler.step(self.opt_G)
                    self.scaler.update()

                    epoch_loss += (loss_g.detach() + loss_d.detach()).item()
                    pbar.update(1)
                    pbar.set_postfix({
                        'Epoch': f'{epoch+1}/{self.config.num_epochs}',
                        'G_loss': f'{loss_g.item():.4f}',
                        'D_loss': f'{loss_d.item():.4f}'
                    })

                # Validation on normal data: reconstruction MSE
                self.G.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, labels in val_loader:
                        data = data.to(self.config.device)
                        labels = labels.to(self.config.device)
                        normal_mask = (labels.view(-1) == 0)
                        if normal_mask.sum() == 0:
                            continue
                        data_n = data[normal_mask]
                        fake = self.G(data_n)
                        val_loss += mse(fake, data_n).item()
                val_loss = val_loss / max(1, len(val_loader))
                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    torch.save(self.G.state_dict(), os.path.join(self.config.output_dir, 'best_generator.pth'))
                else:
                    patience += 1
                if patience >= self.config.early_stopping_patience:
                    pbar.write(f"Early stopping at epoch {epoch+1}")
                    break

    def compute_reconstruction_error(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.G.eval()
        errors = []
        labels_all = []
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Computing errors"):
                data = data.to(self.config.device)
                fake = self.G(data)
                per_voxel_sq = (fake - data) ** 2
                per_patch_mse = per_voxel_sq.mean(dim=(1, 2, 3, 4))
                errors.extend(per_patch_mse.cpu().numpy())
                labels_all.extend(labels.cpu().numpy().flatten())
        return np.array(errors), np.array(labels_all)

    def find_unsupervised_threshold(self, val_loader: DataLoader) -> float:
        errors, labels = self.compute_reconstruction_error(val_loader)
        normal_errors = errors  # val set contains normal-only
        return float(np.percentile(normal_errors, 95))

    def evaluate(self, test_loader: DataLoader, val_loader: DataLoader) -> Dict:
        gen_path = os.path.join(self.config.output_dir, 'best_generator.pth')
        if os.path.exists(gen_path):
            self.G.load_state_dict(torch.load(gen_path))
        # Threshold strictly from validation normals only
        threshold = self.find_unsupervised_threshold(val_loader)
        reconstruction_errors, true_labels = self.compute_reconstruction_error(test_loader)
        preds = (reconstruction_errors > threshold).astype(int)
        try:
            roc_auc = roc_auc_score(true_labels, reconstruction_errors)
            ap = average_precision_score(true_labels, reconstruction_errors)
        except Exception:
            roc_auc, ap = 0.0, 0.0
        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        bal_acc = (sens + spec) / 2
        # Debug prints (optional)
        if getattr(self.config, 'verbose', False):
            print("\nEVAL DIAGNOSTICS:")
            print(f"  Threshold (from val normals): {threshold:.6f}")
            print(f"  Test size: {len(true_labels)} | Normal: {np.sum(true_labels==0)} | Anomaly: {np.sum(true_labels==1)}")
            print(f"  Pred anomaly rate: {np.mean(preds):.3f}")
            print(f"  Error stats - normal: mean={reconstruction_errors[true_labels==0].mean():.4f}, std={reconstruction_errors[true_labels==0].std():.4f}")
            print(f"  Error stats - anomaly: mean={reconstruction_errors[true_labels==1].mean():.4f}, std={reconstruction_errors[true_labels==1].std():.4f}")
        return {
            'reconstruction_errors': reconstruction_errors,
            'true_labels': true_labels,
            'predictions': preds,
            'optimal_threshold': threshold,
            'roc_auc': roc_auc,
            'average_precision': ap,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'balanced_accuracy': bal_acc,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }


def save_curves(config: Config, results: Dict):
    y_true = results['true_labels']
    y_score = results['reconstruction_errors']
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC AUC={auc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(config.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP={ap:.4f}')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(config.output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("WARNING: Only one class present in test labels; skipping ROC/PR plots.")

    threshold = results['optimal_threshold']
    normal = results['reconstruction_errors'][results['true_labels'] == 0]
    anomaly = results['reconstruction_errors'][results['true_labels'] == 1]
    plt.figure(figsize=(12, 6))
    plt.hist(normal, bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(anomaly, bins=50, alpha=0.7, label='Anomaly', density=True)
    plt.axvline(threshold, color='k', linestyle='--', label=f'Th={threshold:.6f}')
    plt.xlabel('Reconstruction Error'); plt.ylabel('Density')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.title('Reconstruction Error Distributions')
    plt.savefig(os.path.join(config.output_dir, 'reconstruction_error_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='3D Transformer-GAN Anomaly Detection for BraTS')
    parser.add_argument('--num_subjects', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--patches_per_volume', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, default='transformer_gan_brats_results')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3)
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05)
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--tumor_core_only', action='store_true', help='Use only tumor core labels [1,4] as anomalies')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    config = Config()
    config.num_subjects = args.num_subjects
    config.patch_size = args.patch_size
    config.patches_per_volume = args.patches_per_volume
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.output_dir = args.output_dir
    config.dataset_path = args.dataset_path
    config.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    config.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    config.anomaly_labels = [1, 4] if args.tumor_core_only else args.anomaly_labels
    config.verbose = args.verbose
    os.makedirs(config.output_dir, exist_ok=True)

    if config.verbose:
        print("="*60)
        print("3D TRANSFORMER-GAN ANOMALY DETECTION FOR BRATS DATASET")
        print("="*60)
        print(f"Device: {config.device}")
        print(f"Dataset path: {config.dataset_path}")
        print(f"Output directory: {config.output_dir}")
        print(f"Patch size: {config.patch_size}^3")
        print(f"Patches per volume: {config.patches_per_volume}")
        print(f"Anomaly labels: {config.anomaly_labels}")

    # Step 1: Extract patches
    processor = BraTSDataProcessor(config)
    patches, labels, subjects = processor.process_dataset(config.num_subjects)
    if len(patches) == 0:
        print("Error: No patches extracted!")
        return

    # Subject-level split
    unique_subjects = list(set(subjects))
    np.random.shuffle(unique_subjects)
    n_subjects = len(unique_subjects)
    train_subjects = unique_subjects[:int(0.6 * n_subjects)]
    val_subjects = unique_subjects[int(0.6 * n_subjects):int(0.8 * n_subjects)]
    test_subjects = unique_subjects[int(0.8 * n_subjects):]
    train_idx = [i for i, s in enumerate(subjects) if s in train_subjects]
    val_idx = [i for i, s in enumerate(subjects) if s in val_subjects]
    test_idx = [i for i, s in enumerate(subjects) if s in test_subjects]
    X_train_all = patches[train_idx]; y_train_all = labels[train_idx]
    X_val_all = patches[val_idx]; y_val_all = labels[val_idx]
    X_test = patches[test_idx]; y_test = labels[test_idx]

    # Ensure test set has both classes; if not, move some subjects from val to test
    if len(np.unique(y_test)) < 2 and len(val_subjects) > 0:
        # Find subjects in val having the missing class
        missing_class = 1 if np.all(y_test == 0) else 0
        moved = False
        for subj in list(val_subjects):
            subj_indices = [i for i, s in enumerate(subjects) if s == subj]
            subj_labels = labels[subj_indices]
            if np.any(subj_labels == missing_class):
                # Move this subject to test
                val_subjects.remove(subj)
                test_subjects.append(subj)
                moved = True
                break
        if moved:
            # Recompute indices and splits
            val_idx = [i for i, s in enumerate(subjects) if s in val_subjects]
            test_idx = [i for i, s in enumerate(subjects) if s in test_subjects]
            X_val_all = patches[val_idx]; y_val_all = labels[val_idx]
            X_test = patches[test_idx]; y_test = labels[test_idx]
    train_mask = (y_train_all == 0)
    val_mask = (y_val_all == 0)
    X_train = X_train_all[train_mask]; y_train = y_train_all[train_mask]
    X_val = X_val_all[val_mask]; y_val = y_val_all[val_mask]

    # Sanity checks and split diagnostics
    assert len(set(train_subjects) & set(val_subjects)) == 0, "Subject overlap between train and val!"
    assert len(set(train_subjects) & set(test_subjects)) == 0, "Subject overlap between train and test!"
    assert len(set(val_subjects) & set(test_subjects)) == 0, "Subject overlap between val and test!"
    if config.verbose:
        print("\nSPLIT DIAGNOSTICS (subject-level, no leakage):")
        print(f"  Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)} | Test subjects: {len(test_subjects)}")
        print(f"  Train patches:  {len(X_train_all)} (normal used: {len(X_train)}) | Anom in train (excluded): {np.sum(y_train_all==1)}")
        print(f"  Val patches:    {len(X_val_all)} (normal used: {len(X_val)}) | Anom in val (excluded): {np.sum(y_val_all==1)}")
        print(f"  Test patches:   {len(X_test)}  | Normal: {np.sum(y_test==0)}, Anomaly: {np.sum(y_test==1)}")

    train_loader = DataLoader(BraTSPatchDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(BraTSPatchDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(BraTSPatchDataset(X_test, y_test), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    detector = TransformerGANAnomalyDetector(config)
    detector.train(train_loader, val_loader)
    results = detector.evaluate(test_loader, val_loader)

    # Print metrics directly to terminal
    print("\nEVALUATION (Transformer-GAN):")
    print("="*60)
    print(f"ROC AUC:           {results['roc_auc']:.4f}")
    print(f"Average Precision: {results['average_precision']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"F1 Score:          {results['f1_score']:.4f}")
    print(f"Precision:         {results['precision']:.4f}")
    print(f"Recall:            {results['recall']:.4f}")
    print(f"Threshold Used:    {results['optimal_threshold']:.6f}")

    # Save curves and report
    save_curves(config, results)
    results_file = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("TRULY UNSUPERVISED 3D Transformer-GAN Anomaly Detection Results\n")
        f.write("="*60 + "\n")
        f.write(f"ROC AUC:           {results['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {results['average_precision']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"F1 Score:          {results['f1_score']:.4f}\n")
        f.write(f"Precision:         {results['precision']:.4f}\n")
        f.write(f"Recall:            {results['recall']:.4f}\n")
        f.write(f"Threshold Used:    {results['optimal_threshold']:.6f}\n")
        f.write(f"Test size: {len(results['true_labels'])} | Normal: {int(np.sum(results['true_labels']==0))} | Anomaly: {int(np.sum(results['true_labels']==1))}\n")
        f.write(f"Pred anomaly rate: {float(np.mean(results['predictions'])):.3f}\n")
        f.write("="*60 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  Patch size: {config.patch_size}\n")
        f.write(f"  Patches per volume: {config.patches_per_volume}\n")
        f.write(f"  Learning rate: {config.learning_rate}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  Number of epochs: {config.num_epochs}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")

    total_time = time.time() - start_time
    mins = int((total_time % 3600) // 60); secs = int(total_time % 60)
    print(f"\nPipeline completed in {mins}m {secs}s. Results saved to: {results_file}")


if __name__ == "__main__":
    main()


