#!/usr/bin/env python3
"""
Diffusion-based Anomaly Detection for BraTS (3D DDPM-style)

This script mirrors gan_brats.py's preprocessing, subject-level split, metrics,
and reporting, but uses a denoising diffusion probabilistic model (DDPM) with a
3D U-Net denoiser trained on NORMAL patches only. Anomaly score is computed as
the average denoising MSE across random timesteps (no supervision).

Outputs:
- Model: best_diffusion_unet.pth
- Plots: confusion matrix, ROC, PR, score histogram
- Report: diffusion_brats_results/evaluation_results.txt
"""

import os
import argparse
import random
import warnings
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc as sk_auc
)
from common.brats_preprocessing import BraTSPreprocessor

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class Config:
    """Configuration shared with gan_brats-style preprocessing and evaluation."""

    def __init__(self):
        # Dataset
        self.dataset_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
        self.output_dir = "diffusion_brats_results"

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

        # Segmentation labels
        self.anomaly_labels = [1, 2, 4]

        # Brain tissue quality
        self.min_brain_tissue_ratio = 0.3
        self.max_background_intensity = 0.1
        self.min_brain_mean_intensity = 0.1
        self.max_high_intensity_ratio = 0.7
        self.high_intensity_threshold = 0.9
        self.edge_margin = 8

        # Diffusion model params
        self.train_epochs = 100
        self.batch_size = 8
        self.num_workers = 4 if torch.cuda.is_available() else 0
        self.learning_rate = 2e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.0

        self.num_diffusion_steps = 200  # T
        self.base_channels = 32
        self.channel_multipliers = [1, 2, 4]  # levels: 32, 64, 128
        self.time_emb_dim = 128
        self.num_score_timesteps_eval = 10  # K for scoring

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Visualization
        self.verbose = False

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


# ---------------------------- Diffusion model -----------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class UNet3D(nn.Module):
    def __init__(self, base_ch: int = 32, channel_mults: List[int] = [1, 2, 4], time_emb_dim: int = 128):
        super().__init__()
        chs = [base_ch * m for m in channel_mults]
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.in_conv = nn.Conv3d(1, chs[0], kernel_size=3, padding=1)
        # Down
        self.down1 = ResidualBlock3D(chs[0], chs[0], time_emb_dim)
        self.down2 = ResidualBlock3D(chs[0], chs[1], time_emb_dim)
        self.downsamp1 = nn.Conv3d(chs[1], chs[1], kernel_size=4, stride=2, padding=1)
        self.down3 = ResidualBlock3D(chs[1], chs[1], time_emb_dim)
        self.down4 = ResidualBlock3D(chs[1], chs[2], time_emb_dim)
        self.downsamp2 = nn.Conv3d(chs[2], chs[2], kernel_size=4, stride=2, padding=1)
        # Bottleneck
        self.bot1 = ResidualBlock3D(chs[2], chs[2], time_emb_dim)
        self.bot2 = ResidualBlock3D(chs[2], chs[2], time_emb_dim)
        # Up
        self.upsamp1 = nn.ConvTranspose3d(chs[2], chs[2], kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock3D(chs[2] + chs[1], chs[1], time_emb_dim)
        self.up2 = ResidualBlock3D(chs[1], chs[1], time_emb_dim)
        self.upsamp2 = nn.ConvTranspose3d(chs[1], chs[1], kernel_size=4, stride=2, padding=1)
        self.up3 = ResidualBlock3D(chs[1] + chs[0], chs[0], time_emb_dim)
        self.up4 = ResidualBlock3D(chs[0], chs[0], time_emb_dim)
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(chs[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)
        d2 = self.down2(d1, t_emb)
        d2s = self.downsamp1(d2)
        d3 = self.down3(d2s, t_emb)
        d4 = self.down4(d3, t_emb)
        d4s = self.downsamp2(d4)
        b = self.bot2(self.bot1(d4s, t_emb), t_emb)
        u1s = self.upsamp1(b)
        u1 = self.up1(torch.cat([u1s, d4], dim=1), t_emb)
        u1 = self.up2(u1, t_emb)
        u2s = self.upsamp2(u1)
        u2 = self.up3(torch.cat([u2s, d2], dim=1), t_emb)
        u2 = self.up4(u2, t_emb)
        out = self.out_conv(self.out_act(self.out_norm(u2)))
        return out


class DDPMAnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.model = UNet3D(config.base_channels, config.channel_multipliers, config.time_emb_dim).to(self.device)
        self._build_diffusion_buffers(config.num_diffusion_steps)

    def _build_diffusion_buffers(self, T: int):
        # Linear beta schedule
        beta_start, beta_end = 1e-4, 2e-2
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer = lambda name, val: setattr(self, name, val.to(self.device))
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1.0 - alpha_bars))

    def _rand_timesteps(self, bsz: int) -> torch.Tensor:
        return torch.randint(0, self.config.num_diffusion_steps, (bsz,), device=self.device, dtype=torch.long)

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omab * noise

    def train_model(self, train_loader: DataLoader):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2), weight_decay=self.config.weight_decay)
        scaler = GradScaler()
        self.model.train()
        total_steps = self.config.train_epochs * max(1, len(train_loader))
        with tqdm(total=total_steps, desc="DDPM Training") as pbar:
            for epoch in range(self.config.train_epochs):
                for x, _ in train_loader:
                    x = x.to(self.device)
                    bsz = x.size(0)
                    t = self._rand_timesteps(bsz)
                    noise = torch.randn_like(x)
                    x_t = self._q_sample(x, t, noise)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast():
                        eps_pred = self.model(x_t, t.float())
                        loss = F.mse_loss(eps_pred, noise)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    pbar.update(1)
                    pbar.set_postfix({
                        'epoch': f"{epoch+1}/{self.config.train_epochs}",
                        'loss': f"{loss.item():.4f}"
                    })

        torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, 'best_diffusion_unet.pth'))

    @torch.no_grad()
    def compute_anomaly_scores(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        scores, labels = [], []
        K = self.config.num_score_timesteps_eval
        for x, y in tqdm(data_loader, desc="Scoring"):
            x = x.to(self.device)
            bsz = x.size(0)
            # Average denoising MSE over K random timesteps per sample
            batch_score = torch.zeros(bsz, device=self.device)
            for _ in range(K):
                t = self._rand_timesteps(bsz)
                noise = torch.randn_like(x)
                x_t = self._q_sample(x, t, noise)
                eps_pred = self.model(x_t, t.float())
                mse = torch.mean((eps_pred - noise) ** 2, dim=(1, 2, 3, 4))
                batch_score += mse
            batch_score = (batch_score / K).detach().cpu().numpy()
            scores.extend(batch_score)
            labels.extend(y.cpu().numpy().flatten())
        return np.array(scores), np.array(labels)

    def find_unsupervised_threshold(self, val_loader: DataLoader) -> float:
        if self.config.verbose:
            print("\nUNSUPERVISED THRESHOLD from normal validation scores (95th percentile)")
        scores, labels = self.compute_anomaly_scores(val_loader)
        normal_scores = scores[labels == 0]
        if len(normal_scores) == 0:
            return float(np.percentile(scores, 95)) if len(scores) else 0.0
        thr = float(np.percentile(normal_scores, 95))
        if self.config.verbose:
            print(f"Normal val scores: mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}, thr={thr:.6f}")
        return thr

    def evaluate(self, test_loader: DataLoader, val_loader: DataLoader) -> Dict:
        # Load checkpoint if present
        ckpt = os.path.join(self.config.output_dir, 'best_diffusion_unet.pth')
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        optimal_threshold = self.find_unsupervised_threshold(val_loader)
        scores, true_labels = self.compute_anomaly_scores(test_loader)
        preds = (scores > optimal_threshold).astype(int)

        try:
            fpr, tpr, _ = roc_curve(true_labels, scores)
            roc_auc = sk_auc(fpr, tpr)
            precision_curve, recall_curve, _ = precision_recall_curve(true_labels, scores)
            average_precision = sk_auc(recall_curve, precision_curve)
        except Exception:
            roc_auc, average_precision = 0.0, 0.0
        accuracy = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0
        dsc = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        results = {
            'anomaly_scores': scores,
            'true_labels': true_labels,
            'predictions': preds,
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
        fig.update_layout(title_text='<b>Confusion Matrix</b><br>(Count and Percentage)', title_x=0.5)
        fig.update_xaxes(side="bottom")
        output_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        try:
            fig.write_image(output_path, width=800, height=800, scale=2)
        except ValueError as e:
            print(f"ERROR: Could not save confusion matrix plot: {e}")

    def plot_roc_curve(self, true_labels: np.ndarray, scores: np.ndarray):
        fpr, tpr, _ = roc_curve(true_labels, scores)
        auc = sk_auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1, label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        plt.savefig(os.path.join(self.config.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curve(self, true_labels: np.ndarray, scores: np.ndarray):
        precision, recall, _ = precision_recall_curve(true_labels, scores)
        pr_auc_value = sk_auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc_value:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        plt.savefig(os.path.join(self.config.output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_score_histogram(self, scores: np.ndarray, true_labels: np.ndarray, optimal_threshold: float):
        normal_scores = scores[true_labels == 0]
        anomaly_scores = scores[true_labels == 1]
        plt.figure(figsize=(12, 6))
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {optimal_threshold:.6f}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config.output_dir, 'anomaly_score_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_report(self, results: Dict):
        self.plot_confusion_matrix(results['true_labels'], results['predictions'])
        self.plot_roc_curve(results['true_labels'], results['anomaly_scores'])
        self.plot_precision_recall_curve(results['true_labels'], results['anomaly_scores'])
        self.plot_score_histogram(results['anomaly_scores'], results['true_labels'], results['optimal_threshold'])
        print(f"\nAll visualizations saved to: {self.config.output_dir}")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='3D Diffusion (DDPM) Anomaly Detection for BraTS')
    parser.add_argument('--num_subjects', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--patches_per_volume', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_diffusion_steps', type=int, default=200)
    parser.add_argument('--score_timesteps_eval', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='diffusion_brats_results')
    parser.add_argument('--dataset_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData')
    parser.add_argument('--max_normal_to_anomaly_ratio', type=int, default=3)
    parser.add_argument('--min_tumor_ratio_in_patch', type=float, default=0.05)
    parser.add_argument('--anomaly_labels', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    config = Config()
    config.num_subjects = args.num_subjects
    config.patch_size = args.patch_size
    config.patches_per_volume = args.patches_per_volume
    config.batch_size = args.batch_size
    config.train_epochs = args.train_epochs
    config.learning_rate = args.lr
    config.num_diffusion_steps = args.num_diffusion_steps
    config.num_score_timesteps_eval = args.score_timesteps_eval
    config.output_dir = args.output_dir
    # Resolve dataset path relative to project root if given as relative path
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == 'src' else _script_dir
    config.dataset_path = args.dataset_path if os.path.isabs(args.dataset_path) else os.path.join(_project_root, args.dataset_path)
    config.max_normal_to_anomaly_ratio = args.max_normal_to_anomaly_ratio
    config.min_tumor_ratio_in_patch = args.min_tumor_ratio_in_patch
    config.anomaly_labels = args.anomaly_labels
    config.verbose = args.verbose
    os.makedirs(config.output_dir, exist_ok=True)

    label_names = {0: "Background/Normal", 1: "NCR/NET (Necrotic/Non-enhancing)", 2: "ED (Edema)", 4: "ET (Enhancing Tumor)"}
    anomaly_names = [f"{l}" for l in config.anomaly_labels]
    if config.verbose:
        print("="*60)
        print("3D DIFFUSION (DDPM) ANOMALY DETECTION FOR BRATS")
        print("="*60)
        print(f"Device: {config.device}")
        print(f"Dataset path: {config.dataset_path}")
        print(f"Output directory: {config.output_dir}")
        print(f"Patch size: {config.patch_size}^3  | Batch size: {config.batch_size}")
        print(f"Epochs: {config.train_epochs} | Diffusion steps: {config.num_diffusion_steps}")
        print(f"Anomaly labels: {anomaly_names}")
        print("="*60)
        print("\n1. Processing dataset and extracting patches...")
    else:
        print(f"3D DDPM | Anomaly labels: {anomaly_names} | Output: {config.output_dir}")

    processor = BraTSPreprocessor(config)
    patches, labels, subjects = processor.process_dataset(config.num_subjects)
    if len(patches) == 0:
        print("Error: No patches extracted! Check dataset path/structure.")
        return

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        print("ERROR: Need both normal and anomalous patches for evaluation.")
        return

    unique_subjects = list(set(subjects))
    np.random.shuffle(unique_subjects)
    n_subjects = len(unique_subjects)
    train_subjects = unique_subjects[:int(0.6 * n_subjects)]
    val_subjects = unique_subjects[int(0.6 * n_subjects):int(0.8 * n_subjects)]
    test_subjects = unique_subjects[int(0.8 * n_subjects):]

    train_indices = [i for i, s in enumerate(subjects) if s in train_subjects]
    val_indices = [i for i, s in enumerate(subjects) if s in val_subjects]
    test_indices = [i for i, s in enumerate(subjects) if s in test_subjects]

    X_train_all = patches[train_indices]
    y_train_all = labels[train_indices]
    X_val_all = patches[val_indices]
    y_val_all = labels[val_indices]
    X_test = patches[test_indices]
    y_test = labels[test_indices]

    train_normal_mask = (y_train_all == 0)
    val_normal_mask = (y_val_all == 0)
    X_train_normal = X_train_all[train_normal_mask]
    y_train_normal = y_train_all[train_normal_mask]
    X_val_normal = X_val_all[val_normal_mask]
    y_val_normal = y_val_all[val_normal_mask]

    if config.verbose:
        print(f"\n=== SUBJECT-LEVEL SPLIT (UNSUPERVISED) ===")
        print(f"Train NORMAL ONLY: {len(X_train_normal)} from {len(train_subjects)} subjects")
        print(f"Val NORMAL ONLY:   {len(X_val_normal)} from {len(val_subjects)} subjects")
        print(f"Test MIXED:        {len(X_test)} from {len(test_subjects)} subjects")
        print("="*60)

    assert len(set(train_subjects) & set(val_subjects)) == 0
    assert len(set(train_subjects) & set(test_subjects)) == 0
    assert len(set(val_subjects) & set(test_subjects)) == 0

    train_dataset = BraTSPatchDataset(X_train_normal, y_train_normal)
    val_dataset = BraTSPatchDataset(X_val_normal, y_val_normal)
    test_dataset = BraTSPatchDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    if config.verbose:
        print("\n2. Training DDPM on normal patches...")
    detector = DDPMAnomalyDetector(config)
    detector.train_model(train_loader)

    if config.verbose:
        print("\n3. Evaluating on test set...")
    results = detector.evaluate(test_loader, val_loader)

    visualizer = Visualizer(config)
    visualizer.create_summary_report(results)

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours}h")
    if minutes > 0:
        time_parts.append(f"{minutes}m")
    time_parts.append(f"{seconds}s")
    time_formatted = " ".join(time_parts)

    results_file = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("TRULY UNSUPERVISED 3D Diffusion (DDPM) Anomaly Detection Results\n")
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
        f.write("Configuration:\n")
        f.write(f"  Patch size: {config.patch_size}\n")
        f.write(f"  Patches per volume: {config.patches_per_volume}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  Train epochs: {config.train_epochs}\n")
        f.write(f"  Diffusion steps: {config.num_diffusion_steps}\n")
        f.write(f"  Scoring steps: {config.num_score_timesteps_eval}\n")
        f.write(f"  Training samples: {len(X_train_normal)}\n")
        f.write(f"  Validation samples: {len(X_val_normal)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Training subjects: {len(train_subjects)}\n")
        f.write(f"  Validation subjects: {len(val_subjects)}\n")
        f.write(f"  Test subjects: {len(test_subjects)}\n")
        f.write(f"  Anomaly labels used: {config.anomaly_labels}\n")
        f.write("="*60 + "\n")
        f.write("EXECUTION TIME:\n")
        f.write(f"  Total time: {time_formatted} ({total_time:.1f} seconds)\n")

    if config.verbose:
        print(f"\nResults saved to: {results_file}")
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("✓ Subject-level splitting, ✓ Unsupervised thresholding, ✓ No test label access")
        print("="*60)
        print(f"⏱️  TOTAL EXECUTION TIME: {time_formatted} ({total_time:.1f} seconds)")
        print("="*60)
    else:
        print(f"\nPipeline completed in {time_formatted}. Results saved to: {results_file}")


if __name__ == "__main__":
    main()


