#!/usr/bin/env python3
"""
BraTS Anomaly Detection using Anatomix Features and Autoencoder
================================================================

This script performs anomaly detection on the BraTS dataset by:
1. Extracting 3D patches from brain MRI scans
2. Labeling patches as normal/abnormal based on segmentation masks
3. Using Anatomix to extract features from patches
4. Training an autoencoder for anomaly detection
5. Evaluating performance with comprehensive metrics

Usage:
    python ae_brats.py --num_subjects 10 --patch_size 32 --stride 16
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
import sys
import contextlib
import warnings
warnings.filterwarnings('ignore')

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Disable memory statistics output
os.environ['PYTHONMALLOCSTATS'] = '0'

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
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            from anatomix.anatomix.model.network import Unet
except ImportError:
    install_anatomix()
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            from anatomix.anatomix.model.network import Unet

def minmax(arr, minclip=None, maxclip=None):
    """Normalize array to 0-1 range"""
    if not (minclip is None) & (maxclip is None):
        arr = np.clip(arr, minclip, maxclip)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def extract_3d_patches(image, seg_mask, patch_size=32, stride=16, normal_threshold=0.05):
    """
    Extract 3D patches from image and label them based on segmentation mask
    
    Args:
        image: 3D image array (H, W, D)
        seg_mask: 3D segmentation mask array (H, W, D)
        patch_size: Size of cubic patches to extract
        stride: Stride for patch extraction
        normal_threshold: Threshold for determining normal vs abnormal patches
    
    Returns:
        patches: List of 3D patches
        labels: List of labels (0=normal, 1=abnormal)
        coordinates: List of patch coordinates
    """
    h, w, d = image.shape
    patches = []
    labels = []
    coordinates = []
    
    if h < patch_size or w < patch_size or d < patch_size:
        print(f"Warning: Image dimensions {image.shape} are smaller than patch size {patch_size}")
        return patches, labels, coordinates
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            for k in range(0, d - patch_size + 1, stride):
                patch = image[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                mask_patch = seg_mask[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                
                # Only include patches that have brain tissue
                if np.mean(patch) > 0.05:
                    # Check if the patch contains abnormality
                    abnormal_voxels = np.sum(mask_patch > 0)
                    total_voxels = patch_size**3
                    abnormal_ratio = abnormal_voxels / total_voxels
                    
                    # Label the patch (0=normal, 1=abnormal)
                    label = 1 if abnormal_ratio > normal_threshold else 0
                    
                    patches.append(patch)
                    labels.append(label)
                    coordinates.append((i, j, k))
    
    return patches, labels, coordinates

def load_anatomix_model():
    """Load pre-trained anatomix model"""
    model = Unet(
        dimension=3,
        input_nc=1,
        output_nc=16,
        num_downs=4,
        ngf=16,
    ).to(device)
    
    # Check if weights exist
    weights_path = "anatomix/model-weights/anatomix.pth"
    if not os.path.exists(weights_path):
        if not os.path.exists("anatomix/model-weights"):
            os.makedirs("anatomix/model-weights", exist_ok=True)
        
        print("Downloading anatomix model weights...")
        os.system(f"wget -O {weights_path} https://github.com/neel-dey/anatomix/raw/main/model-weights/anatomix.pth")
    
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
    model.eval()
    return model

def extract_features(model, patches, batch_size=4):
    """Extract features from patches using anatomix model"""
    features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Extracting features"):
            batch_patches = patches[i:i+batch_size]
            batch_tensor = torch.tensor(np.array(batch_patches), dtype=torch.float32)
            batch_tensor = batch_tensor.unsqueeze(1).to(device)
            
            # Extract features
            batch_features = model(batch_tensor)
            # Global average pooling to get feature vector
            batch_features = torch.mean(batch_features, dim=[2, 3, 4])
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features)

def visualize_features(features, labels, sample_idx=0, save_path="feature_visualization"):
    """Visualize feature maps for a sample"""
    if len(features) == 0:
        return
    
    # Take first sample for visualization
    feature_sample = features[sample_idx]
    label = labels[sample_idx]
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Feature Visualization - Sample {sample_idx} (Label: {"Abnormal" if label else "Normal"})')
    
    for i in range(16):
        row, col = i // 4, i % 4
        axes[row, col].bar(range(len(feature_sample)), feature_sample)
        axes[row, col].set_title(f'Feature {i}')
        axes[row, col].set_ylim([feature_sample.min(), feature_sample.max()])
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature visualization saved as {save_path}_sample_{sample_idx}.png")

class Autoencoder(nn.Module):
    """3D Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim=16, hidden_dims=[32, 16, 8], latent_dim=4):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder(features, labels, epochs=100, batch_size=32, lr=0.001, hidden_dims=[32, 16, 8], latent_dim=4):
    """Train autoencoder on normal samples only"""
    
    # Use only normal samples for training
    normal_indices = np.where(np.array(labels) == 0)[0]
    normal_features = features[normal_indices]
    
    print(f"Training autoencoder on {len(normal_features)} normal samples")
    
    # Create model
    input_dim = features.shape[1]
    model = Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensor
    train_data = torch.tensor(normal_features, dtype=torch.float32).to(device)
    
    # Training loop
    losses = []
    model.train()
    
    for epoch in tqdm(range(epochs), desc="Training autoencoder"):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('autoencoder_losses.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training loss plot saved as autoencoder_losses.png")
    
    return model

def compute_anomaly_scores(model, features):
    """Compute anomaly scores using reconstruction error"""
    model.eval()
    anomaly_scores = []
    
    with torch.no_grad():
        for i in range(len(features)):
            feature_tensor = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
            reconstructed = model(feature_tensor)
            
            # Compute reconstruction error
            mse = torch.mean((feature_tensor - reconstructed) ** 2).item()
            anomaly_scores.append(mse)
    
    return np.array(anomaly_scores)

def evaluate_performance(true_labels, anomaly_scores):
    """Evaluate anomaly detection performance"""
    
    # Convert anomaly scores to binary predictions using threshold
    threshold = np.percentile(anomaly_scores, 75)  # Use 75th percentile as threshold
    pred_labels = (anomaly_scores > threshold).astype(int)
    
    # Compute metrics
    roc_auc = roc_auc_score(true_labels, anomaly_scores)
    avg_precision = average_precision_score(true_labels, anomaly_scores)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    results = {
        'ROC AUC': roc_auc,
        'Average Precision': avg_precision,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Threshold': threshold
    }
    
    return results, pred_labels

def visualize_results(true_labels, anomaly_scores, pred_labels, results):
    """Visualize anomaly detection results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
    axes[0, 0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f'ROC (AUC = {results["ROC AUC"]:.2f})')
    axes[0, 0].fill_between(fpr, tpr, step="pre", alpha=0.25, color="#aec7e8")
    axes[0, 0].plot([0, 1], [0, 1], linestyle='--', color="#888888", lw=1, label='Chance')
    axes[0, 0].set_xlabel('FPR')
    axes[0, 0].set_ylabel('TPR')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    
    # Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, auc as sk_auc
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, anomaly_scores)
    pr_auc_value = sk_auc(recall_curve, precision_curve)
    axes[0, 1].plot(recall_curve, precision_curve, color="#ff7f0e", lw=2, label=f'PR (AUC = {pr_auc_value:.2f})')
    axes[0, 1].fill_between(recall_curve, precision_curve, step="pre", alpha=0.25, color="#ffbb78")
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc='lower left')
    axes[0, 1].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    
    # Anomaly Score Distribution
    normal_scores = anomaly_scores[true_labels == 0]
    abnormal_scores = anomaly_scores[true_labels == 1]
    
    # Flatten the scores to ensure they are 1D
    normal_scores = normal_scores.flatten()
    abnormal_scores = abnormal_scores.flatten()
    
    # Ensure normal_scores and abnormal_scores are separate datasets
    axes[1, 0].hist([normal_scores, abnormal_scores], bins=30, alpha=0.7, 
                    label=['Normal', 'Abnormal'], color=['blue', 'red'])
    axes[1, 0].axvline(results['Threshold'], color='black', linestyle='--', label='Threshold')
    axes[1, 0].set_xlabel('Anomaly Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Anomaly Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    tick_marks = np.arange(2)
    axes[1, 1].set_xticks(tick_marks)
    axes[1, 1].set_yticks(tick_marks)
    axes[1, 1].set_xticklabels(['Normal', 'Abnormal'])
    axes[1, 1].set_yticklabels(['Normal', 'Abnormal'])
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
    
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Results visualization saved as anomaly_detection_results.png")

def load_brats_data(data_dir, num_subjects=10):
    """Load BraTS dataset"""
    
    # Get all subject directories
    subject_dirs = sorted(glob(os.path.join(data_dir, "BraTS-GLI-*")))
    
    if len(subject_dirs) == 0:
        raise ValueError(f"No BraTS subjects found in {data_dir}")
    
    # Limit to specified number of subjects
    subject_dirs = subject_dirs[:num_subjects]
    print(f"Loading {len(subject_dirs)} subjects from BraTS dataset")
    
    all_patches = []
    all_labels = []
    all_coordinates = []
    
    for i, subject_dir in enumerate(tqdm(subject_dirs, desc="Loading subjects")):
        subject_name = os.path.basename(subject_dir)
        
        # Load T1-weighted image (t1n = T1 native)
        t1_path = os.path.join(subject_dir, f"{subject_name}-t1n.nii.gz")
        seg_path = os.path.join(subject_dir, f"{subject_name}-seg.nii.gz")
        
        if not os.path.exists(t1_path) or not os.path.exists(seg_path):
            print(f"Missing files for subject {subject_name}, skipping...")
            continue
        
        # Load and normalize image
        t1_img = nib.load(t1_path).get_fdata()
        t1_img = minmax(t1_img)
        
        # Load segmentation mask
        seg_mask = nib.load(seg_path).get_fdata()
        
        # Extract patches
        patches, labels, coordinates = extract_3d_patches(
            t1_img, seg_mask, patch_size=32, stride=16, normal_threshold=0.05
        )
        
        print(f"Subject {i+1}: {len(patches)} patches extracted "
              f"({np.sum(labels)} abnormal, {len(labels) - np.sum(labels)} normal)")
        
        all_patches.extend(patches)
        all_labels.extend(labels)
        all_coordinates.extend(coordinates)
    
    return all_patches, all_labels, all_coordinates

def main():
    parser = argparse.ArgumentParser(description='BraTS Anomaly Detection with Anatomix and Autoencoder')
    parser.add_argument('--num_subjects', type=int, default=10, 
                       help='Number of subjects to use from dataset')
    parser.add_argument('--patch_size', type=int, default=32,
                       help='Size of 3D patches to extract')
    parser.add_argument('--stride', type=int, default=16,
                       help='Stride for patch extraction')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Ratio of data to use for testing')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs for autoencoder')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for autoencoder')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32, 16],
                       help='List of hidden layer dimensions for the autoencoder')
    parser.add_argument('--latent_dim', type=int, default=4,
                       help='Dimension of the latent space in the autoencoder')
    
    args = parser.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.test_ratio != 1.0:
        raise ValueError("Train and test ratios must sum to 1.0")
    
    print("="*60)
    print("BraTS Anomaly Detection with Anatomix and Autoencoder")
    print("="*60)
    print(f"Number of subjects: {args.num_subjects}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride}")
    print(f"Train/Test split: {args.train_ratio:.1f}/{args.test_ratio:.1f}")
    print(f"Device: {device}")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Load BraTS data and extract patches
    print("\n1. Loading BraTS data and extracting patches...")
    data_dir = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
    patches, labels, coordinates = load_brats_data(data_dir, args.num_subjects)
    
    print(f"Total patches extracted: {len(patches)}")
    print(f"Normal patches: {len(labels) - sum(labels)}")
    print(f"Abnormal patches: {sum(labels)}")
    
    if len(patches) == 0:
        print("No patches extracted. Exiting...")
        return
    
    # 2. Load Anatomix model and extract features
    print("\n2. Loading Anatomix model and extracting features...")
    anatomix_model = load_anatomix_model()
    features = extract_features(anatomix_model, patches, batch_size=4)
    
    print(f"Features shape: {features.shape}")
    
    # Visualize features for first few samples
    print("\n3. Visualizing features...")
    for i in range(min(3, len(features))):
        visualize_features(features, labels, sample_idx=i)
    
    # 3. Split data into train and test sets
    print("\n4. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=args.test_ratio, 
        random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training normal/abnormal: {len(y_train) - sum(y_train)}/{sum(y_train)}")
    print(f"Test normal/abnormal: {len(y_test) - sum(y_test)}/{sum(y_test)}")
    
    # 4. Train autoencoder
    print("\n5. Training autoencoder...")
    autoencoder = train_autoencoder(
        X_train, y_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim
    )
    
    # Save trained model
    torch.save(autoencoder.state_dict(), 'best_autoencoder.pth')
    print("Autoencoder model saved as best_autoencoder.pth")
    
    # 5. Compute anomaly scores on test set
    print("\n6. Computing anomaly scores on test set...")
    test_anomaly_scores = compute_anomaly_scores(autoencoder, X_test)
    
    # 6. Evaluate performance
    print("\n7. Evaluating performance...")
    results, pred_labels = evaluate_performance(y_test, test_anomaly_scores)
    
    print("\nPerformance Results:")
    print("-" * 40)
    for metric, value in results.items():
        if metric != 'Threshold':
            print(f"{metric:20s}: {value:.4f}")
        else:
            print(f"{metric:20s}: {value:.6f}")
    
    # 7. Visualize results
    print("\n8. Creating visualizations...")
    visualize_results(y_test, test_anomaly_scores, pred_labels, results)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Subjects processed: {args.num_subjects}")
    print(f"Total patches: {len(patches)}")
    print(f"Features extracted: {features.shape}")
    print(f"Best ROC AUC: {results['ROC AUC']:.4f}")
    print(f"Best F1 Score: {results['F1 Score']:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
