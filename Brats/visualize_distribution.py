#!/usr/bin/env python3
"""
Visualize the distribution of reconstruction errors between normal and tumor patches.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Import from autoencoder_anomaly.py
from autoencoder_anomaly import get_patient_data, create_patches, minmax, PatchAutoencoder

def load_model(model_path, device, latent_dim=32, patch_size=32):
    """Load pretrained autoencoder model"""
    model = PatchAutoencoder(patch_size=patch_size, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def extract_patches_and_scores(model, device, patient_data, patch_size=(32, 32, 32), overlap=0.25, min_tumor_ratio=0.05):
    """Extract patches and calculate reconstruction errors"""
    # Store normal and tumor patches
    normal_patches = []
    tumor_patches = []
    normal_errors = []
    tumor_errors = []
    normal_latent = []
    tumor_latent = []
    
    for patient_id, paths in tqdm(patient_data.items(), desc="Processing patients"):
        # Load data
        t1_img = np.array(np.load(paths['t1n'])) if paths['t1n'].endswith('.npy') else np.array(np.load(paths['t1n']))
        seg = np.array(np.load(paths['seg'])) if paths['seg'].endswith('.npy') else np.array(np.load(paths['seg']))
        
        # Create patches
        patches, has_tumor, tumor_ratios, _ = create_patches(
            t1_img, seg, patch_size=patch_size, overlap=overlap, min_tumor_ratio=min_tumor_ratio
        )
        
        # Process each patch
        for i, patch in enumerate(tqdm(patches, desc=f"Processing patches for {patient_id}", leave=False)):
            # Normalize patch
            patch_norm = minmax(patch)
            
            # Convert to tensor
            patch_tensor = torch.tensor(patch_norm, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get reconstruction and latent representation
            with torch.no_grad():
                reconstruction, latent = model(patch_tensor)
                
                # Calculate reconstruction error
                mse = torch.mean((reconstruction - patch_tensor.unsqueeze(1))**2).item()
            
            # Store results based on whether the patch contains tumor
            if has_tumor[i]:
                tumor_patches.append(patch_norm)
                tumor_errors.append(mse)
                tumor_latent.append(latent.cpu().numpy())
            else:
                normal_patches.append(patch_norm)
                normal_errors.append(mse)
                normal_latent.append(latent.cpu().numpy())
    
    # Convert to numpy arrays
    normal_errors = np.array(normal_errors)
    tumor_errors = np.array(tumor_errors)
    normal_latent = np.vstack(normal_latent)
    tumor_latent = np.vstack(tumor_latent) if tumor_latent else np.array([])
    
    return normal_patches, tumor_patches, normal_errors, tumor_errors, normal_latent, tumor_latent

def visualize_error_distribution(normal_errors, tumor_errors, save_path=None):
    """Visualize the distribution of reconstruction errors"""
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    sns.histplot(normal_errors, bins=30, alpha=0.5, label='Normal Patches', kde=True)
    sns.histplot(tumor_errors, bins=30, alpha=0.5, label='Tumor Patches', kde=True)
    
    # Add vertical line for median values
    plt.axvline(np.median(normal_errors), color='blue', linestyle='--', 
                label=f'Median Normal: {np.median(normal_errors):.4f}')
    plt.axvline(np.median(tumor_errors), color='orange', linestyle='--', 
                label=f'Median Tumor: {np.median(tumor_errors):.4f}')
    
    # Add text with statistics
    stats_text = (
        f"Normal: mean={np.mean(normal_errors):.4f}, std={np.std(normal_errors):.4f}, n={len(normal_errors)}\n"
        f"Tumor: mean={np.mean(tumor_errors):.4f}, std={np.std(tumor_errors):.4f}, n={len(tumor_errors)}\n"
        f"Overlap: {np.sum((normal_errors > np.min(tumor_errors)) & (normal_errors < np.max(tumor_errors))) / len(normal_errors):.2%} of normal patches"
    )
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.title('Distribution of Reconstruction Errors')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved error distribution plot to {save_path}")
    else:
        plt.show()

def visualize_latent_space(normal_latent, tumor_latent, method='tsne', save_path=None):
    """Visualize the latent space using dimensionality reduction"""
    if len(tumor_latent) == 0:
        print("No tumor latent vectors available for visualization")
        return
    
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization of Latent Space'
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
        title = 'PCA Visualization of Latent Space'
    
    # Combine data for reduction
    combined_latent = np.vstack([normal_latent, tumor_latent])
    
    # Apply dimensionality reduction
    print(f"Applying {method.upper()} to latent vectors...")
    reduced_latent = reducer.fit_transform(combined_latent)
    
    # Split back to normal and tumor
    normal_reduced = reduced_latent[:len(normal_latent)]
    tumor_reduced = reduced_latent[len(normal_latent):]
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.scatter(normal_reduced[:, 0], normal_reduced[:, 1], alpha=0.5, label='Normal Patches')
    plt.scatter(tumor_reduced[:, 0], tumor_reduced[:, 1], alpha=0.5, label='Tumor Patches')
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved latent space visualization to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize autoencoder results')
    parser.add_argument('--data_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData',
                        help='Path to BraTS dataset')
    parser.add_argument('--model_path', type=str, default='autoencoder_results/model.pth',
                        help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, default='autoencoder_results',
                        help='Output directory for visualizations')
    parser.add_argument('--max_patients', type=int, default=2,
                        help='Maximum number of patients to process')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension of the autoencoder')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Size of patches')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model if not exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}. Please train the model first using autoencoder_anomaly.py")
        sys.exit(1)
    
    # Load model
    model = load_model(args.model_path, device, args.latent_dim, args.patch_size)
    
    # Get patient data
    patient_data = get_patient_data(args.data_path, max_patients=args.max_patients)
    
    # Extract patches and calculate reconstruction errors
    normal_patches, tumor_patches, normal_errors, tumor_errors, normal_latent, tumor_latent = extract_patches_and_scores(
        model, device, patient_data, patch_size=(args.patch_size, args.patch_size, args.patch_size)
    )
    
    # Visualize error distributions
    visualize_error_distribution(
        normal_errors, tumor_errors, 
        save_path=os.path.join(args.output_dir, 'error_distribution.png')
    )
    
    # Visualize latent space
    visualize_latent_space(
        normal_latent, tumor_latent, method='tsne',
        save_path=os.path.join(args.output_dir, 'latent_tsne.png')
    )
    
    visualize_latent_space(
        normal_latent, tumor_latent, method='pca',
        save_path=os.path.join(args.output_dir, 'latent_pca.png')
    )

if __name__ == "__main__":
    main() 