#!/usr/bin/env python3
"""
Anomaly detection on BraTS data using a patch-based approach with an Autoencoder.
This script replaces the KNN approach with an Autoencoder for anomaly detection.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import time
import argparse
from scipy.ndimage import zoom

# Suppress warnings
warnings.filterwarnings("ignore")

def minmax(arr, minclip=None, maxclip=None):
    """Normalize array to 0-1 range, with optional clipping"""
    if not ((minclip is None) and (maxclip is None)):
        arr = np.clip(arr, minclip, maxclip)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr.astype(np.float32)  # Ensure float32 output

class PatchAutoencoder(nn.Module):
    """Convolutional Autoencoder for 3D brain patches"""
    def __init__(self, patch_size=32, latent_dim=256):
        super(PatchAutoencoder, self).__init__()
        
        self.patch_size = patch_size
        
        # Encoder - convolutional path
        self.encoder = nn.Sequential(
            # Input: batch_size x 1 x 32 x 32 x 32
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2),  # 16x16x16
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2),  # 8x8x8
            
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2),  # 4x4x4
            
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=2),  # 2x2x2
            
            nn.Flatten(),  # 256*2*2*2 = 2048
            nn.Linear(2048, latent_dim),
        )
        
        # Decoder - transpose convolution path
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder_conv = nn.Sequential(
            # Input: batch_size x 256 x 2 x 2 x 2
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),  # 4x4x4
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),  # 8x8x8
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),  # 16x16x16
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose3d(32, 1, kernel_size=2, stride=2),  # 32x32x32
            nn.Sigmoid(),  # Output between 0 and 1
        )
        
    def forward(self, x):
        # Ensure input has channel dimension
        if len(x.shape) == 4:  # batch, D, H, W
            x = x.unsqueeze(1)  # Add channel dimension: batch, C, D, H, W
            
        # Encode
        z = self.encoder(x)
        
        # Decode
        y = self.decoder_linear(z)
        y = y.view(-1, 256, 2, 2, 2)  # Reshape to match 3D shape
        reconstruction = self.decoder_conv(y)
        
        return reconstruction, z
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for a given input"""
        # Ensure input has channel dimension
        if len(x.shape) == 4:  # batch, D, H, W
            x = x.unsqueeze(1)  # Add channel dimension: batch, C, D, H, W
            
        # Forward pass
        with torch.no_grad():
            reconstruction, _ = self(x)
            
        # Calculate mean squared error
        mse = torch.mean((reconstruction - x)**2, dim=(1, 2, 3, 4))
        
        return mse

def setup_model():
    """Set up the Anatomix model for feature extraction"""
    anatomix_path = os.path.abspath('./anatomix')
    if not os.path.exists(anatomix_path):
        print("Cloning Anatomix repository...")
        os.system('git clone https://github.com/neel-dey/anatomix.git')
    
    # Add anatomix to Python path
    if anatomix_path not in sys.path:
        sys.path.insert(0, anatomix_path)
    
    # Import Anatomix directly from file paths
    try:
        # Add anatomix module to path
        sys.path.insert(0, os.path.join(anatomix_path, 'anatomix'))
        from model.network import Unet
    except ImportError as e:
        print(f"Error importing Unet: {e}")
        print("Falling back to direct file import")
        # Try direct import path
        model_path = os.path.join(anatomix_path, 'anatomix', 'model')
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        from network import Unet
    
    # Create device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = Unet(
        dimension=3,
        input_nc=1,
        output_nc=16,
        num_downs=4,
        ngf=16,
    ).to(device)
    
    # Load weights
    weights_path = os.path.join(anatomix_path, "model-weights", "anatomix.pth")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    
    return model, device

def create_patches(image, segmentation=None, patch_size=(32, 32, 32), overlap=0.5, 
                   min_tumor_ratio_for_eval=0.05, skull_margin_voxels=5):
    """
    Create patches from 3D image.
    Filters patches for AE training based on tumor masks and proximity to skull edge.
    Labels patches as tumor/normal for evaluation purposes.

    Returns:
        patches_list: List of all extracted patch data (np.array).
        is_candidate_for_training_list: List of booleans, True if patch is suitable for AE training.
        has_tumor_label_list: List of booleans, True if patch contains tumor (for evaluation).
        patch_locations_list: List of (x,y,z) coordinates for each patch.
    """
    patches_list = []
    is_candidate_for_training_list = []
    has_tumor_label_list = []
    patch_locations_list = []
    
    stride = [int(p * (1 - overlap)) for p in patch_size]
    
    combined_tumor_mask_map = None
    if segmentation is not None:
        # BraTS labels: 1 (NCR/NET), 2 (ED), 4 (ET) are considered tumor regions
        combined_tumor_mask_map = (segmentation == 1) | (segmentation == 2) | (segmentation == 4)

    for x_coord in range(0, image.shape[0] - patch_size[0] + 1, stride[0]):
        for y_coord in range(0, image.shape[1] - patch_size[1] + 1, stride[1]):
            for z_coord in range(0, image.shape[2] - patch_size[2] + 1, stride[2]):
                patch_data = image[x_coord:x_coord+patch_size[0], y_coord:y_coord+patch_size[1], z_coord:z_coord+patch_size[2]]
                
                if patch_data.shape != tuple(patch_size):
                    continue

                is_valid_for_ae_training = True # Assume valid initially

                # 1. Tumor mask filter for AE training patches
                # AE training patches must NOT overlap with any defined tumor region.
                if combined_tumor_mask_map is not None:
                    seg_patch_for_filter = combined_tumor_mask_map[x_coord:x_coord+patch_size[0], y_coord:y_coord+patch_size[1], z_coord:z_coord+patch_size[2]]
                    if np.any(seg_patch_for_filter):
                        is_valid_for_ae_training = False
                
                # 2. Geometric ROI filter (skull edge) for AE training patches
                too_close_to_start = (x_coord < skull_margin_voxels or 
                                      y_coord < skull_margin_voxels or 
                                      z_coord < skull_margin_voxels)
                too_close_to_end = (x_coord + patch_size[0] > image.shape[0] - skull_margin_voxels or
                                    y_coord + patch_size[1] > image.shape[1] - skull_margin_voxels or
                                    z_coord + patch_size[2] > image.shape[2] - skull_margin_voxels)

                if too_close_to_start or too_close_to_end:
                    is_valid_for_ae_training = False
                
                # Determine `has_tumor_label` for evaluation purposes
                current_patch_is_tumor_for_eval = False
                if combined_tumor_mask_map is not None:
                    seg_patch_for_eval = combined_tumor_mask_map[x_coord:x_coord+patch_size[0], y_coord:y_coord+patch_size[1], z_coord:z_coord+patch_size[2]]
                    tumor_ratio = np.count_nonzero(seg_patch_for_eval) / seg_patch_for_eval.size
                    if tumor_ratio >= min_tumor_ratio_for_eval:
                        current_patch_is_tumor_for_eval = True
                
                patches_list.append(patch_data)
                patch_locations_list.append((x_coord, y_coord, z_coord))
                is_candidate_for_training_list.append(is_valid_for_ae_training)
                has_tumor_label_list.append(current_patch_is_tumor_for_eval)
    
    return patches_list, is_candidate_for_training_list, has_tumor_label_list, patch_locations_list

def get_patient_data(data_path, max_patients=10):
    """Get a subset of patient data"""
    # Get list of patient directories
    patient_dirs = sorted(glob(os.path.join(data_path, 'BraTS-GLI-*')))
    
    if not patient_dirs:
        raise FileNotFoundError(f"No BraTS patient directories found in {data_path}")
    
    # Limit to max_patients
    patient_dirs = patient_dirs[:max_patients]
    print(f"Using {len(patient_dirs)} patient directories")
    
    # Get data for each patient
    patient_data = {}
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        
        # Get modality files
        t1n_file = glob(os.path.join(patient_dir, '*-t1n.nii.gz'))[0]
        t1c_file = glob(os.path.join(patient_dir, '*-t1c.nii.gz'))[0]
        
        # Get segmentation
        seg_file = glob(os.path.join(patient_dir, '*-seg.nii.gz'))[0]
        
        patient_data[patient_id] = {
            't1n': t1n_file,
            't1c': t1c_file,
            'seg': seg_file
        }
    
    return patient_data

def run_autoencoder_anomaly_detection(device, patient_data, patch_size=(32, 32, 32), 
                                     overlap=0.5, latent_dim=128, epochs=50, 
                                     batch_size=32, min_tumor_ratio=0.05,
                                     test_split=0.2, skull_margin_voxels=5):
    """Run autoencoder-based anomaly detection on patient data"""
    # Store candidate patches for AE training and all patches for evaluation
    collected_ae_training_candidates = [] 
    collected_eval_patches_details = [] # Stores dicts: {'patch': data, 'is_tumor': bool, 'patient_id': id, 'is_ae_candidate': bool}
    
    all_patient_patch_info = {} # For heatmap reconstruction and reference
    
    print("Extracting and filtering patches...")
    for patient_id, paths in tqdm(patient_data.items()):
        # Load data
        t1_img = nib.load(paths['t1n']).get_fdata()
        seg_img = nib.load(paths['seg']).get_fdata()
        
        current_patches, current_is_candidate_for_training, current_has_tumor_label, current_locations = create_patches(
            t1_img, seg_img, 
            patch_size=patch_size, 
            overlap=overlap, 
            min_tumor_ratio_for_eval=min_tumor_ratio,
            skull_margin_voxels=skull_margin_voxels
        )
        
        all_patient_patch_info[patient_id] = {
            'patches': current_patches,
            'locations': current_locations,
            'shape': t1_img.shape,
            'has_tumor_label_info': current_has_tumor_label, 
            'is_training_candidate_info': current_is_candidate_for_training
        }
        
        for i, patch_data in enumerate(current_patches):
            if current_is_candidate_for_training[i]:
                collected_ae_training_candidates.append(patch_data)
            
            collected_eval_patches_details.append({
                'patch': patch_data,
                'is_tumor': current_has_tumor_label[i],
                'patient_id': patient_id,
                'is_ae_candidate': current_is_candidate_for_training[i]
            })
    
    np.random.seed(42)
    n_total_ae_candidates = len(collected_ae_training_candidates)

    if n_total_ae_candidates == 0:
        raise ValueError("No patches met the criteria for AE training. Check filters, data, or skull_margin_voxels.")

    shuffled_candidate_indices = np.arange(n_total_ae_candidates)
    np.random.shuffle(shuffled_candidate_indices)
    
    n_test_normal_from_candidates = int(n_total_ae_candidates * test_split)
    
    if test_split > 0 and n_test_normal_from_candidates == 0 and n_total_ae_candidates > 0:
        n_test_normal_from_candidates = 1 
    if n_test_normal_from_candidates >= n_total_ae_candidates and n_total_ae_candidates > 0:
        n_test_normal_from_candidates = max(0, n_total_ae_candidates - 1)

    ae_final_train_indices = shuffled_candidate_indices[n_test_normal_from_candidates:]
    test_normal_indices_from_candidates = shuffled_candidate_indices[:n_test_normal_from_candidates]
    
    train_patches_for_ae = [collected_ae_training_candidates[i] for i in ae_final_train_indices]
    
    if not train_patches_for_ae:
        raise ValueError(f"No patches selected for AE training after split (total candidates: {n_total_ae_candidates}, test_split: {test_split}). Check parameters.")
        
    print(f"Training autoencoder on {len(train_patches_for_ae)} strictly normal/non-artifact patches.")
    
    test_normal_patches = [collected_ae_training_candidates[i] for i in test_normal_indices_from_candidates]
    test_tumor_patches = [item['patch'] for item in collected_eval_patches_details if item['is_tumor']]
    
    print(f"Testing on {len(test_normal_patches)} normal patches (held out from AE candidates) " +
          f"and {len(test_tumor_patches)} tumor patches.")

    # Convert training patches to torch tensors in batches
    print("Preparing data for autoencoder training...")
    
    # Process patches in smaller batches to avoid memory issues
    batch_size_prep = 1000  # Adjust this based on available memory
    n_patches = len(train_patches_for_ae)
    n_batches = (n_patches + batch_size_prep - 1) // batch_size_prep
    
    train_patches_tensor = []
    for i in tqdm(range(n_batches), desc="Normalizing patches"):
        start_idx = i * batch_size_prep
        end_idx = min((i + 1) * batch_size_prep, n_patches)
        batch_patches = train_patches_for_ae[start_idx:end_idx]
        
        # Normalize batch
        batch_normalized = [minmax(patch) for patch in batch_patches]
        batch_tensor = torch.tensor(np.array(batch_normalized), dtype=torch.float32).to(device)
        train_patches_tensor.append(batch_tensor)
    
    # Concatenate all batches
    train_patches_tensor = torch.cat(train_patches_tensor, dim=0)
    
    # Create dataset and dataloader
    dataset = TensorDataset(train_patches_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create autoencoder
    autoencoder = PatchAutoencoder(patch_size=patch_size[0], latent_dim=latent_dim).to(device)
    
    # Train autoencoder
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    # Training loop
    epoch_pbar = tqdm(total=epochs, desc="Training autoencoder")
    for epoch in range(epochs):
        total_loss = 0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in batch_pbar:
            # Get batch
            x = batch[0]
            
            # Forward pass
            optimizer.zero_grad()
            reconstruction, _ = autoencoder(x)
            
            # Calculate loss
            loss = criterion(reconstruction, x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        # Update overall progress
        avg_loss = total_loss/len(dataloader)
        epoch_pbar.set_postfix({'loss': f"{avg_loss:.6f}"})
        epoch_pbar.update(1)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    epoch_pbar.close()
    
    print("Autoencoder training complete")
    
    # Put autoencoder in eval mode
    autoencoder.eval()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    
    # Combine test normal patches and tumor patches for evaluation
    eval_test_patches = test_normal_patches + test_tumor_patches
    eval_test_labels = [0] * len(test_normal_patches) + [1] * len(test_tumor_patches)
    
    test_scores = []

    if not eval_test_patches:
        print("Warning: Test set for patch-level evaluation is empty. Skipping patch-level metrics.")
        test_auc, test_ap, optimal_threshold = 0.0, 0.0, 0.0
        precision, recall, thresholds = np.array([0]), np.array([0]), np.array([0])
    else:
        # Process test patches in batches
        batch_size_eval = 1000  # Adjust based on available memory
        n_test_patches = len(eval_test_patches)
        n_test_batches = (n_test_patches + batch_size_eval - 1) // batch_size_eval
        
        with torch.no_grad():
            for i in tqdm(range(n_test_batches), desc="Evaluating test patches"):
                start_idx = i * batch_size_eval
                end_idx = min((i + 1) * batch_size_eval, n_test_patches)
                batch_patches = eval_test_patches[start_idx:end_idx]
                
                # Normalize and process batch
                batch_normalized = [minmax(patch) for patch in batch_patches]
                batch_tensor = torch.tensor(np.array(batch_normalized), dtype=torch.float32).to(device)
                
                # Get reconstruction error
                reconstruction, _ = autoencoder(batch_tensor)
                batch_mse = torch.mean((reconstruction - batch_tensor.unsqueeze(1))**2, dim=(1, 2, 3, 4))
                test_scores.extend(batch_mse.cpu().numpy())
        
        if len(np.unique(eval_test_labels)) < 2:
            print(f"Warning: Test set for patch-level evaluation contains only one class. Labels: {np.unique(eval_test_labels)}. Metrics might be skewed or invalid.")
            test_auc = np.nan
            test_ap = np.nan
            optimal_threshold = np.percentile(test_scores, 95) if test_scores else 0.0
            precision, recall, thresholds = precision_recall_curve(eval_test_labels, test_scores, pos_label=1 if 1 in eval_test_labels else 0)
            if not thresholds.size: thresholds = np.array([optimal_threshold])
        else:
            test_auc = roc_auc_score(eval_test_labels, test_scores)
            test_ap = average_precision_score(eval_test_labels, test_scores)
            precision, recall, thresholds_pr = precision_recall_curve(eval_test_labels, test_scores)
            thresholds_pr = np.append(thresholds_pr, thresholds_pr[-1]+1e-6 if len(thresholds_pr)>0 else np.array([0]))

            f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
            if len(f1_scores) > 0:
                optimal_idx = np.argmax(f1_scores)
                if optimal_idx < len(thresholds_pr):
                    optimal_threshold = thresholds_pr[optimal_idx]
                elif len(thresholds_pr) > 0:
                    optimal_threshold = thresholds_pr[-1]
                else:
                    optimal_threshold = np.percentile(test_scores, 95) if test_scores else 0.0
            else:
                optimal_idx = 0
                optimal_threshold = np.percentile(test_scores, 95) if test_scores else 0.0
                print("Warning: F1 scores array was empty. Using percentile-based threshold.")

    print(f"Test Set Patch-Level Metrics:")
    print(f"AUC: {test_auc:.4f}")
    print(f"Average Precision: {test_ap:.4f}")
    if 'optimal_idx' in locals() and len(f1_scores) > optimal_idx:
        print(f"Optimal threshold: {optimal_threshold:.6f} (F1: {f1_scores[optimal_idx]:.4f})")
    else:
        print(f"Optimal threshold: {optimal_threshold:.6f} (F1: N/A due to empty or invalid F1 scores)")
    
    # Create heatmaps for each patient
    print("Computing patient-level anomaly scores...")
    results = {}
    
    for patient_id, patch_info in tqdm(all_patient_patch_info.items()):
        # Initialize anomaly score map (same shape as original image)
        anomaly_score_map = np.zeros(patch_info['shape'])
        count_map = np.zeros(patch_info['shape'])  # To average overlapping patches
        
        # Process patches in batches
        batch_size_heatmap = 1000  # Adjust based on available memory
        n_patches = len(patch_info['patches'])
        n_batches = (n_patches + batch_size_heatmap - 1) // batch_size_heatmap
        
        for i in range(n_batches):
            start_idx = i * batch_size_heatmap
            end_idx = min((i + 1) * batch_size_heatmap, n_patches)
            batch_patches = patch_info['patches'][start_idx:end_idx]
            batch_locations = patch_info['locations'][start_idx:end_idx]
            
            # Normalize and process batch
            batch_normalized = [minmax(patch) for patch in batch_patches]
            batch_tensor = torch.tensor(np.array(batch_normalized), dtype=torch.float32).to(device)
            
            # Compute reconstruction error
            with torch.no_grad():
                reconstruction, _ = autoencoder(batch_tensor)
                batch_mse = torch.mean((reconstruction - batch_tensor.unsqueeze(1))**2, dim=(1, 2, 3, 4))
            
            # Add scores to the map
            for j, (x, y, z) in enumerate(batch_locations):
                mse = batch_mse[j].item()
                anomaly_score_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += mse
                count_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += 1
        
        # Average the anomaly scores where patches overlapped
        count_map[count_map == 0] = 1  # Avoid division by zero
        anomaly_score_map /= count_map
        
        # Create ground truth from combined tumor segmentation
        seg_data_for_gt = nib.load(patient_data[patient_id]['seg']).get_fdata()
        combined_tumor_gt_mask = (seg_data_for_gt == 1) | (seg_data_for_gt == 2) | (seg_data_for_gt == 4)
        ground_truth_map = combined_tumor_gt_mask.astype(int)
        
        results[patient_id] = {
            'anomaly_score': anomaly_score_map,
            'seg': ground_truth_map,
            'has_tumor_patches_count': sum(patch_info['has_tumor_label_info']),
            'candidate_training_patches_count': sum(patch_info['is_training_candidate_info']),
            'total_patches_count': len(patch_info['patches']),
            'min_tumor_ratio_setting': min_tumor_ratio,
            'optimal_threshold': optimal_threshold
        }
    
    return results, test_auc, test_ap, optimal_threshold

def evaluate_results(results):
    """Evaluate anomaly detection results"""
    aucs = []
    
    for patient_id, result in results.items():
        # Use segmentation as ground truth (1 for tumor, 0 for normal)
        ground_truth = result['seg'].flatten()
        
        # If no tumor, skip
        if np.sum(ground_truth) == 0:
            continue
        
        # Get anomaly score
        pred = result['anomaly_score'].flatten()
        
        # Calculate AUC
        try:
            auc = roc_auc_score(ground_truth, pred)
            aucs.append(auc)
            print(f"Patient {patient_id}: AUC = {auc:.4f}, " +
                  f"Tumor patches (eval def): {result['has_tumor_patches_count']}, " +
                  f"Candidate training patches: {result['candidate_training_patches_count']}, " +
                  f"Total patches: {result['total_patches_count']}")
        except Exception as e:
            print(f"Error calculating AUC for patient {patient_id}: {e}")
    
    if aucs:
        print(f"\nMean AUC: {np.mean(aucs):.4f}")
    
    return aucs

def visualize_results(results, patient_data, output_dir=None):
    """Visualize anomaly detection results"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for patient_id, result in results.items():
        if 'anomaly_score' in result:
            # Load original T1 image
            t1_img = nib.load(patient_data[patient_id]['t1n']).get_fdata()
            
            # Find slice with maximum tumor
            tumor_sum_per_slice = np.sum(result['seg'], axis=(0, 2))
            slice_idx = np.argmax(tumor_sum_per_slice)
            
            # Create figure
            plt.figure(figsize=(20, 5))
            
            # Original image
            plt.subplot(1, 4, 1)
            plt.imshow(np.rot90(t1_img[:, slice_idx, :]), cmap='gray')
            plt.title(f'Original T1 (Slice {slice_idx})')
            plt.axis('off')
            
            # Segmentation
            plt.subplot(1, 4, 2)
            plt.imshow(np.rot90(t1_img[:, slice_idx, :]), cmap='gray')
            seg_overlay = np.rot90(result['seg'][:, slice_idx, :])
            plt.imshow(seg_overlay, alpha=0.6, cmap='hot')
            plt.title('Tumor Segmentation')
            plt.axis('off')
            
            # Anomaly score
            plt.subplot(1, 4, 3)
            plt.imshow(np.rot90(t1_img[:, slice_idx, :]), cmap='gray')
            anomaly_overlay = np.rot90(result['anomaly_score'][:, slice_idx, :])
            plt.imshow(anomaly_overlay, alpha=0.6, cmap='hot')
            plt.title('Anomaly Score')
            plt.axis('off')
            
            # Thresholded anomaly score (binary) - using optimal threshold
            threshold = result['optimal_threshold'] if 'optimal_threshold' in result else np.percentile(result['anomaly_score'], 95)
            anomaly_binary = result['anomaly_score'] > threshold
            
            plt.subplot(1, 4, 4)
            plt.imshow(np.rot90(t1_img[:, slice_idx, :]), cmap='gray')
            binary_overlay = np.rot90(anomaly_binary[:, slice_idx, :])
            plt.imshow(binary_overlay, alpha=0.6, cmap='hot')
            plt.title(f'Thresholded Anomaly (optimal)')
            plt.axis('off')
            
            plt.suptitle(f'Autoencoder Anomaly Detection - {patient_id}', fontsize=16)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"{patient_id}_autoencoder_anomaly.png"))
                plt.close()
            else:
                plt.show()

def main():
    parser = argparse.ArgumentParser(description='Autoencoder-based BraTS anomaly detection')
    parser.add_argument('--data_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData',
                        help='Path to BraTS dataset')
    parser.add_argument('--output_dir', type=str, default='autoencoder_results',
                        help='Output directory for results')
    parser.add_argument('--max_patients', type=int, default=10,
                        help='Maximum number of patients to use')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Size of patches (cubic patches)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap between patches (0-1)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension size for autoencoder')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train autoencoder')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for autoencoder training')
    parser.add_argument('--min_tumor_ratio', type=float, default=0.05,
                        help='Minimum ratio of tumor voxels to consider a patch as containing tumor for evaluation')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Portion of normal patches to use for testing')
    parser.add_argument('--skull_margin_voxels', type=int, default=5,
                        help='Margin in voxels to exclude patches near skull/image border from AE training')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get patient data
    print("\nGetting patient data...")
    patient_data = get_patient_data(args.data_path, max_patients=args.max_patients)
    
    # Use the same size for all dimensions
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    
    # Run autoencoder-based anomaly detection
    start_time = time.time()
    results, test_auc, test_ap, optimal_threshold = run_autoencoder_anomaly_detection(
        device, 
        patient_data, 
        patch_size=patch_size,
        overlap=args.overlap,
        latent_dim=args.latent_dim, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_tumor_ratio=args.min_tumor_ratio,
        test_split=args.test_split,
        skull_margin_voxels=args.skull_margin_voxels
    )
    end_time = time.time()
    print(f"\nAnomaly detection completed in {end_time - start_time:.2f} seconds")
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, "metrics.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(metrics_file, "w") as f:
        f.write(f"Patch-level metrics:\n")
        f.write(f"AUC: {test_auc:.4f}\n")
        f.write(f"Average Precision: {test_ap:.4f}\n")
        f.write(f"Optimal threshold: {optimal_threshold:.6f}\n\n")
        f.write(f"Patient-level metrics:\n")
    
    # Evaluate results
    print("\nEvaluating results...")
    aucs = evaluate_results(results)
    
    # Append patient-level metrics
    with open(metrics_file, "a") as f:
        f.write(f"Mean patient AUC: {np.mean(aucs):.4f}\n")
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_results(results, patient_data, output_dir=args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Test metrics: AUC = {test_auc:.4f}, AP = {test_ap:.4f}")

if __name__ == "__main__":
    main() 