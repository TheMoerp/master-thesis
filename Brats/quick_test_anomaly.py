#!/usr/bin/env python3
"""
Quick test script for anomaly detection on BraTS data using a patch-based approach.
"""

import os
import sys
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
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
    return arr

def setup_model():
    """Set up the Anatomix model"""
    anatomix_path = os.path.abspath('./anatomix')
    if not os.path.exists(anatomix_path):
        print("Cloning Anatomix repository...")
        os.system('git clone https://github.com/neel-dey/anatomix.git')
        os.chdir('anatomix')
        os.system('pip install -e .')
        os.chdir('..')
    
    # Add anatomix to Python path
    if anatomix_path not in sys.path:
        sys.path.insert(0, anatomix_path)
    
    # Import Anatomix
    try:
        from anatomix.model.network import Unet
    except ImportError:
        sys.path.insert(0, os.path.join(anatomix_path, 'anatomix'))
        from model.network import Unet
    
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

def extract_features_for_patches(model, patches, device):
    """Extract features from brain patches using Anatomix model"""
    # Extract features for each patch
    features_list = []
    
    with torch.no_grad():
        for patch in patches:
            # Normalize patch
            patch_norm = minmax(patch)
            
            # Convert to tensor
            input_tensor = torch.from_numpy(patch_norm[np.newaxis, np.newaxis, ...]).float().to(device)
            
            # Extract features
            features = model(input_tensor)
            
            # Convert to numpy and flatten
            features_flat = features.cpu().numpy().reshape(features.shape[1], -1).T
            features_list.append(features_flat)
    
    return features_list

def create_patches(image, segmentation=None, patch_size=(32, 32, 32), overlap=0.5, min_tumor_ratio=0.05):
    """
    Create patches from 3D image with optional segmentation.
    Returns patches and a flag for each patch indicating if it contains tumor.
    """
    patches = []
    has_tumor = []
    tumor_ratios = []
    patch_locations = []
    
    # Calculate stride (amount to move when extracting patches)
    stride = [int(p * (1 - overlap)) for p in patch_size]
    
    # Extract patches
    for x in range(0, image.shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, image.shape[1] - patch_size[1] + 1, stride[1]):
            for z in range(0, image.shape[2] - patch_size[2] + 1, stride[2]):
                # Extract the patch
                patch = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                
                # Check if patch has the expected size (might be smaller at the edges)
                if patch.shape != tuple(patch_size):
                    continue
                
                # Add the patch
                patches.append(patch)
                patch_locations.append((x, y, z))
                
                # Check if patch contains tumor
                if segmentation is not None:
                    seg_patch = segmentation[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                    tumor_ratio = np.count_nonzero(seg_patch) / seg_patch.size
                    tumor_ratios.append(tumor_ratio)
                    has_tumor.append(tumor_ratio >= min_tumor_ratio)
                else:
                    tumor_ratios.append(0.0)
                    has_tumor.append(False)
    
    return patches, has_tumor, tumor_ratios, patch_locations

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

def run_patch_based_anomaly_detection(model, device, patient_data, patch_size=(32, 32, 32), overlap=0.5, n_neighbors=5, min_tumor_ratio=0.05):
    """Run patch-based anomaly detection on patient data"""
    all_normal_patches = []
    all_normal_features = []
    all_patient_patches = {}
    all_patient_patch_info = {}
    
    # Extract patches for each patient
    print("Extracting patches and features...")
    for patient_id, paths in tqdm(patient_data.items()):
        # Load data
        t1_img = nib.load(paths['t1n']).get_fdata()
        seg = nib.load(paths['seg']).get_fdata()
        
        # Create patches
        patches, has_tumor, tumor_ratios, patch_locations = create_patches(
            t1_img, seg, patch_size=patch_size, overlap=overlap, min_tumor_ratio=min_tumor_ratio
        )
        
        # Extract features for all patches
        patch_features = extract_features_for_patches(model, patches, device)
        
        # Store normal patches and features
        normal_indices = [i for i, tumor in enumerate(has_tumor) if not tumor]
        tumor_indices = [i for i, tumor in enumerate(has_tumor) if tumor]
        
        # Store normal patches for training
        for idx in normal_indices:
            all_normal_patches.append(patches[idx])
            all_normal_features.append(patch_features[idx])
        
        # Store all patient patches for later evaluation
        all_patient_patches[patient_id] = patches
        all_patient_patch_info[patient_id] = {
            'features': patch_features,
            'has_tumor': has_tumor,
            'tumor_ratios': tumor_ratios,
            'locations': patch_locations,
            'shape': t1_img.shape,
            'normal_indices': normal_indices,
            'tumor_indices': tumor_indices
        }
    
    # Check if we have normal patches
    if not all_normal_features:
        raise ValueError("No normal patches found. Try reducing min_tumor_ratio.")
    
    # Stack all normal features
    normal_features_stacked = np.vstack(all_normal_features)
    
    # Run anomaly detection for each patient
    print("\nRunning anomaly detection...")
    results = {}
    
    for patient_id, patch_info in tqdm(all_patient_patch_info.items()):
        # Initialize anomaly score map (same shape as original image)
        anomaly_score_map = np.zeros(patch_info['shape'])
        count_map = np.zeros(patch_info['shape'])  # To average overlapping patches
        
        # Train KNN model on all normal patches except from this patient
        train_indices = []
        current_idx = 0
        for pid, p_info in all_patient_patch_info.items():
            if pid != patient_id:
                # Add normal patches from other patients
                train_indices.extend(range(current_idx, current_idx + len(p_info['normal_indices'])))
            current_idx += len(p_info['normal_indices'])
        
        # If we don't have enough training data, use all normal patches
        if len(train_indices) < n_neighbors:
            train_features = normal_features_stacked
        else:
            train_features = normal_features_stacked[train_indices]
        
        # Train KNN model
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(train_features)))
        knn.fit(train_features)
        
        # Compute anomaly scores for each patch
        for i, features in enumerate(patch_info['features']):
            # Compute distances
            distances, _ = knn.kneighbors(features)
            
            # Compute anomaly score as mean distance to k-nearest neighbors
            anomaly_score = np.mean(distances, axis=1)
            
            # Get patch location
            x, y, z = patch_info['locations'][i]
            
            # Add anomaly score to the map
            anomaly_score_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += anomaly_score.mean()
            count_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += 1
        
        # Average the anomaly scores where patches overlapped
        count_map[count_map == 0] = 1  # Avoid division by zero
        anomaly_score_map /= count_map
        
        # Create ground truth from segmentation
        seg = nib.load(patient_data[patient_id]['seg']).get_fdata()
        ground_truth = (seg > 0).astype(int)
        
        # Store results
        results[patient_id] = {
            'anomaly_score': anomaly_score_map,
            'seg': ground_truth,
            'has_tumor_patches': sum(patch_info['has_tumor']),
            'normal_patches': sum(not t for t in patch_info['has_tumor']),
            'min_tumor_ratio': min_tumor_ratio
        }
    
    return results

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
                  f"Tumor patches: {result['has_tumor_patches']}, " +
                  f"Normal patches: {result['normal_patches']}")
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
            
            # Thresholded anomaly score (binary)
            threshold = np.percentile(result['anomaly_score'], 95)  # Top 5% as anomalies
            anomaly_binary = result['anomaly_score'] > threshold
            
            plt.subplot(1, 4, 4)
            plt.imshow(np.rot90(t1_img[:, slice_idx, :]), cmap='gray')
            binary_overlay = np.rot90(anomaly_binary[:, slice_idx, :])
            plt.imshow(binary_overlay, alpha=0.6, cmap='hot')
            plt.title(f'Thresholded Anomaly (95th percentile)')
            plt.axis('off')
            
            plt.suptitle(f'Patch-based Anomaly Detection - {patient_id}', fontsize=16)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"{patient_id}_patch_anomaly.png"))
                plt.close()
            else:
                plt.show()

def main():
    parser = argparse.ArgumentParser(description='Patch-based BraTS anomaly detection')
    parser.add_argument('--data_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData',
                        help='Path to BraTS dataset')
    parser.add_argument('--output_dir', type=str, default='patch_results',
                        help='Output directory for results')
    parser.add_argument('--max_patients', type=int, default=10,
                        help='Maximum number of patients to use')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Size of patches (cubic patches)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap between patches (0-1)')
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help='Number of neighbors for KNN anomaly detection')
    parser.add_argument('--min_tumor_ratio', type=float, default=0.05,
                        help='Minimum ratio of tumor voxels to consider a patch as containing tumor')
    args = parser.parse_args()
    
    # Set up model
    print("Setting up model...")
    model, device = setup_model()
    
    # Get patient data
    print("\nGetting patient data...")
    patient_data = get_patient_data(args.data_path, max_patients=args.max_patients)
    
    # Use the same size for all dimensions
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    
    # Run patch-based anomaly detection
    start_time = time.time()
    results = run_patch_based_anomaly_detection(
        model, device, patient_data, 
        patch_size=patch_size,
        overlap=args.overlap,
        n_neighbors=args.n_neighbors,
        min_tumor_ratio=args.min_tumor_ratio
    )
    end_time = time.time()
    print(f"\nAnomaly detection completed in {end_time - start_time:.2f} seconds")
    
    # Evaluate results
    print("\nEvaluating results...")
    aucs = evaluate_results(results)
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_results(results, patient_data, output_dir=args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 