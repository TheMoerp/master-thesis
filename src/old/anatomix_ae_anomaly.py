#!/usr/bin/env python3
"""
Anomaly detection on BraTS data using Anatomix feature extraction 
followed by autoencoder-based anomaly detection.
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
import gc
from sklearn.decomposition import PCA

# Suppress warnings
warnings.filterwarnings("ignore")

def optimize_gpu_memory():
    """Configure PyTorch to optimize GPU memory usage"""
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Enable memory efficient algorithms where possible
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
        
        # Set memory allocation to be more efficient
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Print current GPU memory status
        print(f"Initial GPU memory: Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Force garbage collection
        gc.collect()
    
    print("GPU memory optimization configured")

# Call optimization function at startup
optimize_gpu_memory()

def minmax(arr, minclip=None, maxclip=None):
    """Normalize array to 0-1 range, with optional clipping"""
    if not ((minclip is None) and (maxclip is None)):
        arr = np.clip(arr, minclip, maxclip)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr

class AnatomixFeatureAutoencoder(nn.Module):
    """Autoencoder for Anatomix extracted features"""
    def __init__(self, input_dim, latent_dim=64):
        super(AnatomixFeatureAutoencoder, self).__init__()
        
        # Calculate appropriate hidden dimensions based on input dimension
        hidden_dim2 = min(512, max(128, input_dim // 8))
        hidden_dim1 = min(2048, max(256, input_dim // 4))
        
        # Encoder - fully connected layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim)
        )
        
        # Decoder - fully connected layers to reconstruct feature maps
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )
        
    def forward(self, x):
        # Ensure x is flattened
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Encode and decode
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        
        return reconstruction, z

def setup_anatomix_model(output_nc=4):
    """Set up the Anatomix model for feature extraction with reduced dimensions
    
    Args:
        output_nc: Number of output channels (smaller means less memory usage)
    """
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
    
    # Create model with reduced output channels
    model = Unet(
        dimension=3,
        input_nc=1,
        output_nc=output_nc,  # Reduced output channels (original was 16)
        num_downs=4,
        ngf=16,
    ).to(device)
    
    # Load weights
    weights_path = os.path.join(anatomix_path, "model-weights", "anatomix.pth")
    if os.path.exists(weights_path):
        # Load weights but allow mismatched output layer
        state_dict = torch.load(weights_path, map_location=device)
        
        # Filter out output layer weights that don't match
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
        
        model.eval()
        print(f"Loaded Anatomix model weights with reduced output channels: {output_nc}")
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
            
            # Move features to CPU to free GPU memory
            features_cpu = features.detach().cpu()
            features_list.append(features_cpu)
            
            # Clean up to avoid GPU memory buildup
            del input_tensor
            del features
    
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

def run_anatomix_ae_anomaly_detection(anatomix_model, device, patient_data, 
                                     patch_size=(32, 32, 32), overlap=0.5, 
                                     latent_dim=64, epochs=50, batch_size=32, 
                                     min_tumor_ratio=0.05, test_split=0.2):
    """Run anomaly detection using Anatomix features and autoencoder"""
    # Store feature maps for normal and tumor patches
    all_normal_features = []
    all_tumor_features = []
    all_patient_patch_info = {}
    
    # Extract patches and features for each patient
    print("Extracting patches and features...")
    for patient_id, paths in tqdm(patient_data.items(), desc="Processing patients"):
        # Load data
        t1_img = nib.load(paths['t1n']).get_fdata()
        seg = nib.load(paths['seg']).get_fdata()
        
        # Create patches
        patches, has_tumor, tumor_ratios, patch_locations = create_patches(
            t1_img, seg, patch_size=patch_size, overlap=overlap, min_tumor_ratio=min_tumor_ratio
        )
        
        # Extract features using Anatomix
        patch_features = extract_features_for_patches(anatomix_model, patches, device)
        
        # Store patch information
        all_patient_patch_info[patient_id] = {
            'features': patch_features,
            'has_tumor': has_tumor,
            'tumor_ratios': tumor_ratios,
            'locations': patch_locations,
            'shape': t1_img.shape
        }
        
        # Collect normal and tumor features
        normal_indices = [i for i, tumor in enumerate(has_tumor) if not tumor]
        tumor_indices = [i for i, tumor in enumerate(has_tumor) if tumor]
        
        normal_features = [patch_features[i] for i in normal_indices]
        tumor_features = [patch_features[i] for i in tumor_indices]
        
        all_normal_features.extend(normal_features)
        all_tumor_features.extend(tumor_features)
        
        print(f"Processed patient {patient_id}: {len(normal_features)} normal, {len(tumor_features)} tumor patches")
    
    # Split normal features into train and test sets
    np.random.seed(42)  # For reproducibility
    n_normal = len(all_normal_features)
    n_tumor = len(all_tumor_features)
    
    print(f"Total normal feature maps: {n_normal}, Total tumor feature maps: {n_tumor}")
    
    # Create indices for normal features and shuffle them
    normal_indices = np.arange(n_normal)
    np.random.shuffle(normal_indices)
    
    # Split into train and test
    n_test_normal = int(n_normal * test_split)
    train_indices = normal_indices[n_test_normal:]
    test_normal_indices = normal_indices[:n_test_normal]
    
    # Create training set (normal features only)
    train_features = [all_normal_features[i] for i in train_indices]
    print(f"Training on {len(train_features)} normal features, Testing on {n_test_normal} normal and {n_tumor} tumor features")
    
    # Prepare tensors for autoencoder training
    # First determine the input dimension by flattening a feature map
    first_feature = train_features[0]
    feature_dim = int(first_feature.flatten().numel())
    
    # Create a simpler flattened dataset for training
    train_tensors = []
    for feature in train_features:
        # Flatten the feature and move to device
        feature_flat = feature.flatten().to(device)
        train_tensors.append(feature_flat)
    
    # Stack into a tensor dataset
    train_data = torch.stack(train_tensors)
    print(f"Training data shape: {train_data.shape}, Feature dimension: {feature_dim}")
    
    # Create autoencoder
    feature_autoencoder = AnatomixFeatureAutoencoder(input_dim=feature_dim, latent_dim=latent_dim).to(device)
    print(f"Autoencoder created with input dimension {feature_dim} and latent dimension {latent_dim}")
    
    # Create dataset and dataloader
    dataset = TensorDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train autoencoder
    print("Training autoencoder on Anatomix features...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(feature_autoencoder.parameters(), lr=1e-3)
    
    # Training loop
    start_time = time.time()
    for epoch in tqdm(range(epochs), desc="Training autoencoder", unit="epoch"):
        total_loss = 0
        batch_progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in batch_progress:
            # Get batch
            x = batch[0]
            
            # Forward pass
            optimizer.zero_grad()
            reconstruction, _ = feature_autoencoder(x)
            
            # Calculate loss
            loss = criterion(reconstruction, x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_progress.set_postfix({"loss": f"{loss.item():.6f}"})
            
            # Free memory
            del reconstruction
        
        # Print progress and clean memory after each epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            estimated_time = avg_time_per_epoch * remaining_epochs
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}, "
                  f"Elapsed: {elapsed_time:.2f}s, ETA: {estimated_time:.2f}s")
            
            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    total_training_time = time.time() - start_time
    print(f"Autoencoder training complete in {total_training_time:.2f} seconds")
    
    # Save model
    os.makedirs('anatomix_ae_results', exist_ok=True)
    torch.save(feature_autoencoder.state_dict(), 'anatomix_ae_results/model.pth')
    
    # Put autoencoder in eval mode
    feature_autoencoder.eval()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    
    # Prepare test set (both normal and tumor features)
    test_normal_features = [all_normal_features[i] for i in test_normal_indices]
    test_features = test_normal_features + all_tumor_features
    test_labels = [0] * len(test_normal_features) + [1] * len(all_tumor_features)
    
    # Calculate reconstruction error for each test feature
    test_scores = []
    test_start_time = time.time()
    
    with torch.no_grad():
        for feature in tqdm(test_features, desc="Testing samples"):
            # Flatten feature and move to device
            feature_flat = feature.flatten().to(device).unsqueeze(0)  # Add batch dimension
            
            # Get reconstruction
            reconstruction, _ = feature_autoencoder(feature_flat)
            
            # Calculate MSE
            mse = torch.mean((reconstruction - feature_flat)**2).item()
            test_scores.append(mse)
    
    test_time = time.time() - test_start_time
    print(f"Test evaluation completed in {test_time:.2f} seconds")
    
    # Calculate AUC on test set
    test_auc = roc_auc_score(test_labels, test_scores)
    test_ap = average_precision_score(test_labels, test_scores)
    
    print(f"Test Set Patch-Level Metrics:")
    print(f"AUC: {test_auc:.4f}")
    print(f"Average Precision: {test_ap:.4f}")
    
    # Find optimal threshold using PR curve
    precision, recall, thresholds = precision_recall_curve(test_labels, test_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    
    print(f"Optimal threshold: {optimal_threshold:.6f} (F1: {f1_scores[optimal_idx]:.4f})")
    
    # Create patient-level anomaly maps
    print("Computing patient-level anomaly scores...")
    results = {}
    map_start_time = time.time()
    
    for patient_id, patch_info in tqdm(all_patient_patch_info.items(), desc="Creating anomaly maps"):
        # Initialize anomaly score map (same shape as original image)
        anomaly_score_map = np.zeros(patch_info['shape'])
        count_map = np.zeros(patch_info['shape'])  # To average overlapping patches
        
        # Process each patch feature
        for i, (feature, location) in enumerate(zip(patch_info['features'], patch_info['locations'])):
            # Flatten feature and move to device
            feature_flat = feature.flatten().to(device).unsqueeze(0)  # Add batch dimension
            
            # Compute reconstruction error
            with torch.no_grad():
                reconstruction, _ = feature_autoencoder(feature_flat)
                mse = torch.mean((reconstruction - feature_flat)**2).item()
            
            # Get patch location
            x, y, z = location
            
            # Add anomaly score to the map
            anomaly_score_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += mse
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
            'normal_patches': len(patch_info['features']) - sum(patch_info['has_tumor']),
            'min_tumor_ratio': min_tumor_ratio,
            'optimal_threshold': optimal_threshold
        }
    
    map_time = time.time() - map_start_time
    print(f"Anomaly map generation completed in {map_time:.2f} seconds")
    
    return results, test_auc, test_ap, optimal_threshold

def evaluate_results(results):
    """Evaluate anomaly detection results"""
    aucs = []
    
    for patient_id, result in results.items():
        # Use segmentation as ground truth
        ground_truth = result['seg'].flatten()
        
        # Skip if no tumor
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
            threshold = result['optimal_threshold'] if 'optimal_threshold' in result else np.percentile(result['anomaly_score'], 95)
            anomaly_binary = result['anomaly_score'] > threshold
            
            plt.subplot(1, 4, 4)
            plt.imshow(np.rot90(t1_img[:, slice_idx, :]), cmap='gray')
            binary_overlay = np.rot90(anomaly_binary[:, slice_idx, :])
            plt.imshow(binary_overlay, alpha=0.6, cmap='hot')
            plt.title(f'Thresholded Anomaly (optimal)')
            plt.axis('off')
            
            plt.suptitle(f'Anatomix+AE Anomaly Detection - {patient_id}', fontsize=16)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"{patient_id}_anatomix_ae_anomaly.png"))
                plt.close()
            else:
                plt.show()

def main():
    parser = argparse.ArgumentParser(description='Anatomix+AE BraTS anomaly detection')
    parser.add_argument('--data_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData',
                        help='Path to BraTS dataset')
    parser.add_argument('--output_dir', type=str, default='anatomix_ae_results',
                        help='Output directory for results')
    parser.add_argument('--max_patients', type=int, default=3,
                        help='Maximum number of patients to use')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Size of patches (cubic patches)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap between patches (0-1)')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension size for autoencoder')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train autoencoder')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for autoencoder training')
    parser.add_argument('--min_tumor_ratio', type=float, default=0.05,
                        help='Minimum ratio of tumor voxels to consider a patch as containing tumor')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Portion of normal patches to use for testing')
    args = parser.parse_args()
    
    # Set PyTorch memory allocation settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Ensure CUDA cache is empty before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Starting GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Set up anatomix model for feature extraction
    print("Setting up Anatomix model...")
    anatomix_model, device = setup_anatomix_model()
    
    # Get patient data
    print("\nGetting patient data...")
    patient_data = get_patient_data(args.data_path, max_patients=args.max_patients)
    
    # Use the same size for all dimensions
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    
    # Adjust batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_memory < 8:
            args.batch_size = min(args.batch_size, 16)
            print(f"Adjusted batch size to {args.batch_size} based on available GPU memory ({gpu_memory:.1f} GB)")
    
    # Run anomaly detection
    start_time = time.time()
    results, test_auc, test_ap, optimal_threshold = run_anatomix_ae_anomaly_detection(
        anatomix_model, 
        device, 
        patient_data, 
        patch_size=patch_size,
        overlap=args.overlap,
        latent_dim=args.latent_dim, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_tumor_ratio=args.min_tumor_ratio,
        test_split=args.test_split
    )
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nAnomaly detection completed in {total_time:.2f} seconds")
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
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