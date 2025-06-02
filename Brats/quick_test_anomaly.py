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
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import argparse
from scipy.ndimage import zoom
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# Try to import FAISS for faster KNN
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS available - using GPU-accelerated KNN if possible")
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available - using sklearn KNN")

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

def extract_features_for_patches(model, patches, device, batch_size=8):
    """Extract features from brain patches using Anatomix model with batching"""
    features_list = []
    
    # Use a much larger batch size if on GPU
    if device.type == 'cuda':
        batch_size = min(32, len(patches))  # Increase batch size for GPU
    
    with torch.no_grad():
        # Process patches in batches to reduce memory usage and increase throughput
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            # Normalize patches
            batch_norm = [minmax(patch) for patch in batch_patches]
            
            # Convert to tensors and stack into a batch
            batch_tensors = [torch.from_numpy(p[np.newaxis, np.newaxis, ...]).float() for p in batch_norm]
            input_batch = torch.cat(batch_tensors, dim=0).to(device)
            
            # Extract features
            features = model(input_batch)
            
            # Convert to numpy and flatten for each patch in batch
            for j in range(features.shape[0]):
                feature_flat = features[j].cpu().numpy().reshape(features.shape[1], -1).T
                features_list.append(feature_flat)
                
            # Force GPU memory cleanup
            del features
            del input_batch
            del batch_tensors
            del batch_norm
            torch.cuda.empty_cache()
    
    return features_list

def process_patch(img_data, seg_data, x, y, z, patch_size, min_tumor_ratio):
    """Process a single patch - extracted for parallelization"""
    # Extract the patch
    patch = img_data[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
    
    # Check if patch has the expected size (might be smaller at the edges)
    if patch.shape != tuple(patch_size):
        return None, None, None, None
    
    # Check if patch contains tumor
    if seg_data is not None:
        seg_patch = seg_data[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
        tumor_ratio = np.count_nonzero(seg_patch) / seg_patch.size
        has_tumor = tumor_ratio >= min_tumor_ratio
    else:
        tumor_ratio = 0.0
        has_tumor = False
    
    return patch, has_tumor, tumor_ratio, (x, y, z)

def create_patches(image, segmentation=None, patch_size=(32, 32, 32), overlap=0.5, min_tumor_ratio=0.05, max_patches=None, use_parallel=True):
    """
    Create patches from 3D image with optional segmentation.
    Returns patches and a flag for each patch indicating if it contains tumor.
    Uses multiprocessing for faster patch extraction.
    """
    # Calculate stride (amount to move when extracting patches)
    stride = [int(p * (1 - overlap)) for p in patch_size]
    
    # Generate all possible patch coordinates
    coords = []
    for x in range(0, image.shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, image.shape[1] - patch_size[1] + 1, stride[1]):
            for z in range(0, image.shape[2] - patch_size[2] + 1, stride[2]):
                coords.append((x, y, z))
    
    # If too many patches, pre-filter based on brain mask
    if len(coords) > 10000 and segmentation is not None:
        # Create a brain mask (non-zero values in the image)
        brain_mask = (image > 0.01).astype(np.uint8)
        
        # Filter coords based on brain mask coverage
        filtered_coords = []
        for x, y, z in coords:
            # Check if patch contains enough brain voxels
            mask_patch = brain_mask[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
            if mask_patch.mean() > 0.5:  # At least 50% brain
                filtered_coords.append((x, y, z))
        
        coords = filtered_coords
        print(f"Reduced patches from {len(coords)} to {len(filtered_coords)} based on brain mask")
    
    # Limit number of patches if specified
    if max_patches and len(coords) > max_patches:
        if segmentation is not None and np.max(segmentation) > 0:
            # If we have tumor segmentation, prioritize tumor patches
            # Create tumor mask
            tumor_mask = (segmentation > 0).astype(np.uint8)
            
            # Calculate tumor content for each patch
            tumor_coords = []
            normal_coords = []
            
            for x, y, z in coords:
                tumor_patch = tumor_mask[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                tumor_ratio = np.count_nonzero(tumor_patch) / tumor_patch.size
                
                if tumor_ratio >= min_tumor_ratio:
                    tumor_coords.append((x, y, z, tumor_ratio))
                else:
                    normal_coords.append((x, y, z))
            
            # Sort tumor patches by tumor content
            tumor_coords.sort(key=lambda t: t[3], reverse=True)
            
            # Determine how many tumor and normal patches to include
            num_tumor = min(int(max_patches * 0.6), len(tumor_coords))
            num_normal = min(max_patches - num_tumor, len(normal_coords))
            
            # Randomly sample normal patches
            import random
            random.shuffle(normal_coords)
            
            # Combine tumor and normal patches
            coords = [(t[0], t[1], t[2]) for t in tumor_coords[:num_tumor]] + normal_coords[:num_normal]
            print(f"Selected {num_tumor} tumor patches and {num_normal} normal patches")
        else:
            # With no tumor segmentation, randomly sample patches
            import random
            random.shuffle(coords)
            coords = coords[:max_patches]
    
    # Determine number of workers based on system resources
    if use_parallel and len(coords) > 100:  # Only use parallel for significant number of patches
        num_workers = min(os.cpu_count() or 4, 8)  # Limit to avoid excessive memory usage
    else:
        num_workers = 0  # Sequential processing
    
    # Process patches
    patches = []
    has_tumor = []
    tumor_ratios = []
    patch_locations = []
    
    # Batched processing for lower memory usage
    batch_size = 500
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:min(i+batch_size, len(coords))]
        
        if num_workers > 0:
            # Parallel processing for this batch
            process_func = partial(process_patch, image, segmentation, 
                                  patch_size=patch_size, min_tumor_ratio=min_tumor_ratio)
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_func, 
                                          [c[0] for c in batch_coords], 
                                          [c[1] for c in batch_coords], 
                                          [c[2] for c in batch_coords]))
                
                # Process batch results
                for result in results:
                    if result[0] is not None:  # Skip None results
                        patches.append(result[0])
                        has_tumor.append(result[1])
                        tumor_ratios.append(result[2])
                        patch_locations.append(result[3])
        else:
            # Sequential processing
            for x, y, z in batch_coords:
                result = process_patch(image, segmentation, x, y, z, patch_size, min_tumor_ratio)
                if result[0] is not None:
                    patches.append(result[0])
                    has_tumor.append(result[1])
                    tumor_ratios.append(result[2])
                    patch_locations.append(result[3])
    
    print(f"Created {len(patches)} valid patches")
    return np.array(patches), has_tumor, tumor_ratios, patch_locations

def get_patient_data(data_path, max_patients=10, test_patients=None):
    """Get a subset of patient data, optionally separating into train and test"""
    # Get list of patient directories
    patient_dirs = sorted(glob(os.path.join(data_path, 'BraTS-GLI-*')))
    
    if not patient_dirs:
        raise FileNotFoundError(f"No BraTS patient directories found in {data_path}")
    
    # If test_patients is provided, use it to separate patients
    train_patients = []
    test_patient_dirs = []
    
    if test_patients:
        # Use the specified number of patients for testing
        if isinstance(test_patients, int):
            train_dirs = patient_dirs[:-test_patients] if test_patients > 0 else patient_dirs
            test_patient_dirs = patient_dirs[-test_patients:] if test_patients > 0 else []
        # Use specific patient IDs for testing
        elif isinstance(test_patients, list):
            test_patient_dirs = [d for d in patient_dirs if os.path.basename(d) in test_patients]
            train_dirs = [d for d in patient_dirs if os.path.basename(d) not in test_patients]
        
        # Limit train to max_patients
        train_dirs = train_dirs[:max_patients]
        train_patients = [os.path.basename(d) for d in train_dirs]
    else:
        # No test split, use all for training up to max_patients
        train_dirs = patient_dirs[:max_patients]
        train_patients = [os.path.basename(d) for d in train_dirs]
    
    print(f"Using {len(train_dirs)} patient directories for training")
    if test_patient_dirs:
        print(f"Using {len(test_patient_dirs)} patient directories for testing")
    
    # Get data for each patient (both train and test)
    patient_data = {}
    
    for patient_dir in train_dirs + test_patient_dirs:
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
    
    return patient_data, train_patients, [os.path.basename(d) for d in test_patient_dirs]

def create_faiss_knn(features, n_neighbors, use_gpu=True):
    """Create a FAISS KNN index for faster search"""
    # Get dimensions
    n_samples, n_features = features.shape
    
    # Convert to float32 as required by FAISS
    features = features.astype(np.float32)
    
    # Use IVFFlat for large datasets (more than 100k vectors)
    use_ivf = n_samples > 100000
    
    # Create index - prefer GPU
    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0  # Use first GPU
            config.useFloat16 = False  # Use full precision for accuracy
            
            if use_ivf:
                # For large datasets, use IVFFLAT index for faster search
                # Create quantizer on GPU
                quantizer = faiss.GpuIndexFlatL2(res, n_features, config)
                # Number of centroids to use (rule of thumb: sqrt(n) or n/50)
                nlist = int(min(4096, max(8, np.sqrt(n_samples))))
                # Create IVF index
                index = faiss.GpuIndexIVFFlat(res, n_features, nlist, faiss.METRIC_L2)
                # Use the GPU quantizer
                index.quantizer = quantizer
                # Train the index
                print(f"Training IVF index with {nlist} clusters...")
                index.train(features)
                print("Using GPU-accelerated FAISS IVF index for faster search")
            else:
                # For smaller datasets, use flat index
                index = faiss.GpuIndexFlatL2(res, n_features, config)
                print("Using GPU-accelerated FAISS flat index")
        except Exception as e:
            print(f"Error creating GPU FAISS index: {e}. Falling back to CPU.")
            index = faiss.IndexFlatL2(n_features)
    else:
        # CPU fallback - also try to use IVF for large datasets
        if use_ivf and n_samples > 100000:
            # Number of centroids
            nlist = int(min(4096, max(8, np.sqrt(n_samples))))
            quantizer = faiss.IndexFlatL2(n_features)
            index = faiss.IndexIVFFlat(quantizer, n_features, nlist, faiss.METRIC_L2)
            print(f"Training CPU IVF index with {nlist} clusters...")
            index.train(features)
            print("Using CPU FAISS IVF index")
        else:
            index = faiss.IndexFlatL2(n_features)
            print("Using CPU FAISS flat index")
    
    # Add vectors to the index
    index.add(features)
    
    # Set nprobe for IVF indexes (how many clusters to visit during search)
    if hasattr(index, 'nprobe'):
        # Adjust based on needs (higher = more accurate but slower)
        index.nprobe = min(64, nlist)
    
    return index

def query_faiss_knn(index, query_vectors, n_neighbors):
    """Query the FAISS index for k-nearest neighbors"""
    # Convert to float32 as required by FAISS
    query_vectors = query_vectors.astype(np.float32)
    
    # Search for nearest neighbors
    distances, indices = index.search(query_vectors, n_neighbors)
    
    return distances, indices

def compute_anomaly_scores_batch(patch_features, patch_locations, knn_index, n_neighbors, 
                               shape, patch_size, is_faiss=False, batch_size=100):
    """Compute anomaly scores for all patches of a patient in batches"""
    # Initialize anomaly score map and count map
    anomaly_score_map = np.zeros(shape)
    count_map = np.zeros(shape)
    
    # Process patches in batches to better utilize GPU
    num_patches = len(patch_features)
    
    # Use larger batches for FAISS to maximize GPU utilization
    if is_faiss:
        batch_size = min(1000, num_patches)  # Much larger batches for GPU
    
    for start_idx in range(0, num_patches, batch_size):
        end_idx = min(start_idx + batch_size, num_patches)
        batch_features = np.vstack(patch_features[start_idx:end_idx])
        
        # Compute distances for the batch
        if is_faiss:
            # Pre-transfer to GPU memory format
            batch_features = batch_features.astype(np.float32)
            distances, _ = knn_index.search(batch_features, n_neighbors)
        else:
            distances, _ = knn_index.kneighbors(batch_features)
        
        # Process each patch in the batch
        for i in range(end_idx - start_idx):
            idx = start_idx + i
            # Get patch location
            x, y, z = patch_locations[idx]
            
            # Compute anomaly score as mean distance to k neighbors
            patch_anomaly_score = np.mean(distances[i])
            
            # Add anomaly score to the map
            anomaly_score_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += patch_anomaly_score
            count_map[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += 1
    
    # Average the anomaly scores where patches overlapped
    count_map[count_map == 0] = 1  # Avoid division by zero
    anomaly_score_map /= count_map
    
    return anomaly_score_map

def run_patch_based_anomaly_detection(model, device, patient_data, train_patients, test_patients=None, 
                                     patch_size=(32, 32, 32), overlap=0.5, n_neighbors=5, min_tumor_ratio=0.05,
                                     max_patches_per_patient=None, batch_size=8, use_parallel=True,
                                     force_gpu=False):
    """Run patch-based anomaly detection on patient data with separation of train and test"""
    all_normal_features = []
    all_patient_patch_info = {}
    
    # If no test patients specified, evaluate on all patients
    patients_to_evaluate = test_patients if test_patients else train_patients
    
    # Set larger batch size for GPU
    if device.type == 'cuda':
        batch_size = min(32, batch_size * 4)  # Increase batch size on GPU
    
    # Extract patches for each patient
    print("Extracting patches and features...")
    for patient_id, paths in tqdm(patient_data.items()):
        # Skip patients not in either train or test set
        if patient_id not in train_patients + (test_patients or []):
            continue
            
        # Load data
        t1_img = nib.load(paths['t1n']).get_fdata()
        seg = nib.load(paths['seg']).get_fdata()
        
        # Create patches with a lower overlap for speed if we have too many patches
        current_overlap = overlap
        if max_patches_per_patient and max_patches_per_patient < 5000:
            # Keep the original overlap
            pass
        else:
            # Reduce overlap for very large volumes to limit patches
            # Use less overlap for larger patch sizes
            if patch_size[0] >= 48:
                current_overlap = min(0.25, overlap)  # Reduce overlap for larger patches
            elif patch_size[0] >= 32:
                current_overlap = min(0.3, overlap)
            
        # Create patches
        patches, has_tumor, tumor_ratios, patch_locations = create_patches(
            t1_img, seg, patch_size=patch_size, overlap=current_overlap, 
            min_tumor_ratio=min_tumor_ratio, max_patches=max_patches_per_patient,
            use_parallel=use_parallel
        )
        
        print(f"Patient {patient_id}: Extracted {len(patches)} patches, extracting features...")
        # Extract features for all patches with larger batch size to use GPU more efficiently
        patch_features = extract_features_for_patches(model, patches, device, batch_size=batch_size)
        
        # Store normal patches and features
        normal_indices = [i for i, tumor in enumerate(has_tumor) if not tumor]
        tumor_indices = [i for i, tumor in enumerate(has_tumor) if tumor]
        
        # Store normal patches from training patients only
        if patient_id in train_patients:
            for idx in normal_indices:
                all_normal_features.append(patch_features[idx])
        
        # Store all patient patch info for later evaluation
        all_patient_patch_info[patient_id] = {
            'features': patch_features,
            'has_tumor': has_tumor,
            'tumor_ratios': tumor_ratios,
            'locations': patch_locations,
            'shape': t1_img.shape,
            'normal_indices': normal_indices,
            'tumor_indices': tumor_indices
        }
        
        # Clear memory
        del patches
        del patch_features
        torch.cuda.empty_cache()
    
    # Check if we have normal features
    if not all_normal_features:
        raise ValueError("No normal patches found. Try reducing min_tumor_ratio.")
    
    # Stack all normal features
    normal_features_stacked = np.vstack(all_normal_features)
    print(f"Total normal features: {normal_features_stacked.shape}")
    
    # Create KNN index - choose between FAISS and scikit-learn
    print("Training KNN model on normal features...")
    if FAISS_AVAILABLE:
        # Force GPU based on the parameter
        knn_index = create_faiss_knn(normal_features_stacked, n_neighbors, use_gpu=force_gpu or torch.cuda.is_available())
        is_faiss = True
    else:
        knn_index = NearestNeighbors(n_neighbors=min(n_neighbors, len(normal_features_stacked)))
        knn_index.fit(normal_features_stacked)
        is_faiss = False
    
    # Run anomaly detection for each patient to evaluate
    print("\nRunning anomaly detection...")
    results = {}
    
    # Increase batch size for anomaly detection
    anomaly_batch_size = 1000 if is_faiss else 100
    
    # Process patients one by one to focus GPU resources on one patient at a time
    for patient_id in tqdm(patients_to_evaluate):
        # Skip if patient data not available
        if patient_id not in all_patient_patch_info:
            continue
            
        patch_info = all_patient_patch_info[patient_id]
        
        print(f"Computing anomaly scores for patient {patient_id} with {len(patch_info['features'])} patches...")
        # Compute anomaly scores for all patches in batches to maximize GPU utilization
        anomaly_score_map = compute_anomaly_scores_batch(
            patch_info['features'], 
            patch_info['locations'], 
            knn_index, 
            n_neighbors, 
            patch_info['shape'], 
            patch_size, 
            is_faiss=is_faiss,
            batch_size=anomaly_batch_size
        )
        
        # Create ground truth from segmentation
        seg = nib.load(patient_data[patient_id]['seg']).get_fdata()
        ground_truth = (seg > 0).astype(int)
        
        # Store results
        results[patient_id] = {
            'anomaly_score': anomaly_score_map,
            'seg': ground_truth,
            'has_tumor_patches': sum(patch_info['has_tumor']),
            'normal_patches': sum(not t for t in patch_info['has_tumor']),
            'min_tumor_ratio': min_tumor_ratio,
            'is_test': patient_id in (test_patients or [])
        }
        
        # Clear CUDA cache between patients
        torch.cuda.empty_cache()
    
    return results

def evaluate_results(results, test_patients=None):
    """Evaluate anomaly detection results with AUC and Average Precision"""
    aucs = []
    aps = []  # Average Precision scores
    test_aucs = []
    test_aps = []
    
    # If test_patients is None, consider all as evaluation
    patients_to_report = test_patients or list(results.keys())
    
    for patient_id, result in results.items():
        # Use segmentation as ground truth (1 for tumor, 0 for normal)
        ground_truth = result['seg'].flatten()
        
        # If no tumor, skip
        if np.sum(ground_truth) == 0:
            continue
        
        # Get anomaly score
        pred = result['anomaly_score'].flatten()
        
        # Calculate metrics
        try:
            auc = roc_auc_score(ground_truth, pred)
            ap = average_precision_score(ground_truth, pred)
            
            # Store metrics based on whether this is a test patient
            is_test = result.get('is_test', False)
            if is_test:
                test_aucs.append(auc)
                test_aps.append(ap)
            else:
                aucs.append(auc)
                aps.append(ap)
            
            # Only report for patients we're interested in
            if patient_id in patients_to_report:
                patient_type = "TEST" if is_test else "TRAIN"
                print(f"Patient {patient_id} ({patient_type}): AUC = {auc:.4f}, AP = {ap:.4f}, " +
                      f"Tumor patches: {result['has_tumor_patches']}, " +
                      f"Normal patches: {result['normal_patches']}")
        except Exception as e:
            print(f"Error calculating metrics for patient {patient_id}: {e}")
    
    # Print summary statistics
    if aucs:
        print(f"\nTRAIN Mean AUC: {np.mean(aucs):.4f}, Mean AP: {np.mean(aps):.4f}")
    
    if test_aucs:
        print(f"TEST Mean AUC: {np.mean(test_aucs):.4f}, Mean AP: {np.mean(test_aps):.4f}")
    
    return {'train_auc': aucs, 'train_ap': aps, 'test_auc': test_aucs, 'test_ap': test_aps}

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
                        help='Maximum number of patients to use for training')
    parser.add_argument('--test_patients', type=int, default=2,
                        help='Number of patients to use for testing (taken from the end of the list)')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Size of patches (cubic patches)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap between patches (0-1)')
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help='Number of neighbors for KNN anomaly detection')
    parser.add_argument('--min_tumor_ratio', type=float, default=0.05,
                        help='Minimum ratio of tumor voxels to consider a patch as containing tumor')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for feature extraction')
    parser.add_argument('--max_patches', type=int, default=3000,
                        help='Maximum number of patches per patient (None for all patches)')
    parser.add_argument('--no_parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--no_faiss', action='store_true',
                        help='Disable FAISS for KNN (use sklearn instead)')
    parser.add_argument('--force_gpu', action='store_true',
                        help='Force GPU usage for KNN with FAISS')
    args = parser.parse_args()
    
    if args.no_faiss:
        global FAISS_AVAILABLE
        FAISS_AVAILABLE = False
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available - using CPU only")
    
    # Set up model
    print("Setting up model...")
    model, device = setup_model()
    
    # Get patient data with train/test split
    print("\nGetting patient data...")
    patient_data, train_patients, test_patients = get_patient_data(
        args.data_path, 
        max_patients=args.max_patients,
        test_patients=args.test_patients
    )
    
    # Use the same size for all dimensions
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    
    # Run patch-based anomaly detection
    start_time = time.time()
    results = run_patch_based_anomaly_detection(
        model, device, patient_data, 
        train_patients=train_patients,
        test_patients=test_patients,
        patch_size=patch_size,
        overlap=args.overlap,
        n_neighbors=args.n_neighbors,
        min_tumor_ratio=args.min_tumor_ratio,
        max_patches_per_patient=args.max_patches,
        batch_size=args.batch_size,
        use_parallel=not args.no_parallel,
        force_gpu=args.force_gpu
    )
    end_time = time.time()
    print(f"\nAnomaly detection completed in {end_time - start_time:.2f} seconds")
    
    # Evaluate results
    print("\nEvaluating results...")
    metrics = evaluate_results(results, test_patients)
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_results(results, patient_data, output_dir=args.output_dir)
    
    # Save metrics to file
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_file = os.path.join(args.output_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Train Patients: {train_patients}\n")
            f.write(f"Test Patients: {test_patients}\n\n")
            
            if metrics['train_auc']:
                f.write(f"TRAIN Mean AUC: {np.mean(metrics['train_auc']):.4f}\n")
                f.write(f"TRAIN Mean AP: {np.mean(metrics['train_ap']):.4f}\n")
            
            if metrics['test_auc']:
                f.write(f"TEST Mean AUC: {np.mean(metrics['test_auc']):.4f}\n")
                f.write(f"TEST Mean AP: {np.mean(metrics['test_ap']):.4f}\n")
        
        print(f"Metrics saved to {metrics_file}")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 