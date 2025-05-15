#!/usr/bin/env python3
"""
Script to compare different anomaly detection methods on BraTS dataset using Anatomix features.
"""

import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from collections import defaultdict
import argparse
from tqdm import tqdm
import time
import warnings

# Suppress NumPy and PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import the feature extraction code from the main script
from brats_feature_extraction import (
    setup_anatomix, minmax, get_brats_paths, 
    extract_all_modality_features, compute_feature_vectors
)

def anomaly_detection_knn(features_list, test_features, k=5):
    """
    Perform anomaly detection using k-nearest neighbors.
    
    Args:
        features_list: List of feature vectors from normal samples
        test_features: Feature vectors from test sample
        k: Number of neighbors
        
    Returns:
        anomaly_score: Anomaly score map
    """
    # Concatenate all training features
    all_features = np.vstack(features_list)
    
    # Train KNN model
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(all_features)
    
    # Compute distances to k-nearest neighbors
    distances, _ = knn.kneighbors(test_features)
    
    # Compute anomaly score as mean distance to k-nearest neighbors
    anomaly_score = np.mean(distances, axis=1)
    
    return anomaly_score

def anomaly_detection_isolation_forest(features_list, test_features, n_estimators=100, contamination=0.1):
    """
    Perform anomaly detection using Isolation Forest.
    
    Args:
        features_list: List of feature vectors from normal samples
        test_features: Feature vectors from test sample
        n_estimators: Number of estimators
        contamination: Contamination parameter
        
    Returns:
        anomaly_score: Anomaly score map
    """
    # Concatenate all training features
    all_features = np.vstack(features_list)
    
    # Train Isolation Forest model
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    model.fit(all_features)
    
    # Higher negative values indicate stronger anomalies
    # So we multiply by -1 to make higher values correspond to anomalies
    anomaly_score = -model.score_samples(test_features)
    
    return anomaly_score

def anomaly_detection_one_class_svm(features_list, test_features, nu=0.1, gamma='scale'):
    """
    Perform anomaly detection using One-Class SVM.
    
    Args:
        features_list: List of feature vectors from normal samples
        test_features: Feature vectors from test sample
        nu: An upper bound on the fraction of training errors
        gamma: Kernel coefficient
        
    Returns:
        anomaly_score: Anomaly score map
    """
    # Concatenate all training features
    all_features = np.vstack(features_list)
    
    # Train One-Class SVM model
    model = OneClassSVM(nu=nu, gamma=gamma)
    model.fit(all_features)
    
    # Decision function outputs negative values for outliers
    # So we multiply by -1 to make higher values correspond to anomalies
    anomaly_score = -model.decision_function(test_features)
    
    return anomaly_score

def evaluate_method(results, method_name):
    """Evaluate anomaly detection results using segmentation as ground truth"""
    aucs = []
    precision_aucs = []
    computation_times = []
    
    for patient_id, result in results.items():
        if result['segmentation'] is not None:
            # Use segmentation as ground truth (1 for anomaly, 0 for normal)
            ground_truth = (result['segmentation'] > 0).astype(int).flatten()
            
            # If no anomaly in ground truth, skip this case
            if np.sum(ground_truth) == 0 or np.sum(ground_truth) == len(ground_truth):
                continue
                
            # Flatten anomaly score
            pred = result['combined_score'].flatten()
            
            # Calculate AUC and PR AUC
            try:
                # ROC AUC
                roc_auc = roc_auc_score(ground_truth, pred)
                aucs.append(roc_auc)
                
                # Precision-Recall AUC
                precision, recall, _ = precision_recall_curve(ground_truth, pred)
                pr_auc = auc(recall, precision)
                precision_aucs.append(pr_auc)
                
                # Get computation time
                computation_times.append(result['computation_time'])
                
                print(f"Patient {patient_id} ({method_name}): ROC AUC = {roc_auc:.4f}, PR AUC = {pr_auc:.4f}, Time = {result['computation_time']:.4f}s")
            except Exception as e:
                print(f"Error calculating metrics for patient {patient_id}: {e}")
    
    if aucs:
        print(f"\n--- {method_name} Results ---")
        print(f"Mean ROC AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"Mean PR AUC: {np.mean(precision_aucs):.4f} ± {np.std(precision_aucs):.4f}")
        print(f"Mean computation time: {np.mean(computation_times):.4f}s ± {np.std(computation_times):.4f}s")
        return {
            'method': method_name,
            'roc_auc_mean': np.mean(aucs),
            'roc_auc_std': np.std(aucs),
            'pr_auc_mean': np.mean(precision_aucs),
            'pr_auc_std': np.std(precision_aucs),
            'time_mean': np.mean(computation_times),
            'time_std': np.std(computation_times)
        }
    else:
        print(f"No valid scores calculated for {method_name}.")
        return None

def visualize_comparison(sample_paths, results_dict, segmentation, patient_id, slice_idx=None):
    """Visualize comparison of different anomaly detection methods"""
    # Load a sample image for reference
    t1n = nib.load(sample_paths['t1n']).get_fdata()
    t1n = minmax(t1n)
    
    # Find slice with maximum tumor (if segmentation is available)
    if slice_idx is None:
        if segmentation is not None:
            # Find slice with maximum tumor area
            tumor_sum_per_slice = np.sum(segmentation > 0, axis=(0, 2))
            slice_idx = np.argmax(tumor_sum_per_slice)
        else:
            # Use middle slice
            slice_idx = t1n.shape[1] // 2
    
    # Number of methods + original image + segmentation
    num_methods = len(results_dict)
    fig_width = min(20, num_methods * 4)
    
    # Visualize
    plt.figure(figsize=(fig_width, 8))
    
    # Original image
    plt.subplot(2, num_methods + 1, 1)
    plt.imshow(np.rot90(t1n[:, slice_idx, :]), cmap='gray')
    plt.title(f'Original T1 (Slice {slice_idx})')
    plt.axis('off')
    
    # Segmentation (if available)
    if segmentation is not None:
        plt.subplot(2, num_methods + 1, 2)
        plt.imshow(np.rot90(t1n[:, slice_idx, :]), cmap='gray')
        seg_overlay = np.rot90(segmentation[:, slice_idx, :] > 0)
        plt.imshow(seg_overlay, alpha=0.6, cmap='hot')
        plt.title('Ground Truth')
        plt.axis('off')
    
    # Anomaly scores for each method
    for i, (method_name, result) in enumerate(results_dict.items()):
        plt.subplot(2, num_methods + 1, 3 + i)
        plt.imshow(np.rot90(t1n[:, slice_idx, :]), cmap='gray')
        anomaly_overlay = np.rot90(result['combined_score'][:, slice_idx, :])
        plt.imshow(anomaly_overlay, alpha=0.6, cmap='hot')
        
        # Add AUC to title if available
        if 'auc' in result:
            plt.title(f'{method_name}\nAUC: {result["auc"]:.3f}')
        else:
            plt.title(method_name)
        
        plt.axis('off')
    
    plt.suptitle(f'Anomaly Detection Methods Comparison - {patient_id}')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{patient_id}_methods_comparison.png")
    plt.close()

def compare_methods(args):
    """Compare different anomaly detection methods"""
    # Setup
    model, device = setup_anatomix()
    
    # Get BraTS dataset paths
    print("Gathering BraTS dataset paths...")
    samples = get_brats_paths(args.data_path)
    
    # Split into train (normal) and test sets
    patient_ids = list(samples.keys())
    
    # Use a subset of data if specified
    if args.max_samples > 0 and args.max_samples < len(patient_ids):
        patient_ids = patient_ids[:args.max_samples]
    
    if args.train_ratio > 0:
        # Use a portion of data for training
        np.random.seed(42)
        np.random.shuffle(patient_ids)
        split_idx = int(len(patient_ids) * args.train_ratio)
        train_ids = patient_ids[:split_idx]
        test_ids = patient_ids[split_idx:]
    else:
        # Use all data for both training and testing (leave-one-out approach)
        train_ids = patient_ids
        test_ids = patient_ids
    
    print(f"Using {len(train_ids)} samples for training and {len(test_ids)} for testing")
    
    # Extract features for all samples
    print("Extracting features for training samples...")
    train_features = {}
    for patient_id in tqdm(train_ids):
        features, seg = extract_all_modality_features(model, device, samples[patient_id])
        flattened_features = compute_feature_vectors(features)
        train_features[patient_id] = flattened_features
    
    # For each modality, collect all feature vectors
    modalities = ['t1n', 't1c', 't2w', 't2f']
    modality_features = {modality: [] for modality in modalities}
    
    for patient_id in train_ids:
        for modality in modalities:
            modality_features[modality].append(train_features[patient_id][modality])
    
    # Define the methods to compare
    methods = {
        'KNN': {
            'function': anomaly_detection_knn,
            'params': {'k': args.knn_neighbors},
            'results': {}
        },
        'Isolation Forest': {
            'function': anomaly_detection_isolation_forest,
            'params': {'n_estimators': 100, 'contamination': 0.1},
            'results': {}
        },
        'One-Class SVM': {
            'function': anomaly_detection_one_class_svm,
            'params': {'nu': 0.1, 'gamma': 'scale'},
            'results': {}
        }
    }
    
    # Now run anomaly detection on test samples with each method
    print("Running anomaly detection with different methods on test samples...")
    method_results = {}
    
    for patient_id in tqdm(test_ids):
        # Skip if it's the same as training (in leave-one-out approach)
        if args.train_ratio <= 0:
            # Remove current sample from training
            current_modality_features = {modality: [train_features[pid][modality] for pid in train_ids if pid != patient_id] 
                                        for modality in modalities}
        else:
            current_modality_features = modality_features
        
        # Extract features for test sample
        test_feat, test_seg = extract_all_modality_features(model, device, samples[patient_id])
        test_flat_feat = compute_feature_vectors(test_feat)
        
        # Store results for visualization
        patient_results = {}
        
        # Apply each method
        for method_name, method_info in methods.items():
            method_function = method_info['function']
            params = method_info['params']
            
            # Perform anomaly detection for each modality
            anomaly_scores = {}
            start_time = time.time()
            
            for modality in modalities:
                anomaly_score = method_function(
                    current_modality_features[modality],
                    test_flat_feat[modality],
                    **params
                )
                
                # Reshape anomaly score back to 3D volume
                volume_shape = nib.load(samples[patient_id][modality]).shape
                anomaly_map = anomaly_score.reshape(volume_shape)
                
                anomaly_scores[modality] = anomaly_map
            
            # Compute total processing time
            computation_time = time.time() - start_time
            
            # Combine anomaly scores from different modalities (average)
            combined_score = np.mean([anomaly_scores[m] for m in modalities], axis=0)
            
            # Calculate AUC if segmentation available
            auc_value = None
            if test_seg is not None:
                try:
                    ground_truth = (test_seg > 0).astype(int).flatten()
                    if np.sum(ground_truth) > 0 and np.sum(ground_truth) < len(ground_truth):
                        auc_value = roc_auc_score(ground_truth, combined_score.flatten())
                except Exception:
                    pass
            
            # Store results
            method_info['results'][patient_id] = {
                'individual_scores': anomaly_scores,
                'combined_score': combined_score,
                'segmentation': test_seg,
                'computation_time': computation_time,
                'auc': auc_value
            }
            
            # Store for visualization
            patient_results[method_name] = {
                'combined_score': combined_score,
                'auc': auc_value
            }
        
        # Visualize results for this patient
        if args.visualize and len(method_results) < args.max_visualize:
            visualize_comparison(
                samples[patient_id],
                patient_results,
                test_seg,
                patient_id
            )
        
        # Store results for this patient
        method_results[patient_id] = patient_results
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate and compare methods
    comparison_results = []
    for method_name, method_info in methods.items():
        print(f"\nEvaluating {method_name}...")
        result = evaluate_method(method_info['results'], method_name)
        if result:
            comparison_results.append(result)
    
    # Visualize method comparison
    if comparison_results:
        visualize_method_comparison(comparison_results, args.output_dir)

def visualize_method_comparison(comparison_results, output_dir):
    """Visualize comparison of different methods"""
    method_names = [r['method'] for r in comparison_results]
    roc_aucs = [r['roc_auc_mean'] for r in comparison_results]
    roc_stds = [r['roc_auc_std'] for r in comparison_results]
    pr_aucs = [r['pr_auc_mean'] for r in comparison_results]
    pr_stds = [r['pr_auc_std'] for r in comparison_results]
    times = [r['time_mean'] for r in comparison_results]
    time_stds = [r['time_std'] for r in comparison_results]
    
    # Create bar plots
    plt.figure(figsize=(15, 10))
    
    # ROC AUC
    plt.subplot(2, 2, 1)
    bars = plt.bar(method_names, roc_aucs, yerr=roc_stds, capsize=5)
    plt.ylim(0, 1)
    plt.title('ROC AUC')
    plt.ylabel('AUC')
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{roc_aucs[i]:.3f}', ha='center')
    
    # PR AUC
    plt.subplot(2, 2, 2)
    bars = plt.bar(method_names, pr_aucs, yerr=pr_stds, capsize=5)
    plt.ylim(0, 1)
    plt.title('Precision-Recall AUC')
    plt.ylabel('AUC')
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{pr_aucs[i]:.3f}', ha='center')
    
    # Computation time
    plt.subplot(2, 2, 3)
    bars = plt.bar(method_names, times, yerr=time_stds, capsize=5)
    plt.title('Computation Time')
    plt.ylabel('Time (s)')
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{times[i]:.3f}s', ha='center')
    
    plt.suptitle('Comparison of Anomaly Detection Methods')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'))
    plt.close()
    
    # Save results to CSV
    import csv
    with open(os.path.join(output_dir, 'method_comparison.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'ROC AUC', 'ROC AUC STD', 'PR AUC', 'PR AUC STD', 'Time (s)', 'Time STD'])
        for r in comparison_results:
            writer.writerow([
                r['method'], 
                r['roc_auc_mean'], 
                r['roc_auc_std'],
                r['pr_auc_mean'],
                r['pr_auc_std'],
                r['time_mean'],
                r['time_std']
            ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Different Anomaly Detection Methods')
    parser.add_argument('--data_path', type=str, 
                        default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData',
                        help='Path to BraTS dataset')
    parser.add_argument('--output_dir', type=str, default='results_comparison',
                        help='Output directory for results')
    parser.add_argument('--knn_neighbors', type=int, default=5,
                        help='Number of neighbors for KNN anomaly detection')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data to use for training (0 for leave-one-out)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Maximum number of samples to use (0 for all)')
    parser.add_argument('--max_visualize', type=int, default=5,
                        help='Maximum number of patients to visualize')
    
    args = parser.parse_args()
    compare_methods(args) 