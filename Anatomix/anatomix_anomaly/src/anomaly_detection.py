import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm
import nibabel as nib
import torch
from pathlib import Path

def load_features(features_dir, feature_level=0):
    """
    Load extracted Anatomix features from the features directory.
    
    Args:
        features_dir: Directory containing feature files
        feature_level: Feature level to use for anomaly detection (default=0)
    
    Returns:
        Dictionary of features by subject and modality
    """
    features_dict = {}
    for file in tqdm(os.listdir(features_dir), desc="Loading features"):
        if file.endswith('_features.npz'):
            # Extract subject and modality from filename
            parts = file.split('_features.npz')[0].split('_')
            if len(parts) >= 2:
                subject = '_'.join(parts[:-1])  # Everything except the last part
                modality = parts[-1]  # Last part is the modality
                
                # Load features for specific level
                feature_file = os.path.join(features_dir, file)
                data = np.load(feature_file)
                
                # Get the requested feature level
                level_key = f'level_{feature_level}'
                if level_key in data:
                    features = data[level_key]
                    
                    # Store the features
                    if subject not in features_dict:
                        features_dict[subject] = {}
                    features_dict[subject][modality] = features
    
    return features_dict

def prepare_features_for_clustering(features_dict, modality='t1ce'):
    """
    Prepare features for clustering by selecting a specific modality
    and reshaping the features to a 2D array.
    
    Args:
        features_dict: Dictionary of features by subject and modality
        modality: Modality to use for anomaly detection (default='t1ce')
    
    Returns:
        subject_ids: List of subject IDs
        X: 2D array of features (n_samples, n_features)
    """
    subject_ids = []
    feature_arrays = []
    
    for subject, modalities in features_dict.items():
        if modality in modalities:
            subject_ids.append(subject)
            
            # Get feature array for the specified modality
            feature_array = modalities[modality]
            
            # Flatten the feature array to a 1D vector
            # We exclude the batch dimension (first dimension)
            flattened = feature_array.reshape(feature_array.shape[0], -1)
            
            # Take the first item (batch dimension)
            feature_arrays.append(flattened[0])
    
    # Stack all feature arrays into a single 2D array
    X = np.stack(feature_arrays)
    
    return subject_ids, X

def reduce_dimensionality(X, n_components=50):
    """
    Reduce the dimensionality of the features using PCA
    
    Args:
        X: 2D array of features (n_samples, n_features)
        n_components: Number of PCA components (default=50)
    
    Returns:
        X_reduced: PCA-reduced features
        pca: Fitted PCA object
    """
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]))
    X_reduced = pca.fit_transform(X_scaled)
    
    print(f"Reduced dimensionality from {X.shape[1]} to {X_reduced.shape[1]}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    
    return X_reduced, pca, scaler

def apply_kmeans_clustering(X, n_clusters=2):
    """
    Apply K-means clustering to the features
    
    Args:
        X: 2D array of features (n_samples, n_features)
        n_clusters: Number of clusters (default=2)
    
    Returns:
        kmeans: Fitted KMeans object
    """
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    return kmeans, cluster_labels

def calculate_anomaly_scores(X, kmeans):
    """
    Calculate anomaly scores as distances to nearest cluster center
    
    Args:
        X: 2D array of features (n_samples, n_features)
        kmeans: Fitted KMeans object
    
    Returns:
        anomaly_scores: Array of anomaly scores
    """
    # Calculate distance to nearest cluster center
    distances = np.min(np.sqrt(np.sum((X[:, np.newaxis, :] - kmeans.cluster_centers_[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
    
    return distances

def plot_anomaly_results(X_reduced, cluster_labels, anomaly_scores, subject_ids, output_dir):
    """
    Plot anomaly detection results
    
    Args:
        X_reduced: PCA-reduced features
        cluster_labels: Cluster labels from K-means
        anomaly_scores: Anomaly scores
        subject_ids: List of subject IDs
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scatter plot of first two PCA components colored by cluster
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Cluster Assignments (PCA)')
    plt.savefig(os.path.join(output_dir, 'cluster_assignments.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot of first two PCA components colored by anomaly score
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=anomaly_scores, cmap='plasma', alpha=0.8)
    plt.colorbar(scatter, label='Anomaly Score')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Anomaly Scores (PCA)')
    plt.savefig(os.path.join(output_dir, 'anomaly_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Histogram of anomaly scores
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores, bins=20, alpha=0.7)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.savefig(os.path.join(output_dir, 'anomaly_score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save anomaly scores and subject IDs to a CSV file
    results = np.column_stack((subject_ids, anomaly_scores, cluster_labels))
    header = 'Subject_ID,Anomaly_Score,Cluster'
    np.savetxt(os.path.join(output_dir, 'anomaly_results.csv'), results, delimiter=',', fmt='%s', header=header)
    
    # 5. Find top N anomalies
    top_n = min(10, len(anomaly_scores))
    top_indices = np.argsort(anomaly_scores)[-top_n:][::-1]
    
    print(f"\nTop {top_n} anomalies:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. Subject: {subject_ids[idx]}, Anomaly Score: {anomaly_scores[idx]:.4f}, Cluster: {cluster_labels[idx]}")
        
    return top_indices

def main(features_dir, output_dir, modality='t1ce', feature_level=0, n_clusters=2, n_components=50):
    """
    Main function for anomaly detection
    
    Args:
        features_dir: Directory containing feature files
        output_dir: Directory to save results
        modality: Modality to use for anomaly detection
        feature_level: Feature level to use
        n_clusters: Number of clusters for K-means
        n_components: Number of PCA components
    """
    # 1. Load features
    features_dict = load_features(features_dir, feature_level)
    
    # 2. Prepare features for clustering
    subject_ids, X = prepare_features_for_clustering(features_dict, modality)
    
    # 3. Reduce dimensionality
    X_reduced, pca, scaler = reduce_dimensionality(X, n_components)
    
    # 4. Apply K-means clustering
    kmeans, cluster_labels = apply_kmeans_clustering(X_reduced, n_clusters)
    
    # 5. Calculate anomaly scores
    anomaly_scores = calculate_anomaly_scores(X_reduced, kmeans)
    
    # 6. Plot results
    top_indices = plot_anomaly_results(X_reduced, cluster_labels, anomaly_scores, subject_ids, output_dir)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly detection using Anatomix features and K-means")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory containing extracted features")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--modality", type=str, default="t1ce", help="Modality to use for anomaly detection")
    parser.add_argument("--feature_level", type=int, default=0, help="Feature level to use (0-3)")
    parser.add_argument("--n_clusters", type=int, default=2, help="Number of clusters for K-means")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components")
    
    args = parser.parse_args()
    
    main(args.features_dir, args.output_dir, args.modality, args.feature_level, args.n_clusters, args.n_components) 