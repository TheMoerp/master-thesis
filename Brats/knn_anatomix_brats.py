import os
import argparse
import time
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import faiss
import sys
import contextlib

# Disable memory statistics output
os.environ['PYTHONMALLOCSTATS'] = '0'

# Ensure anatomix is installed
def install_anatomix():
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
    # Temporarily suppress stdout during import
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
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # Added epsilon to avoid division by zero
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
                         (percentage of voxels that can be abnormal in a "normal" patch)
    """
    h, w, d = image.shape
    patches = []
    labels = []
    coordinates = []
    
    if h < patch_size or w < patch_size or d < patch_size:
        raise ValueError(f"Image dimensions {image.shape} are smaller than patch size {patch_size}")
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            for k in range(0, d - patch_size + 1, stride):
                patch = image[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                mask_patch = seg_mask[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                
                # Only include patches that have brain tissue (not just background)
                if np.mean(patch) > 0.05:  # Increased threshold for brain tissue
                    # Check if the patch contains abnormality
                    abnormal_voxels = np.sum(mask_patch > 0)
                    total_voxels = patch_size**3
                    abnormal_ratio = abnormal_voxels / total_voxels
                    
                    # Label the patch
                    label = 1 if abnormal_ratio > normal_threshold else 0
                    
                    patches.append(patch)
                    labels.append(label)
                    coordinates.append((i, j, k))
    
    return patches, labels, coordinates

def visualize_patches(patches, labels, coordinates, original_image, original_mask, num_samples=5, patch_size=32):
    """Visualize random patches with their labels and locations in the original image"""
    if len(patches) == 0:
        print("No patches to visualize")
        return
    
    # Select random indices for visualization
    if num_samples > len(patches):
        num_samples = len(patches)
    
    indices = np.random.choice(len(patches), num_samples, replace=False)
    
    # For each patch, we'll show 3 slices: front, middle, and back
    slice_positions = [patch_size//4, patch_size//2, 3*patch_size//4]
    
    for idx in indices:
        patch = patches[idx]
        label = labels[idx]
        coord = coordinates[idx]
        x, y, z = coord
        
        # Create a figure with 3 rows (for different slices) and 3 columns
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f"Patch Analysis (Label: {'Abnormal' if label else 'Normal'})", fontsize=16)
        
        # Row titles
        row_names = ['Front Slice', 'Middle Slice', 'Back Slice']
        col_names = ['Patch', 'Location in Image', 'Segmentation']
        
        for row, slice_pos in enumerate(slice_positions):
            # Show the patch slice
            axes[row, 0].imshow(patch[:, :, slice_pos], cmap='gray')
            axes[row, 0].set_title(f"{row_names[row]} - Patch")
            
            # Show the location in original image
            img_slice = original_image[:, :, z + slice_pos]
            axes[row, 1].imshow(img_slice, cmap='gray')
            axes[row, 1].add_patch(plt.Rectangle((y, x), patch_size, patch_size, 
                                               linewidth=2, edgecolor='r', facecolor='none'))
            axes[row, 1].set_title(f"{row_names[row]} - Location")
            
            # Show the segmentation overlay
            mask_overlay = np.zeros((*original_image.shape[:2], 3))
            mask_slice = original_mask[:, :, z + slice_pos]
            mask_overlay[:, :, 0] = np.where(mask_slice > 0, 1, 0)
            
            axes[row, 2].imshow(img_slice, cmap='gray')
            axes[row, 2].imshow(mask_overlay, alpha=0.3)
            axes[row, 2].add_patch(plt.Rectangle((y, x), patch_size, patch_size,
                                               linewidth=2, edgecolor='r', facecolor='none'))
            axes[row, 2].set_title(f"{row_names[row]} - Segmentation")
            
            # Add patch statistics
            if row == 0:
                abnormal_voxels = np.sum(original_mask[x:x+patch_size, 
                                                     y:y+patch_size, 
                                                     z:z+patch_size] > 0)
                total_voxels = patch_size**3
                abnormal_ratio = abnormal_voxels / total_voxels
                stats_text = f"Abnormal voxels: {abnormal_voxels}\n"
                stats_text += f"Total voxels: {total_voxels}\n"
                stats_text += f"Abnormal ratio: {abnormal_ratio:.3f}"
                plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                          bbox=dict(facecolor='white', alpha=0.8))
        
        # Turn off axis for all subplots
        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"patch_visualization_{idx}.png")
        plt.close()

def load_anatomix_model():
    """Load pre-trained anatomix model"""
    model = Unet(
        dimension=3,      # Only 3D supported for now
        input_nc=1,       # number of input channels
        output_nc=16,     # number of output channels
        num_downs=4,      # number of downsampling layers
        ngf=16,           # channel multiplier
    ).cuda()
    
    # Check if weights are already downloaded
    if not os.path.exists("anatomix/model-weights/anatomix.pth"):
        if not os.path.exists("anatomix/model-weights"):
            os.makedirs("anatomix/model-weights", exist_ok=True)
        
        print("Downloading anatomix model weights...")
        os.system("wget -O anatomix/model-weights/anatomix.pth https://github.com/neel-dey/anatomix/raw/main/model-weights/anatomix.pth")
    
    model.load_state_dict(
        torch.load("anatomix/model-weights/anatomix.pth"),
        strict=True,
    )
    model.eval()
    return model

def extract_features(model, patches):
    """Extract features from patches using anatomix model"""
    features = []
    batch_size = 4  # Adjust based on GPU memory
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Extracting features"):
            batch_patches = patches[i:i+batch_size]
            # Convert to torch tensor with shape (B, 1, D, H, W)
            batch_tensor = torch.tensor(np.array(batch_patches), dtype=torch.float32)
            batch_tensor = batch_tensor.unsqueeze(1).cuda()
            
            # Extract features
            batch_features = model(batch_tensor)
            
            # Average pool across spatial dimensions to get a feature vector per patch
            batch_features = torch.mean(batch_features, dim=(2, 3, 4))
            
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features)

def visualize_feature_embeddings(features, labels, n_samples=1000):
    """Visualize feature embeddings using t-SNE"""
    from sklearn.manifold import TSNE
    
    # Subsample for faster visualization if needed
    if len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        features_subset = features[indices]
        labels_subset = [labels[i] for i in indices]
    else:
        features_subset = features
        labels_subset = labels
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedded_features = tsne.fit_transform(features_subset)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1], 
                         c=labels_subset, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Label (0: Normal, 1: Abnormal)')
    plt.title('t-SNE Visualization of Feature Embeddings')
    plt.savefig("feature_embeddings.png")
    plt.close()

def build_faiss_index(features, labels, device_id=0):
    """Build a GPU-accelerated KNN index using faiss"""
    d = features.shape[1]  # Feature dimension
    
    # Convert features to float32 if needed
    if features.dtype != np.float32:
        features = features.astype(np.float32)
    
    # Create a flat index on GPU with reduced memory usage
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = device_id
    config.useFloat16 = True  # Use half precision to reduce memory usage
    
    # Create index with memory-efficient settings
    index = faiss.GpuIndexFlatL2(res, d, config)
    
    # Add vectors in batches to reduce memory usage
    batch_size = 10000
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        index.add(batch)
    
    # Store labels for retrieval
    xb_labels = np.array(labels, dtype=np.int64)
    
    return index, xb_labels

def knn_anomaly_detection(index, xb_labels, test_features, k=5):
    """
    Perform KNN-based anomaly detection with class weighting
    """
    # Process test features in batches
    batch_size = 1000
    anomaly_scores = []
    pred_labels = []
    
    # Calculate class weights from training labels
    n_normal = sum(xb_labels == 0)
    n_abnormal = sum(xb_labels == 1)
    normal_weight = 1.0
    abnormal_weight = n_normal / n_abnormal if n_abnormal > 0 else 1.0
    
    for i in range(0, len(test_features), batch_size):
        batch = test_features[i:i+batch_size]
        # Search for k nearest neighbors
        D, I = index.search(batch, k)
        
        # Get labels of nearest neighbors
        nn_labels = np.array([xb_labels[i] for i in I])
        
        # Apply class weights to neighbor votes
        weighted_votes = np.where(nn_labels == 1, abnormal_weight, normal_weight)
        
        # Compute weighted anomaly score
        batch_scores = np.sum(weighted_votes * nn_labels, axis=1) / np.sum(weighted_votes, axis=1)
        anomaly_scores.extend(batch_scores)
        
        # Use a lower threshold due to class weighting
        batch_preds = (batch_scores > 0.3).astype(int)
        pred_labels.extend(batch_preds)
    
    return np.array(anomaly_scores), np.array(pred_labels)

def evaluate_performance(true_labels, anomaly_scores, pred_labels):
    """Evaluate anomaly detection performance"""
    # ROC AUC score (using anomaly scores)
    roc_auc = roc_auc_score(true_labels, anomaly_scores)
    
    # Average Precision score (using anomaly scores)
    ap = average_precision_score(true_labels, anomaly_scores)
    
    # Calculate accuracy, precision, recall, and F1 score (using predicted labels)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    # Create a dictionary of metrics
    metrics = {
        'ROC AUC': roc_auc,
        'Average Precision': ap,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    return metrics

def visualize_anomaly_detection(test_patches, test_coords, original_images, original_masks, 
                               true_labels, pred_labels, anomaly_scores, patch_size=32, num_samples=5):
    """Visualize anomaly detection results"""
    # Select random samples with both correct and incorrect predictions
    correct_indices = np.where(np.array(true_labels) == np.array(pred_labels))[0]
    incorrect_indices = np.where(np.array(true_labels) != np.array(pred_labels))[0]
    
    selected_indices = []
    
    # Add correct predictions
    if len(correct_indices) > 0:
        n_correct = min(num_samples // 2, len(correct_indices))
        selected_indices.extend(np.random.choice(correct_indices, n_correct, replace=False))
    
    # Add incorrect predictions
    if len(incorrect_indices) > 0:
        n_incorrect = min(num_samples - len(selected_indices), len(incorrect_indices))
        selected_indices.extend(np.random.choice(incorrect_indices, n_incorrect, replace=False))
    
    # Fill remaining spots with random samples if needed
    remaining = num_samples - len(selected_indices)
    if remaining > 0:
        all_indices = np.arange(len(test_patches))
        mask = np.ones(len(test_patches), dtype=bool)
        mask[selected_indices] = False
        remaining_indices = all_indices[mask]
        selected_indices.extend(np.random.choice(remaining_indices, remaining, replace=False))
    
    # Create visualization
    fig, axes = plt.subplots(len(selected_indices), 3, figsize=(15, 5 * len(selected_indices)))
    
    for i, idx in enumerate(selected_indices):
        patch = test_patches[idx]
        coord = test_coords[idx]
        true_label = true_labels[idx]
        pred_label = pred_labels[idx]
        score = anomaly_scores[idx]
        
        # Get the corresponding original image and mask
        img_idx = 0  # This needs to be updated if patches come from multiple images
        original_image = original_images[img_idx]
        original_mask = original_masks[img_idx]
        
        # Get the middle slice of the patch
        mid_slice = patch_size // 2
        
        # Show the patch
        axes[i, 0].imshow(patch[:, :, mid_slice], cmap='gray')
        axes[i, 0].set_title(f"Patch (True: {'Abnormal' if true_label else 'Normal'}, "
                            f"Pred: {'Abnormal' if pred_label else 'Normal'}, "
                            f"Score: {score:.2f})")
        axes[i, 0].axis('off')
        
        # Show the location in the original image
        x, y, z = coord
        axes[i, 1].imshow(original_image[:, :, z + mid_slice], cmap='gray')
        axes[i, 1].add_patch(plt.Rectangle((y, x), patch_size, patch_size, linewidth=2, 
                                        edgecolor='g' if true_label == pred_label else 'r', 
                                        facecolor='none'))
        axes[i, 1].set_title("Location in Original Image")
        axes[i, 1].axis('off')
        
        # Show the segmentation mask for the same slice
        mask_overlay = np.zeros((*original_image.shape[:2], 3))
        mask_slice = original_mask[:, :, z + mid_slice]
        
        # Create RGB overlay: Red for abnormal regions
        mask_overlay[:, :, 0] = np.where(mask_slice > 0, 1, 0)
        
        axes[i, 2].imshow(original_image[:, :, z + mid_slice], cmap='gray')
        axes[i, 2].imshow(mask_overlay, alpha=0.3)
        axes[i, 2].add_patch(plt.Rectangle((y, x), patch_size, patch_size, linewidth=2, 
                                        edgecolor='g' if true_label == pred_label else 'r', 
                                        facecolor='none'))
        axes[i, 2].set_title("Segmentation Overlay")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("anomaly_detection_results.png")
    plt.close()

def main(args):
    # Start timing
    start_time = time.time()
    
    # Load anatomix model
    model = load_anatomix_model()
    
    # Get list of all BraTS data paths
    data_dir = os.path.join(args.data_dir, "BraTS2025-GLI-PRE-Challenge-TrainingData")
    subject_folders = sorted(glob(os.path.join(data_dir, "BraTS-GLI-*")))
    
    # Limit the number of subjects if specified
    if args.num_subjects > 0:
        subject_folders = subject_folders[:args.num_subjects]
    
    print(f"Processing {len(subject_folders)} subjects")
    
    # Process data and extract patches
    all_patches = []
    all_labels = []
    all_coords = []
    original_images = []
    original_masks = []
    
    for subject_folder in tqdm(subject_folders, desc="Processing subjects"):
        # Load images (T1, T1c, T2, FLAIR) and segmentation mask
        t1_path = glob(os.path.join(subject_folder, "*-t1n.nii.gz"))[0]
        t1c_path = glob(os.path.join(subject_folder, "*-t1c.nii.gz"))[0]
        t2_path = glob(os.path.join(subject_folder, "*-t2w.nii.gz"))[0]
        flair_path = glob(os.path.join(subject_folder, "*-t2f.nii.gz"))[0]
        seg_path = glob(os.path.join(subject_folder, "*-seg.nii.gz"))[0]
        
        # Load the images using nibabel
        t1_nib = nib.load(t1_path)
        t1c_nib = nib.load(t1c_path)
        t2_nib = nib.load(t2_path)
        flair_nib = nib.load(flair_path)
        seg_nib = nib.load(seg_path)
        
        # Get the image data
        t1 = t1_nib.get_fdata()
        t1c = t1c_nib.get_fdata()
        t2 = t2_nib.get_fdata()
        flair = flair_nib.get_fdata()
        seg = seg_nib.get_fdata()
        
        # Normalize images
        t1_norm = minmax(t1)
        t1c_norm = minmax(t1c)
        t2_norm = minmax(t2)
        flair_norm = minmax(flair)
        
        # Use T1c for feature extraction (shows tumor enhancement well)
        input_image = t1c_norm
        
        # Extract patches with adjusted threshold
        patches, labels, coords = extract_3d_patches(
            input_image, seg, 
            patch_size=args.patch_size, 
            stride=args.stride, 
            normal_threshold=args.normal_threshold
        )
        
        # Store original images for visualization
        original_images.append(input_image)
        original_masks.append(seg)
        
        # Add to overall lists
        all_patches.extend(patches)
        all_labels.extend(labels)
        all_coords.extend(coords)
    
    print(f"Extracted {len(all_patches)} patches ({sum(all_labels)} abnormal, {len(all_labels) - sum(all_labels)} normal)")
    
    # Visualize some patches
    if args.visualize:
        visualize_patches(all_patches, all_labels, all_coords, 
                         original_images[0], original_masks[0],
                         num_samples=5, patch_size=args.patch_size)
    
    # Convert patches to numpy array for batch processing
    all_patches_np = np.array(all_patches)
    
    # Extract features using anatomix model
    features = extract_features(model, all_patches_np)
    print(f"Extracted features with shape: {features.shape}")
    
    # Visualize feature embeddings
    if args.visualize:
        visualize_feature_embeddings(features, all_labels)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, train_coords, test_coords = train_test_split(
        features, all_labels, all_coords, test_size=args.test_size, random_state=42, stratify=all_labels
    )
    
    # Also split patches for visualization
    _, test_patches, _, _, _, _ = train_test_split(
        all_patches, all_labels, all_coords, test_size=args.test_size, random_state=42, stratify=all_labels
    )
    
    print(f"Train set: {len(X_train)} samples ({sum(y_train)} abnormal, {len(y_train) - sum(y_train)} normal)")
    print(f"Test set: {len(X_test)} samples ({sum(y_test)} abnormal, {len(y_test) - sum(y_test)} normal)")
    
    # Build KNN index with faiss (GPU accelerated)
    index, xb_labels = build_faiss_index(X_train, y_train)
    
    # Perform anomaly detection on test set
    anomaly_scores, pred_labels = knn_anomaly_detection(index, xb_labels, X_test, k=args.k_neighbors)
    
    # Evaluate performance
    metrics = evaluate_performance(y_test, anomaly_scores, pred_labels)
    
    # Print metrics
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize anomaly detection results
    if args.visualize:
        visualize_anomaly_detection(test_patches, test_coords, original_images, original_masks,
                                  y_test, pred_labels, anomaly_scores, 
                                  patch_size=args.patch_size, num_samples=5)
    
    # Report total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS Anomaly Detection using anatomix features and KNN")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Path to the dataset directory")
    parser.add_argument("--num_subjects", type=int, default=-1, help="Number of subjects to process (-1 for all)")
    parser.add_argument("--patch_size", type=int, default=32, help="Size of 3D patches")
    parser.add_argument("--stride", type=int, default=16, help="Stride for patch extraction")
    parser.add_argument("--normal_threshold", type=float, default=0.05, help="Threshold for labeling patches as normal/abnormal")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--k_neighbors", type=int, default=7, help="Number of neighbors for KNN")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of patches and results")
    
    args = parser.parse_args()
    main(args) 