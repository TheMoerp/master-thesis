import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from monai.utils import set_determinism
from monai.visualize import blend_images, matshow3d
import nibabel as nib

from data_preparation import prepare_ribfrac_dataset
from model import AutoEncoder3D, AnomalyDetector

def load_model(checkpoint_path, model, device):
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model architecture
        device: Device to load the model to
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, device, output_dir=None):
    """
    Evaluate the model on test data.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        output_dir: Directory to save results
        
    Returns:
        Evaluation metrics
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    all_error_scores = []
    all_has_fracture = []
    
    # Process test data
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            has_fracture = batch["has_fracture"]
            
            # Get reconstruction and error
            reconstructions, error_maps, error_scores = model.get_reconstruction_error(images)
            
            # Store results
            all_error_scores.extend(error_scores.cpu().numpy())
            all_has_fracture.extend(has_fracture.cpu().numpy())
            
            # Save some example visualizations
            if batch_idx < 5 and output_dir is not None:
                for i in range(min(2, images.shape[0])):
                    # Convert to numpy for visualization
                    image = images[i, 0].cpu().numpy()  # First channel
                    recon = reconstructions[i, 0].cpu().numpy()
                    error = error_maps[i, 0].cpu().numpy()
                    label = labels[i, 0].cpu().numpy()
                    
                    # Create mid-slice visualizations
                    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                    
                    # Get middle slices for each dimension
                    slice_idx_x = image.shape[0] // 2
                    slice_idx_y = image.shape[1] // 2
                    slice_idx_z = image.shape[2] // 2
                    
                    # Original image
                    axes[0, 0].imshow(image[slice_idx_x, :, :], cmap='gray')
                    axes[0, 0].set_title('Original (X-slice)')
                    axes[0, 1].imshow(image[:, slice_idx_y, :], cmap='gray')
                    axes[0, 1].set_title('Original (Y-slice)')
                    axes[0, 2].imshow(image[:, :, slice_idx_z], cmap='gray')
                    axes[0, 2].set_title('Original (Z-slice)')
                    
                    # Reconstruction
                    axes[1, 0].imshow(recon[slice_idx_x, :, :], cmap='gray')
                    axes[1, 0].set_title('Reconstruction (X-slice)')
                    axes[1, 1].imshow(recon[:, slice_idx_y, :], cmap='gray')
                    axes[1, 1].set_title('Reconstruction (Y-slice)')
                    axes[1, 2].imshow(recon[:, :, slice_idx_z], cmap='gray')
                    axes[1, 2].set_title('Reconstruction (Z-slice)')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'example_{batch_idx}_{i}_recon.png'))
                    plt.close()
                    
                    # Error visualization with label overlay
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    
                    # Normalize error map for visualization
                    error_norm = error / (np.max(error) + 1e-8)
                    
                    # Overlay label on error map
                    errors_x = error_norm[slice_idx_x, :, :]
                    errors_y = error_norm[:, slice_idx_y, :]
                    errors_z = error_norm[:, :, slice_idx_z]
                    
                    labels_x = label[slice_idx_x, :, :]
                    labels_y = label[:, slice_idx_y, :]
                    labels_z = label[:, :, slice_idx_z]
                    
                    # Create error heatmaps with label contours
                    axes[0].imshow(errors_x, cmap='hot', alpha=0.7)
                    if np.any(labels_x > 0):
                        axes[0].contour(labels_x > 0, colors='blue', linewidths=0.5)
                    axes[0].set_title('Error Map (X-slice)')
                    
                    axes[1].imshow(errors_y, cmap='hot', alpha=0.7)
                    if np.any(labels_y > 0):
                        axes[1].contour(labels_y > 0, colors='blue', linewidths=0.5)
                    axes[1].set_title('Error Map (Y-slice)')
                    
                    axes[2].imshow(errors_z, cmap='hot', alpha=0.7)
                    if np.any(labels_z > 0):
                        axes[2].contour(labels_z > 0, colors='blue', linewidths=0.5)
                    axes[2].set_title('Error Map (Z-slice)')
                    
                    plt.tight_layout()
                    has_frac = "fracture" if has_fracture[i] else "normal"
                    plt.savefig(os.path.join(output_dir, f'example_{batch_idx}_{i}_error_{has_frac}.png'))
                    plt.close()
    
    # Convert to numpy arrays
    all_error_scores = np.array(all_error_scores)
    all_has_fracture = np.array(all_has_fracture)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_has_fracture, all_error_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve and average precision
    precision, recall, pr_thresholds = precision_recall_curve(all_has_fracture, all_error_scores)
    average_precision = average_precision_score(all_has_fracture, all_error_scores)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate performance at optimal threshold
    y_pred = all_error_scores >= optimal_threshold
    correct = y_pred == all_has_fracture
    accuracy = np.mean(correct)
    
    # Plot ROC curve
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
                label=f'Optimal threshold: {optimal_threshold:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Plot precision-recall curve
    plt.subplot(2, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # Plot error score distributions
    plt.subplot(2, 2, 3)
    plt.hist(all_error_scores[all_has_fracture == 0], bins=30, alpha=0.5, label='Normal', color='green')
    plt.hist(all_error_scores[all_has_fracture == 1], bins=30, alpha=0.5, label='Fracture', color='red')
    plt.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Threshold: {optimal_threshold:.3f}')
    plt.xlabel('Reconstruction Error Score')
    plt.ylabel('Count')
    plt.title('Error Score Distribution')
    plt.legend()
    
    # Add summary of results
    plt.subplot(2, 2, 4)
    plt.axis('off')
    result_text = (
        f"Evaluation Results:\n\n"
        f"ROC AUC: {roc_auc:.4f}\n"
        f"Average Precision: {average_precision:.4f}\n"
        f"Optimal Threshold: {optimal_threshold:.4f}\n"
        f"Accuracy at Optimal Threshold: {accuracy:.4f}\n\n"
        f"Number of Test Samples: {len(all_has_fracture)}\n"
        f"Number of Normal Samples: {np.sum(all_has_fracture == 0)}\n"
        f"Number of Fracture Samples: {np.sum(all_has_fracture == 1)}"
    )
    plt.text(0.1, 0.9, result_text, fontsize=10, va='top', ha='left')
    
    plt.tight_layout()
    
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'))
        
        # Save results as numpy file for later analysis
        np.savez(os.path.join(output_dir, 'evaluation_results.npz'),
                 error_scores=all_error_scores,
                 has_fracture=all_has_fracture,
                 fpr=fpr,
                 tpr=tpr,
                 thresholds=thresholds,
                 roc_auc=roc_auc,
                 precision=precision,
                 recall=recall,
                 pr_thresholds=pr_thresholds,
                 average_precision=average_precision,
                 optimal_threshold=optimal_threshold,
                 accuracy=accuracy)
    
    plt.show()
    
    # Print results
    print("\nEvaluation Results:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy at Optimal Threshold: {accuracy:.4f}")
    print(f"Number of Test Samples: {len(all_has_fracture)}")
    print(f"Number of Normal Samples: {np.sum(all_has_fracture == 0)}")
    print(f"Number of Fracture Samples: {np.sum(all_has_fracture == 1)}")
    
    return {
        'roc_auc': roc_auc,
        'average_precision': average_precision,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }

def main(args):
    # Set deterministic behavior for reproducibility
    set_determinism(seed=args.seed)
    
    # Device selection - use CUDA if available and not explicitly disabled
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA - GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA not available or disabled)")
    
    # Prepare datasets and data loaders
    print("Preparing dataset...")
    _, _, test_loader = prepare_ribfrac_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = AutoEncoder3D(
        in_channels=1,
        out_channels=1,
        features=[16, 32, 64]
    )
    
    # Load trained model
    print(f"Loading model from {args.checkpoint_path}...")
    model = load_model(args.checkpoint_path, model, device)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir
    )
    
    print("Evaluation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D autoencoder for rib fracture anomaly detection")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Path to RibFrac dataset")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--cache_rate", type=float, default=0.5, help="Cache rate for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataset preparation")
    
    args = parser.parse_args()
    
    main(args) 