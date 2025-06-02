#!/usr/bin/env python3
"""
Autoencoder-based Anomaly Detection for 3D BRATS Dataset
========================================================

This script implements an autoencoder for anomaly detection on the BRATS dataset.
Since BRATS contains only abnormal data, healthy 3D patches are extracted from
abnormal volumes to serve as training data.

Usage: python ae_brats2.py --subjects <number_of_subjects>
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try PyTorch first, fallback to TensorFlow
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    from tqdm import tqdm
    FRAMEWORK = 'pytorch'
    print("Using PyTorch")
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        FRAMEWORK = 'tensorflow'
        print("Using TensorFlow")
        # Enable GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except ImportError:
        raise ImportError("Neither PyTorch nor TensorFlow is available!")


class BRATSDataProcessor:
    """Handles BRATS data loading and preprocessing"""
    
    def __init__(self, data_path=".", patch_size=(32, 32, 32), stride=16):
        self.data_path = Path(data_path)
        self.patch_size = patch_size
        self.stride = stride
        self.healthy_patches = []
        self.abnormal_patches = []
        
    def find_brats_files(self, max_subjects=None):
        """Find BRATS NIfTI files"""
        patterns = [
            "**/BraTS*/*_t1.nii*",
            "**/BraTS*/*_flair.nii*", 
            "**/*_t1.nii*",
            "**/*_flair.nii*",
            "**/t1.nii*",
            "**/flair.nii*"
        ]
        
        files = []
        for pattern in patterns:
            found = list(self.data_path.glob(pattern))
            files.extend(found)
            if files:
                break
        
        if not files:
            # Try current directory
            files = list(Path(".").glob("*.nii*"))
        
        if max_subjects:
            files = files[:max_subjects]
            
        print(f"Found {len(files)} BRATS files")
        return files
    
    def load_nifti(self, filepath):
        """Load NIfTI file"""
        try:
            img = nib.load(str(filepath))
            data = img.get_fdata()
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def normalize_volume(self, volume):
        """Normalize volume to [0, 1] range"""
        volume = volume.astype(np.float32)
        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min())
        return volume
    
    def extract_patches_3d(self, volume, patch_size, stride):
        """Extract 3D patches from volume"""
        patches = []
        d, h, w = volume.shape
        pd, ph, pw = patch_size
        
        for z in range(0, d - pd + 1, stride):
            for y in range(0, h - ph + 1, stride):
                for x in range(0, w - pw + 1, stride):
                    patch = volume[z:z+pd, y:y+ph, x:x+pw]
                    if patch.shape == patch_size:
                        patches.append(patch)
        
        return np.array(patches)
    
    def is_healthy_patch(self, patch, threshold=0.02):
        """Determine if patch is healthy based on intensity variance"""
        # Healthy patches should have low variance (uniform tissue)
        # and moderate mean intensity (not background, not extreme lesion)
        mean_intensity = np.mean(patch)
        std_intensity = np.std(patch)
        
        # Filter out background patches
        if mean_intensity < 0.1:
            return False
        
        # Filter out high-variance patches (likely containing lesions)
        if std_intensity > threshold:
            return False
            
        # Filter out very bright patches (likely lesions)
        if mean_intensity > 0.8:
            return False
            
        return True
    
    def process_subject(self, filepath):
        """Process a single subject file"""
        print(f"Processing: {filepath}")
        
        volume = self.load_nifti(filepath)
        if volume is None:
            return
        
        # Normalize volume
        volume = self.normalize_volume(volume)
        
        # Extract patches
        patches = self.extract_patches_3d(volume, self.patch_size, self.stride)
        
        # Classify patches as healthy or abnormal
        healthy_count = 0
        abnormal_count = 0
        
        for patch in patches:
            if self.is_healthy_patch(patch):
                self.healthy_patches.append(patch)
                healthy_count += 1
            else:
                self.abnormal_patches.append(patch)
                abnormal_count += 1
        
        print(f"  Extracted {healthy_count} healthy and {abnormal_count} abnormal patches")
    
    def visualize_patches(self, patches, title="Patches", max_display=5):
        """Visualize 3D patches as slices"""
        if len(patches) == 0:
            print(f"No {title.lower()} to visualize")
            return
            
        fig, axes = plt.subplots(max_display, self.patch_size[0], 
                                figsize=(15, max_display * 2))
        fig.suptitle(f"{title} - All Slices", fontsize=16)
        
        for i in range(min(max_display, len(patches))):
            patch = patches[i]
            for slice_idx in range(self.patch_size[0]):
                if max_display == 1:
                    ax = axes[slice_idx]
                else:
                    ax = axes[i, slice_idx]
                
                ax.imshow(patch[slice_idx], cmap='gray')
                ax.set_title(f"Patch {i+1}, Slice {slice_idx+1}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_data(self, max_subjects=None):
        """Prepare training and test data"""
        files = self.find_brats_files(max_subjects)
        
        if not files:
            raise ValueError("No BRATS files found!")
        
        # Process all subjects
        for filepath in files:
            self.process_subject(filepath)
        
        print(f"\nTotal extracted patches:")
        print(f"  Healthy: {len(self.healthy_patches)}")
        print(f"  Abnormal: {len(self.abnormal_patches)}")
        
        if len(self.healthy_patches) == 0:
            raise ValueError("No healthy patches found! Try adjusting threshold parameters.")
        
        # Visualize sample patches
        print("\nVisualizing sample healthy patches...")
        self.visualize_patches(self.healthy_patches[:3], "Healthy Patches", 3)
        
        print("Visualizing sample abnormal patches...")
        self.visualize_patches(self.abnormal_patches[:3], "Abnormal Patches", 3)
        
        # Prepare training data (only healthy patches)
        healthy_patches = np.array(self.healthy_patches)
        
        # Split healthy patches for training (80%) and testing (20%)
        train_healthy, test_healthy = train_test_split(
            healthy_patches, test_size=0.2, random_state=42
        )
        
        # For testing, use some healthy and some abnormal patches
        max_test_abnormal = min(len(test_healthy), len(self.abnormal_patches))
        test_abnormal = np.array(self.abnormal_patches[:max_test_abnormal])
        
        return train_healthy, test_healthy, test_abnormal


# PyTorch Implementation
if FRAMEWORK == 'pytorch':
    class BratsDataset(Dataset):
        def __init__(self, patches):
            self.patches = torch.FloatTensor(patches)
            if len(self.patches.shape) == 4:  # Add channel dimension
                self.patches = self.patches.unsqueeze(1)
        
        def __len__(self):
            return len(self.patches)
        
        def __getitem__(self, idx):
            return self.patches[idx], self.patches[idx]  # Input = Target for autoencoder

    class Autoencoder3D(nn.Module):
        def __init__(self, input_shape=(1, 32, 32, 32)):
            super(Autoencoder3D, self).__init__()
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                
                nn.Conv3d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                
                nn.Conv3d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                
                nn.Conv3d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((2, 2, 2))
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                
                nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                
                nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                
                nn.ConvTranspose3d(16, 1, 3, padding=1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def train_pytorch_model(train_data, epochs=50, batch_size=8, lr=0.001):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        dataset = BratsDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = Autoencoder3D().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        losses = []
        
        print(f"\nTraining autoencoder for {epochs} epochs...")
        
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return model, losses

    def evaluate_pytorch_model(model, test_healthy, test_abnormal):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        def get_reconstruction_errors(patches):
            errors = []
            dataset = BratsDataset(patches)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
            
            with torch.no_grad():
                for data, _ in dataloader:
                    data = data.to(device)
                    reconstructed = model(data)
                    
                    # Calculate MSE for each sample
                    mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2, 3, 4))
                    errors.extend(mse.cpu().numpy())
            
            return np.array(errors)
        
        print("\nEvaluating model...")
        
        healthy_errors = get_reconstruction_errors(test_healthy)
        abnormal_errors = get_reconstruction_errors(test_abnormal)
        
        return healthy_errors, abnormal_errors

# TensorFlow Implementation
else:
    def create_tensorflow_autoencoder(input_shape=(32, 32, 32, 1)):
        # Encoder
        inputs = keras.Input(shape=input_shape)
        
        x = layers.Conv3D(16, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling3D(2, padding='same')(x)
        
        x = layers.Conv3D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling3D(2, padding='same')(x)
        
        x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling3D(2, padding='same')(x)
        
        x = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
        encoded = layers.GlobalAveragePooling3D()(x)
        
        # Decoder
        x = layers.Reshape((2, 2, 2, 128))(encoded)
        
        x = layers.Conv3DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.Conv3DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.Conv3DTranspose(16, 3, strides=2, activation='relu', padding='same')(x)
        
        outputs = layers.Conv3D(1, 3, activation='sigmoid', padding='same')(x)
        
        autoencoder = keras.Model(inputs, outputs)
        return autoencoder

    def train_tensorflow_model(train_data, epochs=50, batch_size=8, lr=0.001):
        print(f"Using TensorFlow with GPU: {tf.config.list_physical_devices('GPU')}")
        
        # Prepare data
        train_data = train_data[..., np.newaxis]  # Add channel dimension
        
        model = create_tensorflow_autoencoder(train_data.shape[1:])
        model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
        
        print(f"\nTraining autoencoder for {epochs} epochs...")
        
        # Custom callback for smooth progress bar
        class SmoothProgressCallback(keras.callbacks.Callback):
            def __init__(self, epochs):
                self.epochs = epochs
                self.pbar = None
                
            def on_train_begin(self, logs=None):
                self.pbar = tqdm(total=self.epochs, desc="Training Progress")
                
            def on_epoch_end(self, epoch, logs=None):
                self.pbar.set_postfix({'loss': f"{logs.get('loss', 0):.6f}"})
                self.pbar.update(1)
                
            def on_train_end(self, logs=None):
                self.pbar.close()
        
        history = model.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[SmoothProgressCallback(epochs)]
        )
        
        return model, history.history['loss']

    def evaluate_tensorflow_model(model, test_healthy, test_abnormal):
        print("\nEvaluating model...")
        
        def get_reconstruction_errors(patches):
            patches = patches[..., np.newaxis]  # Add channel dimension
            reconstructed = model.predict(patches, verbose=0)
            errors = np.mean((patches - reconstructed) ** 2, axis=(1, 2, 3, 4))
            return errors
        
        healthy_errors = get_reconstruction_errors(test_healthy)
        abnormal_errors = get_reconstruction_errors(test_abnormal)
        
        return healthy_errors, abnormal_errors


def calculate_metrics_and_visualize(healthy_errors, abnormal_errors):
    """Calculate metrics and create visualizations"""
    
    # Prepare labels and scores
    y_true = np.concatenate([
        np.zeros(len(healthy_errors)),  # Healthy = 0 (normal)
        np.ones(len(abnormal_errors))   # Abnormal = 1 (anomaly)
    ])
    
    y_scores = np.concatenate([healthy_errors, abnormal_errors])
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_scores > optimal_threshold).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print(f"\n{'='*50}")
    print("ANOMALY DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"ROC AUC:           {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    print(f"{'='*50}")
    
    # Create visualizations
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training Loss (if available from global scope)
    plt.subplot(3, 4, 1)
    if 'training_losses' in globals():
        plt.plot(training_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Training Loss\nNot Available', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 2. Reconstruction Error Histogram
    plt.subplot(3, 4, 2)
    plt.hist(healthy_errors, bins=50, alpha=0.7, label='Healthy', density=True)
    plt.hist(abnormal_errors, bins=50, alpha=0.7, label='Abnormal', density=True)
    plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Threshold: {optimal_threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. ROC Curve
    plt.subplot(3, 4, 3)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 4. Precision-Recall Curve
    plt.subplot(3, 4, 4)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall_curve, precision_curve, color='green', lw=2, 
             label=f'PR (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # 5. Confusion Matrix
    plt.subplot(3, 4, 5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 6. Metrics Bar Plot
    plt.subplot(3, 4, 6)
    metrics = ['ROC AUC', 'Avg Precision', 'Accuracy', 'Precision', 'Recall', 'F1']
    values = [roc_auc, avg_precision, accuracy, precision, recall, f1]
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 
                                          'gold', 'plum', 'lightsalmon'])
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 7. Error Scatter Plot
    plt.subplot(3, 4, 7)
    plt.scatter(range(len(healthy_errors)), healthy_errors, 
               alpha=0.6, label='Healthy', s=20)
    plt.scatter(range(len(healthy_errors), len(healthy_errors) + len(abnormal_errors)), 
               abnormal_errors, alpha=0.6, label='Abnormal', s=20)
    plt.axhline(optimal_threshold, color='red', linestyle='--', 
                label=f'Threshold: {optimal_threshold:.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Errors by Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Box Plot of Errors
    plt.subplot(3, 4, 8)
    data_to_plot = [healthy_errors, abnormal_errors]
    box_plot = plt.boxplot(data_to_plot, labels=['Healthy', 'Abnormal'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel('Reconstruction Error')
    plt.title('Error Distribution Box Plot')
    plt.grid(True, alpha=0.3)
    
    # 9. Threshold Analysis
    plt.subplot(3, 4, 9)
    thresholds_range = np.linspace(min(y_scores), max(y_scores), 100)
    accuracies = []
    f1_scores = []
    
    for thresh in thresholds_range:
        y_pred_thresh = (y_scores > thresh).astype(int)
        acc = accuracy_score(y_true, y_pred_thresh)
        f1_thresh = f1_score(y_true, y_pred_thresh) if len(np.unique(y_pred_thresh)) > 1 else 0
        accuracies.append(acc)
        f1_scores.append(f1_thresh)
    
    plt.plot(thresholds_range, accuracies, label='Accuracy', color='blue')
    plt.plot(thresholds_range, f1_scores, label='F1 Score', color='green')
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal: {optimal_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Error Statistics
    plt.subplot(3, 4, 10)
    stats_data = {
        'Healthy Errors': {
            'Mean': np.mean(healthy_errors),
            'Std': np.std(healthy_errors),
            'Min': np.min(healthy_errors),
            'Max': np.max(healthy_errors)
        },
        'Abnormal Errors': {
            'Mean': np.mean(abnormal_errors),
            'Std': np.std(abnormal_errors),
            'Min': np.min(abnormal_errors),
            'Max': np.max(abnormal_errors)
        }
    }
    
    # Create text summary
    text_summary = "Error Statistics:\n\n"
    text_summary += "Healthy Patches:\n"
    text_summary += f"  Mean: {stats_data['Healthy Errors']['Mean']:.6f}\n"
    text_summary += f"  Std:  {stats_data['Healthy Errors']['Std']:.6f}\n"
    text_summary += f"  Min:  {stats_data['Healthy Errors']['Min']:.6f}\n"
    text_summary += f"  Max:  {stats_data['Healthy Errors']['Max']:.6f}\n\n"
    text_summary += "Abnormal Patches:\n"
    text_summary += f"  Mean: {stats_data['Abnormal Errors']['Mean']:.6f}\n"
    text_summary += f"  Std:  {stats_data['Abnormal Errors']['Std']:.6f}\n"
    text_summary += f"  Min:  {stats_data['Abnormal Errors']['Min']:.6f}\n"
    text_summary += f"  Max:  {stats_data['Abnormal Errors']['Max']:.6f}\n"
    
    plt.text(0.05, 0.95, text_summary, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    plt.title('Statistical Summary')
    
    # 11. Detection Performance by Error Range
    plt.subplot(3, 4, 11)
    error_ranges = np.percentile(y_scores, [0, 25, 50, 75, 100])
    range_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    
    detection_rates = []
    for i in range(len(error_ranges)-1):
        mask = (y_scores >= error_ranges[i]) & (y_scores < error_ranges[i+1])
        if i == len(error_ranges)-2:  # Last range includes maximum
            mask = (y_scores >= error_ranges[i]) & (y_scores <= error_ranges[i+1])
        
        if np.sum(mask) > 0:
            anomaly_rate = np.mean(y_true[mask])
            detection_rates.append(anomaly_rate)
        else:
            detection_rates.append(0)
    
    bars = plt.bar(range_labels, detection_rates, color='lightsteelblue')
    plt.title('Anomaly Rate by Error Percentile')
    plt.ylabel('Anomaly Rate')
    plt.xlabel('Error Percentile Range')
    plt.ylim([0, 1])
    
    # Add value labels
    for bar, rate in zip(bars, detection_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2f}', ha='center', va='bottom')
    
    # 12. Summary Information
    plt.subplot(3, 4, 12)
    summary_text = f"""
ANOMALY DETECTION SUMMARY

Dataset Information:
• Healthy patches: {len(healthy_errors)}
• Abnormal patches: {len(abnormal_errors)}
• Total test samples: {len(y_true)}

Model Performance:
• ROC AUC: {roc_auc:.4f}
• Average Precision: {avg_precision:.4f}
• Accuracy: {accuracy:.4f}
• F1 Score: {f1:.4f}

Threshold: {optimal_threshold:.6f}

Classification Results:
• True Positives: {cm[1,1]}
• False Positives: {cm[0,1]}
• True Negatives: {cm[0,0]}
• False Negatives: {cm[1,0]}
"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.axis('off')
    plt.title('Summary')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': optimal_threshold,
        'confusion_matrix': cm
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='3D Autoencoder Anomaly Detection for BRATS')
    parser.add_argument('--subjects', type=int, default=10,
                        help='Number of subjects to process (default: 10)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[32, 32, 32],
                        help='3D patch size (default: 32 32 32)')
    parser.add_argument('--data_path', type=str, default=".",
                        help='Path to BRATS data (default: current directory)')
    
    args = parser.parse_args()
    
    print("3D Autoencoder Anomaly Detection for BRATS Dataset")
    print("=" * 55)
    print(f"Framework: {FRAMEWORK.upper()}")
    print(f"Subjects to process: {args.subjects}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patch size: {args.patch_size}")
    print(f"Data path: {args.data_path}")
    
    # Initialize data processor
    processor = BRATSDataProcessor(
        data_path=args.data_path,
        patch_size=tuple(args.patch_size),
        stride=16
    )
    
    # Prepare data
    try:
        train_healthy, test_healthy, test_abnormal = processor.prepare_data(args.subjects)
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
    
    print(f"\nDataset split:")
    print(f"  Training (healthy): {len(train_healthy)}")
    print(f"  Testing (healthy): {len(test_healthy)}")
    print(f"  Testing (abnormal): {len(test_abnormal)}")
    
    # Train model
    if FRAMEWORK == 'pytorch':
        model, losses = train_pytorch_model(
            train_healthy, epochs=args.epochs, 
            batch_size=args.batch_size
        )
        healthy_errors, abnormal_errors = evaluate_pytorch_model(
            model, test_healthy, test_abnormal
        )
    else:
        model, losses = train_tensorflow_model(
            train_healthy, epochs=args.epochs, 
            batch_size=args.batch_size
        )
        healthy_errors, abnormal_errors = evaluate_tensorflow_model(
            model, test_healthy, test_abnormal
        )
    
    # Store training losses globally for visualization
    global training_losses
    training_losses = losses
    
    # Calculate metrics and create visualizations
    results = calculate_metrics_and_visualize(healthy_errors, abnormal_errors)
    
    print("\nAnalysis complete! Check the generated plots for detailed results.")


if __name__ == "__main__":
    main() 