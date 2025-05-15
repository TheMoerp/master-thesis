import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

def load_anatomix_model():
    """
    Load the pretrained Anatomix model
    """
    try:
        import torch.hub
        # Load the model from the official GitHub repository
        model = torch.hub.load('neeldey/anatomix', 'anatomix', pretrained=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    except Exception as e:
        print(f"Error loading Anatomix model: {e}")
        print("Please make sure to install Anatomix dependencies: pip install torch monai numpy nibabel tqdm")
        sys.exit(1)

def preprocess_volume(volume):
    """
    Preprocess the volume for Anatomix feature extraction
    """
    # Normalize volume to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # Convert to tensor and add batch and channel dimensions
    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    
    if torch.cuda.is_available():
        volume_tensor = volume_tensor.cuda()
        
    return volume_tensor

def extract_features(model, volume_tensor):
    """
    Extract features from the volume using Anatomix
    """
    with torch.no_grad():
        features = model.extract_features(volume_tensor)
    
    return features

def process_brats_dataset(input_dir, output_dir):
    """
    Process all BraTS volumes in input_dir and save the extracted features to output_dir
    """
    model = load_anatomix_model()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subject directories in the BraTS dataset
    subjects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_dir = os.path.join(input_dir, subject)
        
        # Find the T1-weighted, T1-contrast, T2-weighted, and FLAIR volumes
        modalities = {
            't1': None,
            't1ce': None,
            't2': None,
            'flair': None
        }
        
        for file in os.listdir(subject_dir):
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                for modality in modalities:
                    if modality in file.lower():
                        modalities[modality] = os.path.join(subject_dir, file)
        
        # Process each modality
        for modality, filepath in modalities.items():
            if filepath is None:
                print(f"Warning: {modality} not found for subject {subject}")
                continue
            
            # Load volume
            nifti_img = nib.load(filepath)
            volume = nifti_img.get_fdata()
            
            # Preprocess volume
            volume_tensor = preprocess_volume(volume)
            
            # Extract features
            features = extract_features(model, volume_tensor)
            
            # Save extracted features
            output_file = os.path.join(output_dir, f"{subject}_{modality}_features.npz")
            
            # Convert features to numpy arrays and save
            features_np = {f"level_{i}": feat.cpu().numpy() for i, feat in enumerate(features)}
            np.savez_compressed(output_file, **features_np)
            
            print(f"Saved features for {subject} {modality} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Anatomix features from BraTS dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to BraTS dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save extracted features")
    args = parser.parse_args()
    
    process_brats_dataset(args.input_dir, args.output_dir) 