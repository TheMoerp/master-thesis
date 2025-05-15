#!/usr/bin/env python3
"""
Simple test script for Anatomix model.
"""

import os
import sys
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def run_simple_test():
    """Run a simple test with the Anatomix model"""
    print("Running simple Anatomix test...")
    
    # Set up paths
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
        # Try alternative path
        sys.path.insert(0, os.path.join(anatomix_path, 'anatomix'))
        from model.network import Unet
    
    # Create model
    print("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = Unet(
        dimension=3,
        input_nc=1,
        output_nc=16,
        num_downs=4,
        ngf=16,
    ).to(device)
    print("Model created.")
    
    # Set model to eval mode
    model.eval()
    
    # Create a small dummy input (64x64x64)
    print("Creating dummy input...")
    dummy_input = torch.randn(1, 1, 64, 64, 64).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")
    
    # Try to load one BraTS image and process it
    print("\nTrying to process a real BraTS image...")
    # Find BraTS images
    brats_dir = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
    patient_dirs = sorted(glob(os.path.join(brats_dir, 'BraTS-GLI-*')))
    
    if not patient_dirs:
        print("No BraTS directories found.")
        return
    
    # Get first patient directory
    patient_dir = patient_dirs[0]
    print(f"Using patient directory: {patient_dir}")
    
    # Get T1 image
    t1_files = glob(os.path.join(patient_dir, '*-t1n.nii.gz'))
    if not t1_files:
        print("No T1 files found.")
        return
    
    t1_file = t1_files[0]
    print(f"Using T1 file: {t1_file}")
    
    # Load image
    print("Loading image...")
    img_nib = nib.load(t1_file)
    img = img_nib.get_fdata()
    print(f"Original image shape: {img.shape}")
    
    # Resize to 64x64x64 for testing
    from scipy.ndimage import zoom
    
    target_shape = (64, 64, 64)
    zoom_factors = [target_shape[i] / img.shape[i] for i in range(3)]
    print(f"Resizing image to {target_shape}...")
    img_resized = zoom(img, zoom_factors, order=1)
    
    # Normalize
    img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())
    
    # Convert to tensor
    input_tensor = torch.from_numpy(img_resized[np.newaxis, np.newaxis, ...]).float().to(device)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Run forward pass
    print("Running forward pass on real image...")
    with torch.no_grad():
        features = model(input_tensor)
    print(f"Features shape: {features.shape}")
    print("Forward pass successful on real image!")
    
    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    run_simple_test() 