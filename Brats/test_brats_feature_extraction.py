#!/usr/bin/env python3
"""
Test script for BraTS feature extraction.
"""

import os
import sys
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
from brats_feature_extraction import (
    setup_anatomix, minmax, get_brats_paths, 
    extract_all_modality_features, compute_feature_vectors,
    visualize_features
)

# Suppress warnings
warnings.filterwarnings("ignore")

def test_brats_feature_extraction():
    """Test BraTS feature extraction on a sample patient"""
    print("Testing BraTS feature extraction...")
    
    # Setup Anatomix
    print("\n[1/5] Setting up Anatomix model...")
    try:
        model, device = setup_anatomix()
        print("✅ Successfully set up Anatomix model")
    except Exception as e:
        print(f"❌ Error setting up Anatomix model: {e}")
        return False
    
    # Get paths to BraTS dataset
    print("\n[2/5] Finding BraTS dataset paths...")
    data_path = "datasets/BraTS2025-GLI-PRE-Challenge-TrainingData"
    try:
        samples = get_brats_paths(data_path)
        print(f"✅ Successfully found {len(samples)} patients")
        
        if len(samples) == 0:
            print("❌ No samples found! Check if your BraTS dataset is correctly located")
            return False
        
        # Print the first patient directory and modalities
        patient_id = list(samples.keys())[0]
        print(f"   First patient: {patient_id}")
        for modality, path in samples[patient_id].items():
            print(f"   - {modality}: {os.path.basename(path)}")
    except Exception as e:
        print(f"❌ Error getting BraTS paths: {e}")
        return False
    
    # Extract features for one patient
    print("\n[3/5] Extracting features for one patient...")
    try:
        # Get first patient
        patient_id = list(samples.keys())[0]
        features, seg = extract_all_modality_features(model, device, samples[patient_id])
        print(f"✅ Successfully extracted features for patient {patient_id}")
        
        # Print feature shapes
        for modality, feat in features.items():
            print(f"   - {modality} feature shape: {feat.shape}")
        
        if seg is not None:
            print(f"   - Segmentation shape: {seg.shape}")
        else:
            print("   - No segmentation found for this patient")
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return False
    
    # Compute feature vectors
    print("\n[4/5] Computing feature vectors...")
    try:
        feature_vectors = compute_feature_vectors(features)
        print("✅ Successfully computed feature vectors")
        
        # Print feature vector shapes
        for modality, feat_vec in feature_vectors.items():
            print(f"   - {modality} feature vector shape: {feat_vec.shape}")
    except Exception as e:
        print(f"❌ Error computing feature vectors: {e}")
        return False
    
    # Visualize features (optional)
    print("\n[5/5] Visualizing features...")
    try:
        visualize_features(features, slice_idx=80, num_channels=4)
        print("✅ Successfully visualized features")
    except Exception as e:
        print(f"❌ Error visualizing features: {e}")
        # This is not a critical error, so continue
    
    print("\n========================================")
    print("BraTS feature extraction test completed successfully!")
    print("========================================")
    return True

if __name__ == "__main__":
    test_brats_feature_extraction() 