#!/usr/bin/env python3
"""
Test script to verify that our Anatomix setup is working correctly.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_anatomix_test():
    """Setup and test the Anatomix model"""
    print("Testing Anatomix setup...")
    
    # Clone the Anatomix repository if it doesn't exist
    anatomix_path = os.path.abspath('./anatomix')
    if not os.path.exists(anatomix_path):
        print("Cloning Anatomix repository...")
        os.system('git clone https://github.com/neel-dey/anatomix.git')
        
        # Install Anatomix and its dependencies
        print("Installing Anatomix...")
        os.chdir('anatomix')
        os.system('pip install -e .')
        os.chdir('..')
    else:
        print(f"Using existing Anatomix installation at {anatomix_path}")
    
    # Add anatomix to Python path
    if anatomix_path not in sys.path:
        sys.path.insert(0, anatomix_path)
    
    # Try to import Anatomix modules
    try:
        print("Trying to import Anatomix modules...")
        from anatomix.model.network import Unet
        print("✅ Successfully imported Unet from anatomix.model.network")
    except ImportError as e:
        print(f"❌ Error importing from anatomix.model.network: {e}")
        print("Trying alternative import path...")
        sys.path.insert(0, os.path.join(anatomix_path, 'anatomix'))
        try:
            from model.network import Unet
            print("✅ Successfully imported Unet from alternative path")
        except ImportError as e2:
            print(f"❌ Failed to import from alternative path: {e2}")
            return False
    
    # Create a dummy input tensor
    print("\nCreating a dummy input tensor...")
    try:
        dummy_input = torch.randn(1, 1, 32, 32, 32)
        print(f"✅ Created dummy input tensor with shape {dummy_input.shape}")
    except Exception as e:
        print(f"❌ Error creating dummy tensor: {e}")
        return False
    
    # Try to instantiate the model
    print("\nInstantiating the Unet model...")
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            print("Using CUDA")
            device = torch.device('cuda')
        else:
            print("Using CPU")
            device = torch.device('cpu')
        
        model = Unet(
            dimension=3,
            input_nc=1,
            output_nc=16,
            num_downs=4,
            ngf=16,
        ).to(device)
        print("✅ Successfully instantiated the Unet model")
    except Exception as e:
        print(f"❌ Error instantiating model: {e}")
        return False
    
    # Try a forward pass
    print("\nTrying a forward pass with the model...")
    try:
        with torch.no_grad():
            dummy_input = dummy_input.to(device)
            output = model(dummy_input)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        return False
    
    # Check model weights directory
    weights_path = os.path.join(anatomix_path, "model-weights", "anatomix.pth")
    if os.path.exists(weights_path):
        print(f"\n✅ Model weights found at {weights_path}")
    else:
        print(f"\n❌ Model weights not found at {weights_path}")
        print("You need to manually obtain the weights file.")
    
    print("\n========================================")
    print("Anatomix setup test completed successfully!")
    print("========================================")
    return True

if __name__ == "__main__":
    setup_anatomix_test() 