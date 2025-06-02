#!/usr/bin/env python3
"""
Test script to verify Anatomix model loading.
"""

import os
import sys
import torch

def main():
    """Main function to test model loading"""
    # Get anatomix path
    anatomix_path = os.path.abspath('./anatomix')
    
    # Add anatomix to Python path
    if anatomix_path not in sys.path:
        sys.path.insert(0, anatomix_path)
    
    # Add anatomix module to path
    sys.path.insert(0, os.path.join(anatomix_path, 'anatomix'))
    
    try:
        # Try to import the model
        print("Trying to import Unet from model.network...")
        from model.network import Unet
        print("Successfully imported Unet!")
        
        # Create device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model
        print("Creating model...")
        model = Unet(
            dimension=3,
            input_nc=1,
            output_nc=16,
            num_downs=4,
            ngf=16,
        ).to(device)
        print("Successfully created model!")
        
        # Load weights
        weights_path = os.path.join(anatomix_path, "model-weights", "anatomix.pth")
        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print("Successfully loaded weights!")
        else:
            print(f"Model weights not found at {weights_path}")
        
        print("Model test completed successfully!")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 