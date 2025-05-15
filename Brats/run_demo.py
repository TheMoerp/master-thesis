#!/usr/bin/env python3
"""
Demo script to run BraTS feature extraction and anomaly detection.
"""

import os
import subprocess
import argparse

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Install requirements if requested
    if args.install_requirements:
        print("Installing requirements...")
        subprocess.run("pip install -r requirements.txt", shell=True)
    
    # Run the feature extraction and anomaly detection script
    cmd = [
        "python", "brats_feature_extraction.py",
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--knn_neighbors", str(args.knn_neighbors)
    ]
    
    if args.visualize:
        cmd.append("--visualize")
    
    if args.train_ratio is not None:
        cmd.extend(["--train_ratio", str(args.train_ratio)])
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BraTS Feature Extraction Demo')
    parser.add_argument('--data_path', type=str, 
                        default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData',
                        help='Path to BraTS dataset')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--knn_neighbors', type=int, default=5,
                        help='Number of neighbors for KNN anomaly detection')
    parser.add_argument('--train_ratio', type=float, default=None,
                        help='Ratio of data to use for training (0.8 by default)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--install_requirements', action='store_true',
                        help='Install requirements before running')
    
    args = parser.parse_args()
    main(args) 