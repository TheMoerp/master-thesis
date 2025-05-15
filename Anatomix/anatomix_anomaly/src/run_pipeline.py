#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import numpy as np
        import nibabel
        import matplotlib
        import sklearn
        from tqdm import tqdm
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install the required packages using: pip install -r requirements.txt")
        sys.exit(1)

def run_command(cmd):
    """Run a command and print its output"""
    print(f"\nRunning: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        sys.exit(process.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run Anatomix feature extraction and anomaly detection pipeline")
    parser.add_argument("--brats_dir", type=str, required=True, help="Path to BraTS dataset directory")
    parser.add_argument("--output_dir", type=str, default="anatomix_anomaly/results", help="Base directory for outputs")
    parser.add_argument("--modality", type=str, default="t1ce", help="Modality to use for anomaly detection")
    parser.add_argument("--feature_level", type=int, default=0, help="Feature level to use (0-3)")
    parser.add_argument("--n_clusters", type=int, default=2, help="Number of clusters for K-means")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip feature extraction step")
    
    args = parser.parse_args()
    
    # Create output directories
    features_dir = os.path.join(args.output_dir, "features")
    results_dir = os.path.join(args.output_dir, "anomaly_results")
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check requirements
    check_requirements()
    
    # Get the directory of the current script
    script_dir = Path(__file__).parent.absolute()
    
    # Step 1: Extract features using Anatomix
    if not args.skip_extraction:
        extract_script = os.path.join(script_dir, "extract_features.py")
        extract_cmd = [
            sys.executable, 
            extract_script, 
            "--input_dir", args.brats_dir, 
            "--output_dir", features_dir
        ]
        run_command(extract_cmd)
    else:
        print("Skipping feature extraction step.")
    
    # Step 2: Run anomaly detection with K-means
    anomaly_script = os.path.join(script_dir, "anomaly_detection.py")
    anomaly_cmd = [
        sys.executable, 
        anomaly_script, 
        "--features_dir", features_dir, 
        "--output_dir", results_dir, 
        "--modality", args.modality, 
        "--feature_level", str(args.feature_level), 
        "--n_clusters", str(args.n_clusters), 
        "--n_components", str(args.n_components)
    ]
    run_command(anomaly_cmd)
    
    print(f"\nPipeline completed successfully!")
    print(f"Features saved to: {features_dir}")
    print(f"Anomaly detection results saved to: {results_dir}")

if __name__ == "__main__":
    main() 