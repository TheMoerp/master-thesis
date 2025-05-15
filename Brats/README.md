# BraTS Feature Extraction and Anomaly Detection

This repository contains code for extracting features from the BraTS dataset using Anatomix and performing anomaly detection using a k-nearest neighbors approach.

## Overview

The code performs the following steps:
1. Sets up Anatomix and loads the pretrained model
2. Extracts features from BraTS dataset MRI modalities (t1n, t1c, t2w, t2f)
3. Performs anomaly detection using k-nearest neighbors on the extracted features
4. Evaluates the results using ROC AUC (if ground truth segmentations are available)
5. Visualizes and saves the results

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

The code will automatically clone and install Anatomix from GitHub when first run.

## Usage

```bash
python brats_feature_extraction.py [OPTIONS]
```

### Options

- `--data_path`: Path to BraTS dataset (default: 'datasets/BraTS2025-GLI-PRE-Challenge-TrainingData')
- `--output_dir`: Output directory for results (default: 'results')
- `--knn_neighbors`: Number of neighbors for KNN anomaly detection (default: 5)
- `--train_ratio`: Ratio of data to use for training; if 0, uses leave-one-out approach (default: 0.8)
- `--visualize`: Enable visualization of results

### Example

```bash
python brats_feature_extraction.py --data_path datasets/BraTS2025-GLI-PRE-Challenge-TrainingData --output_dir results --visualize
```

## Dataset Structure

The code expects the BraTS dataset to be organized as follows:

```
data_path/
  ├── BraTS-GLI-XXXXX-XXX/
  │   ├── BraTS-GLI-XXXXX-XXX-t1n.nii.gz
  │   ├── BraTS-GLI-XXXXX-XXX-t1c.nii.gz
  │   ├── BraTS-GLI-XXXXX-XXX-t2w.nii.gz
  │   ├── BraTS-GLI-XXXXX-XXX-t2f.nii.gz
  │   └── BraTS-GLI-XXXXX-XXX-seg.nii.gz (optional)
  └── ...
```

## Output

The script generates:
1. Anomaly score maps for each patient (saved as NIfTI files)
2. Visualization plots for selected patients (if `--visualize` is enabled)
3. Evaluation metrics (AUC) if ground truth segmentations are available 