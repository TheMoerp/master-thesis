# BraTS Anomaly Detection

This project implements an anomaly detection pipeline for 3D brain MRI scans from the BraTS dataset using the following approach:

1. Extract 3D patches from brain MRI scans
2. Label patches as normal/abnormal based on segmentation masks
3. Extract features using pre-trained Anatomix model
4. Perform anomaly detection using KNN with faiss (GPU-accelerated)

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- nibabel
- faiss-gpu
- scikit-learn
- matplotlib
- tqdm

## Dataset Structure

The code expects the BraTS dataset to be organized in the following structure:
```
datasets/
└── BraTS2025-GLI-PRE-Challenge-TrainingData/
    ├── BraTS-GLI-XXXXX-000/
    │   ├── BraTS-GLI-XXXXX-000-t1n.nii.gz  (T1 scan)
    │   ├── BraTS-GLI-XXXXX-000-t1c.nii.gz  (T1 contrast-enhanced scan)
    │   ├── BraTS-GLI-XXXXX-000-t2w.nii.gz  (T2 scan)
    │   ├── BraTS-GLI-XXXXX-000-t2f.nii.gz  (FLAIR scan)
    │   └── BraTS-GLI-XXXXX-000-seg.nii.gz  (Segmentation mask)
    ├── BraTS-GLI-YYYYY-000/
    └── ...
```

## Usage

To run the anomaly detection pipeline:

```bash
python brats_anomaly_detection.py --num_subjects 10 --visualize
```

### Command Line Arguments

- `--data_dir`: Path to the dataset directory (default: "datasets")
- `--num_subjects`: Number of subjects to process, -1 for all (default: -1)
- `--patch_size`: Size of 3D patches (default: 32)
- `--stride`: Stride for patch extraction (default: 16)
- `--normal_threshold`: Threshold for labeling patches as normal/abnormal (default: 0.01)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--k_neighbors`: Number of neighbors for KNN (default: 5)
- `--visualize`: Enable visualization of patches and results (default: False)

## Method

### 1. Patch Extraction and Labeling

The code extracts 3D cubic patches from the brain MRI scans. A patch is labeled as:
- **Normal** if it contains less than a specified threshold (default: 1%) of tumor voxels
- **Abnormal** if it contains more than the threshold of tumor voxels

### 2. Feature Extraction with Anatomix

The pre-trained Anatomix 3D U-Net model is used to extract features from each patch. These features capture anatomical representation.

### 3. Anomaly Detection with KNN

A K-Nearest Neighbors (KNN) model is trained using faiss for GPU acceleration. For each test patch:
- The K nearest neighbors from the training set are found
- The anomaly score is computed as the ratio of abnormal neighbors
- The patch is classified as abnormal if the majority of its neighbors are abnormal

### 4. Evaluation

The model is evaluated using:
- ROC AUC score
- Average Precision score
- Accuracy, Precision, Recall, and F1 score

## Visualizations

When the `--visualize` flag is set, the code generates:
1. Patch visualization (patch_visualization.png)
2. Feature embeddings visualization (feature_embeddings.png)
3. Anomaly detection results (anomaly_detection_results.png) 