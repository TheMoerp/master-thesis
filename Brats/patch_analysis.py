#!/usr/bin/env python3
"""
Script to analyze patch distribution in BraTS dataset.
This script divides brain scans into patches and counts how many are normal vs. tumor patches.
"""

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from collections import defaultdict

def create_patches(image, segmentation, patch_size=(32, 32, 32), overlap=0.5, min_tumor_ratio=0.05):
    """
    Create patches from 3D image with segmentation.
    Returns patches and a flag for each patch indicating if it contains tumor.
    """
    patches = []
    has_tumor = []
    tumor_ratios = []
    patch_locations = []
    
    # Calculate stride (amount to move when extracting patches)
    stride = [int(p * (1 - overlap)) for p in patch_size]
    
    # Extract patches
    for x in range(0, image.shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, image.shape[1] - patch_size[1] + 1, stride[1]):
            for z in range(0, image.shape[2] - patch_size[2] + 1, stride[2]):
                # Extract the patch
                patch = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                
                # Check if patch has the expected size (might be smaller at the edges)
                if patch.shape != tuple(patch_size):
                    continue
                
                # Add the patch
                patches.append(patch)
                patch_locations.append((x, y, z))
                
                # Check if patch contains tumor
                seg_patch = segmentation[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                tumor_ratio = np.count_nonzero(seg_patch) / seg_patch.size
                tumor_ratios.append(tumor_ratio)
                has_tumor.append(tumor_ratio >= min_tumor_ratio)
    
    return patches, has_tumor, tumor_ratios, patch_locations

def get_patient_data(data_path, max_patients=None):
    """Get patient data up to max_patients"""
    # Get list of patient directories
    patient_dirs = sorted(glob(os.path.join(data_path, 'BraTS-GLI-*')))
    
    if not patient_dirs:
        raise FileNotFoundError(f"No BraTS patient directories found in {data_path}")
    
    # Limit to max_patients if specified
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]
    print(f"Using {len(patient_dirs)} patient directories")
    
    # Get data for each patient
    patient_data = {}
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        
        # Get modality files
        t1n_file = glob(os.path.join(patient_dir, '*-t1n.nii.gz'))[0]
        
        # Get segmentation
        seg_file = glob(os.path.join(patient_dir, '*-seg.nii.gz'))[0]
        
        patient_data[patient_id] = {
            't1n': t1n_file,
            'seg': seg_file
        }
    
    return patient_data

def analyze_patches(patient_data, patch_size=(32, 32, 32), overlap=0.5, min_tumor_ratio=0.05):
    """Analyze patch distribution for each patient"""
    results = {}
    tumor_ratio_distribution = []
    
    for patient_id, paths in tqdm(patient_data.items()):
        # Load data
        t1_img = nib.load(paths['t1n']).get_fdata()
        seg = nib.load(paths['seg']).get_fdata()
        
        # Create patches
        _, has_tumor, tumor_ratios, _ = create_patches(
            t1_img, seg, patch_size=patch_size, overlap=overlap, min_tumor_ratio=min_tumor_ratio
        )
        
        # Count normal and tumor patches
        normal_patches = sum(1 for x in has_tumor if not x)
        tumor_patches = sum(has_tumor)
        total_patches = len(has_tumor)
        
        # Save results
        results[patient_id] = {
            'normal_patches': normal_patches,
            'tumor_patches': tumor_patches,
            'total_patches': total_patches,
            'normal_ratio': normal_patches / total_patches if total_patches > 0 else 0,
            'tumor_ratio': tumor_patches / total_patches if total_patches > 0 else 0
        }
        
        # Add tumor ratios to distribution
        tumor_ratio_distribution.extend(tumor_ratios)
    
    return results, tumor_ratio_distribution, min_tumor_ratio

def visualize_results(results, tumor_ratio_distribution, output_dir, min_tumor_ratio=0.05):
    """Visualize patch distribution analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame.from_dict(results, orient='index')
    df['patient_id'] = df.index
    
    # Plot normal vs tumor patches for each patient
    plt.figure(figsize=(20, 10))
    df_plot = df.sort_values('normal_ratio', ascending=False)
    
    # Plot stacked bar chart
    ax = df_plot.plot(
        kind='bar', 
        x='patient_id', 
        y=['normal_patches', 'tumor_patches'], 
        stacked=True,
        figsize=(20, 10),
        color=['green', 'red']
    )
    
    plt.title('Normal vs Tumor Patches per Patient', fontsize=16)
    plt.xlabel('Patient ID', fontsize=14)
    plt.ylabel('Number of Patches', fontsize=14)
    plt.xticks(rotation=90)
    plt.legend(['Normal Patches', 'Tumor Patches'], fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patches_per_patient.png'))
    
    # Plot ratio of normal patches
    plt.figure(figsize=(12, 6))
    df_ratio = df_plot.copy()
    df_ratio = df_ratio.sort_values('normal_ratio', ascending=False)
    sns.barplot(x='patient_id', y='normal_ratio', data=df_ratio)
    plt.title('Ratio of Normal Patches per Patient', fontsize=16)
    plt.xlabel('Patient ID', fontsize=14)
    plt.ylabel('Ratio of Normal Patches', fontsize=14)
    plt.xticks(rotation=90)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normal_ratio_per_patient.png'))
    
    # Plot histogram of tumor ratios in patches
    plt.figure(figsize=(12, 6))
    plt.hist(tumor_ratio_distribution, bins=50)
    plt.title('Distribution of Tumor Ratios in Patches', fontsize=16)
    plt.xlabel('Tumor Ratio', fontsize=14)
    plt.ylabel('Number of Patches', fontsize=14)
    plt.axvline(x=min_tumor_ratio, color='r', linestyle='--', label=f'Threshold ({min_tumor_ratio})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tumor_ratio_distribution.png'))
    
    # Calculate overall statistics
    total_normal = df['normal_patches'].sum()
    total_tumor = df['tumor_patches'].sum()
    total_patches = total_normal + total_tumor
    
    # Create summary plot
    plt.figure(figsize=(8, 8))
    plt.pie([total_normal, total_tumor], labels=['Normal', 'Tumor'], autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Overall Distribution of Patches', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_distribution.png'))
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Total patients analyzed: {len(results)}\n")
        f.write(f"Total patches: {total_patches}\n")
        f.write(f"Normal patches: {total_normal} ({total_normal/total_patches*100:.1f}%)\n")
        f.write(f"Tumor patches: {total_tumor} ({total_tumor/total_patches*100:.1f}%)\n")
        f.write(f"Ratio of normal to tumor patches: {total_normal/total_tumor:.2f}\n")
        f.write("\nPatients with highest percentage of normal patches:\n")
        
        for patient_id, row in df_plot.head(5).iterrows():
            f.write(f"  {patient_id}: {row['normal_ratio']*100:.1f}% normal patches\n")
        
        f.write("\nPatients with lowest percentage of normal patches:\n")
        for patient_id, row in df_plot.tail(5).iterrows():
            f.write(f"  {patient_id}: {row['normal_ratio']*100:.1f}% normal patches\n")
    
    # Print summary to console
    total_normal = df['normal_patches'].sum()
    total_tumor = df['tumor_patches'].sum()
    print(f"\nSummary:")
    print(f"Total patches: {total_normal + total_tumor}")
    print(f"Normal patches: {total_normal} ({total_normal/(total_normal + total_tumor)*100:.1f}%)")
    print(f"Tumor patches: {total_tumor} ({total_tumor/(total_normal + total_tumor)*100:.1f}%)")
    
    # Create tumor size distribution analysis
    patches_by_tumor_ratio = defaultdict(int)
    for ratio in tumor_ratio_distribution:
        bucket = round(ratio * 20) / 20  # Round to nearest 0.05
        patches_by_tumor_ratio[bucket] += 1
    
    # Plot sorted distribution
    plt.figure(figsize=(15, 8))
    sorted_buckets = sorted(patches_by_tumor_ratio.items())
    buckets, counts = zip(*sorted_buckets)
    plt.bar([str(f"{b:.2f}") for b in buckets], counts)
    plt.title('Distribution of Patches by Tumor Ratio (Discretized)', fontsize=16)
    plt.xlabel('Tumor Ratio', fontsize=14)
    plt.ylabel('Number of Patches', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tumor_ratio_buckets.png'))
    
    return df

def visualize_patch_examples(patient_data, patch_size=(32, 32, 32), output_dir='patch_examples'):
    """Visualize examples of patches with different tumor ratios"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a single patient
    patient_id = list(patient_data.keys())[0]
    paths = patient_data[patient_id]
    
    # Load data
    t1_img = nib.load(paths['t1n']).get_fdata()
    seg = nib.load(paths['seg']).get_fdata()
    
    # Create patches
    patches, _, tumor_ratios, locations = create_patches(
        t1_img, seg, patch_size=patch_size, overlap=0.5, min_tumor_ratio=0
    )
    
    # Group patches by tumor ratio
    ratio_examples = {}
    ratio_brackets = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    
    # Find examples for each ratio bracket
    for i, ratio in enumerate(tumor_ratios):
        for j in range(len(ratio_brackets) - 1):
            min_ratio = ratio_brackets[j]
            max_ratio = ratio_brackets[j+1]
            bracket = f"{min_ratio:.2f}-{max_ratio:.2f}"
            
            if min_ratio <= ratio < max_ratio and bracket not in ratio_examples:
                ratio_examples[bracket] = {
                    'patch': patches[i],
                    'ratio': ratio,
                    'location': locations[i]
                }
    
    # Visualize example patches
    plt.figure(figsize=(15, 10))
    
    for i, (bracket, example) in enumerate(sorted(ratio_examples.items())):
        plt.subplot(2, 3, i+1)
        
        # Show middle slice of the patch
        patch = example['patch']
        middle_slice = patch.shape[0] // 2
        plt.imshow(patch[middle_slice, :, :], cmap='gray')
        
        # Get the corresponding segmentation
        x, y, z = example['location']
        seg_patch = seg[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
        
        # Overlay segmentation
        seg_mask = seg_patch[middle_slice, :, :] > 0
        plt.imshow(seg_mask, alpha=0.3, cmap='hot')
        
        plt.title(f"Tumor Ratio: {example['ratio']:.3f}\nBracket: {bracket}")
        plt.axis('off')
    
    plt.suptitle("Examples of Patches with Different Tumor Ratios", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'patch_examples.png'))
    
    # Show examples along 3 axes
    for bracket, example in sorted(ratio_examples.items()):
        plt.figure(figsize=(15, 5))
        
        # Get patch and segmentation
        patch = example['patch']
        x, y, z = example['location']
        seg_patch = seg[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
        
        # Show middle slices along each axis
        middle_x = patch.shape[0] // 2
        middle_y = patch.shape[1] // 2
        middle_z = patch.shape[2] // 2
        
        # X-axis (Sagittal)
        plt.subplot(1, 3, 1)
        plt.imshow(patch[middle_x, :, :], cmap='gray')
        plt.imshow(seg_patch[middle_x, :, :] > 0, alpha=0.3, cmap='hot')
        plt.title(f"Sagittal (X={middle_x})")
        plt.axis('off')
        
        # Y-axis (Coronal)
        plt.subplot(1, 3, 2)
        plt.imshow(patch[:, middle_y, :], cmap='gray')
        plt.imshow(seg_patch[:, middle_y, :] > 0, alpha=0.3, cmap='hot')
        plt.title(f"Coronal (Y={middle_y})")
        plt.axis('off')
        
        # Z-axis (Axial)
        plt.subplot(1, 3, 3)
        plt.imshow(patch[:, :, middle_z], cmap='gray')
        plt.imshow(seg_patch[:, :, middle_z] > 0, alpha=0.3, cmap='hot')
        plt.title(f"Axial (Z={middle_z})")
        plt.axis('off')
        
        plt.suptitle(f"Patch with Tumor Ratio: {example['ratio']:.3f} (Bracket: {bracket})", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'patch_views_{bracket.replace(".", "p")}.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze patch distribution in BraTS dataset')
    parser.add_argument('--data_path', type=str, default='datasets/BraTS2025-GLI-PRE-Challenge-TrainingData',
                        help='Path to BraTS dataset')
    parser.add_argument('--output_dir', type=str, default='patch_analysis_results',
                        help='Output directory for results')
    parser.add_argument('--max_patients', type=int, default=20,
                        help='Maximum number of patients to analyze')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Size of patches (cubic patches)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap between patches (0-1)')
    parser.add_argument('--min_tumor_ratio', type=float, default=0.05,
                        help='Minimum ratio of tumor voxels to consider a patch as containing tumor')
    args = parser.parse_args()
    
    # Get patient data
    print("Getting patient data...")
    patient_data = get_patient_data(args.data_path, max_patients=args.max_patients)
    
    # Use the same size for all dimensions
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    
    # Analyze patches
    print("\nAnalyzing patches...")
    results, tumor_ratio_distribution, min_tumor_ratio = analyze_patches(
        patient_data, 
        patch_size=patch_size,
        overlap=args.overlap,
        min_tumor_ratio=args.min_tumor_ratio
    )
    
    # Visualize results
    print("\nVisualizing results...")
    df = visualize_results(results, tumor_ratio_distribution, args.output_dir, min_tumor_ratio)
    
    # Save to CSV
    df.to_csv(os.path.join(args.output_dir, 'patch_analysis.csv'))
    
    # Visualize example patches
    print("\nVisualizing example patches...")
    visualize_patch_examples(patient_data, patch_size=patch_size, output_dir=args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 