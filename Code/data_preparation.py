import os
import glob
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, 
    LoadImaged, 
    ScaleIntensityd, 
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    EnsureTyped,
    ToTensord,
    EnsureChannelFirstD
)
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
import SimpleITK as sitk
import pandas as pd

def prepare_ribfrac_dataset(data_dir="datasets", output_dir=None, cache_rate=0.0, batch_size=2, num_workers=2):
    """
    Prepare the RibFrac dataset for training using MONAI transforms.
    
    Args:
        data_dir: Directory containing the RibFrac dataset
        output_dir: Optional directory to save processed data
        cache_rate: Cache rate for MONAI CacheDataset
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation and testing
    """
    set_determinism(seed=42)
    
    # Updated paths for the new structure
    images_dir = os.path.join(data_dir, "Images")
    labels_dir = os.path.join(data_dir, "Labels")
    
    # Get all image files with the correct pattern - using os.path.join for Windows compatibility
    all_images = sorted(glob.glob(os.path.join(images_dir, "RibFrac*-image.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(labels_dir, "RibFrac*-label.nii.gz")))
    
    print(f"Found {len(all_images)} images and {len(all_labels)} labels")
    
    if len(all_images) == 0 or len(all_labels) == 0:
        raise ValueError(f"No images or labels found in {images_dir} and {labels_dir}")
    
    # Create dictionaries for MONAI data loading
    data_dicts = []
    
    # For each image-label pair
    for img_path, lbl_path in zip(all_images, all_labels):
        # Verify that the image and label IDs match
        img_id = os.path.basename(img_path).split('-')[0]
        lbl_id = os.path.basename(lbl_path).split('-')[0]
        
        if img_id == lbl_id:
            # Extract case ID number
            # RibFrac123 -> 123
            case_id = int(img_id.replace("RibFrac", ""))
            
            # Artificially designate half of the dataset as "normal" and half as "fracture"
            # For demonstration, we'll consider first half as normal, second half as fracture
            # This is artificial but allows us to test the anomaly detection approach
            has_fracture = case_id > 150  # Arbitrary split
            
            data_dict = {
                "image": img_path,
                "label": lbl_path,
                "has_fracture": has_fracture
            }
            data_dicts.append(data_dict)
    
    print(f"Successfully paired {len(data_dicts)} image-label pairs")
    
    # Count how many fracture and normal cases we have
    fracture_count = sum(1 for d in data_dicts if d["has_fracture"])
    normal_count = sum(1 for d in data_dicts if not d["has_fracture"])
    print(f"Total count: {len(data_dicts)} images - {fracture_count} fracture cases, {normal_count} normal cases")
    
    # Split into training, validation and test sets (70/15/15)
    # Ensure we have a balanced test set with both normal and fracture cases
    fracture_dicts = [d for d in data_dicts if d["has_fracture"]]
    normal_dicts = [d for d in data_dicts if not d["has_fracture"]]
    
    # Shuffle both lists
    np.random.shuffle(fracture_dicts)
    np.random.shuffle(normal_dicts)
    
    # Calculate number of samples for each set while ensuring balance in test set
    n_fracture = len(fracture_dicts)
    n_normal = len(normal_dicts)
    
    # For test set, take equal numbers of normal and fracture cases (up to 15 of each)
    n_test_per_class = min(15, min(n_fracture, n_normal) // 3)
    
    # For validation, take up to 15 of each as well
    n_val_per_class = min(15, min(n_fracture, n_normal) // 3)
    
    # Rest goes to training
    test_fracture = fracture_dicts[:n_test_per_class]
    test_normal = normal_dicts[:n_test_per_class]
    
    val_fracture = fracture_dicts[n_test_per_class:n_test_per_class+n_val_per_class]
    val_normal = normal_dicts[n_test_per_class:n_test_per_class+n_val_per_class]
    
    train_fracture = fracture_dicts[n_test_per_class+n_val_per_class:]
    train_normal = normal_dicts[n_test_per_class+n_val_per_class:]
    
    # Combine and shuffle
    train_files = train_fracture + train_normal
    val_files = val_fracture + val_normal
    test_files = test_fracture + test_normal
    
    np.random.shuffle(train_files)
    np.random.shuffle(val_files)
    np.random.shuffle(test_files)
    
    # Use all available images (removed the limit of 60 samples)
    
    print(f"Using {len(train_files)} training samples ({len(train_fracture)} fracture, {len(train_normal)} normal)")
    print(f"Using {len(val_files)} validation samples ({len(val_fracture)} fracture, {len(val_normal)} normal)")
    print(f"Using {len(test_files)} testing samples ({len(test_fracture)} fracture, {len(test_normal)} normal)")
    
    # Define the data transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),  # Add channel dimension
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest")
        ),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=20
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),  # 3D patches
            pos=1,
            neg=1,
            num_samples=1,  # Changed from 4 to 1 for quicker testing
            image_key="image",
            image_threshold=0,
        ),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),  # Add channel dimension
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest")
        ),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=20
        ),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])
    
    # Create datasets with reduced memory usage
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers
    )
    
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers
    )
    
    test_ds = CacheDataset(
        data=test_files,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset loading with minimal memory usage
    try:
        train_loader, val_loader, test_loader = prepare_ribfrac_dataset(
            batch_size=1,  # Minimal batch size
            cache_rate=0.0,  # No caching
            num_workers=2  # Increased number of workers
        )
        
        print("\nDataset loading successful!")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of testing batches: {len(test_loader)}")
        
        # Test loading one batch
        print("\nTesting batch loading...")
        for batch in train_loader:
            image = batch["image"]
            label = batch["label"]
            print(f"Image shape: {image.shape}")
            print(f"Label shape: {label.shape}")
            break
            
    except Exception as e:
        print(f"\nError loading dataset: {str(e)}") 