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

def prepare_ribfrac_dataset(data_dir="datasets", output_dir=None, cache_rate=0.0, batch_size=2):
    """
    Prepare the RibFrac dataset for training using MONAI transforms.
    
    Args:
        data_dir: Directory containing the RibFrac dataset
        output_dir: Optional directory to save processed data
        cache_rate: Cache rate for MONAI CacheDataset
        batch_size: Batch size for DataLoader
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation and testing
    """
    set_determinism(seed=42)
    
    # Updated paths for the new structure
    images_dir = os.path.join(data_dir, "Images")
    labels_dir = os.path.join(data_dir, "Labels")
    
    # Get all image files with the correct pattern
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
            data_dict = {
                "image": img_path,
                "label": lbl_path
            }
            data_dicts.append(data_dict)
    
    print(f"Successfully paired {len(data_dicts)} image-label pairs")
    
    # Split into training, validation and test sets (70/15/15)
    np.random.shuffle(data_dicts)
    n_total = len(data_dicts)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_files = data_dicts[:n_train]
    val_files = data_dicts[n_train:n_train+n_val]
    test_files = data_dicts[n_train+n_val:]
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Testing files: {len(test_files)}")
    
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
            num_samples=4,
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
        cache_rate=cache_rate,  # Reduced cache rate to save memory
        num_workers=2  # Reduced number of workers
    )
    
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=2
    )
    
    test_ds = CacheDataset(
        data=test_files,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=2
    )
    
    # Create data loaders with reduced batch size and workers
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset loading with minimal memory usage
    try:
        train_loader, val_loader, test_loader = prepare_ribfrac_dataset(
            batch_size=1,  # Minimal batch size
            cache_rate=0.0  # No caching
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