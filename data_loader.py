import os
import csv
import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRangePercentilesd, ScaleIntensityRanged, CropForegroundd, ResizeWithPadOrCropd,
    RandFlipd, RandAffined, RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd,
    RandSpatialCropd, CenterSpatialCropd, SpatialPadd, EnsureTyped, Resized, RandGibbsNoised
)
from monai.data import DataLoader, PersistentDataset
from monai.utils import set_determinism

import glob

def parse_dataset_csv(csv_path, images_dir):
    data_dicts = []
    if not os.path.exists(csv_path):
        print(f"Warning: CSV {csv_path} not found.")
        return data_dicts
        
    # Pre-map all NIfTI files in the directory for fast lookup
    print(f"Indexing images in {images_dir}... (this may take a minute)")
    all_files = glob.glob(os.path.join(images_dir, "**/*.nii*"), recursive=True)
    file_map = {os.path.basename(f): f for f in all_files}
    print(f"Indexed {len(file_map)} unique volumes.")

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            vol_name = row[0]
            # Try to find the file in our map
            img_path = file_map.get(vol_name)
            
            # Try without .gz if needed
            if not img_path and vol_name.endswith('.gz'):
                img_path = file_map.get(vol_name[:-3])

            if img_path:
                labels = np.array([float(x) for x in row[1:]], dtype=np.float32)
                data_dicts.append({
                    "image": img_path,
                    "label": labels
                })
    
    print(f"Matched {len(data_dicts)} images from CSV.")
    return data_dicts

def get_transforms(mode="train"):
    # STAGE 3 FRONTIER UPGRADES:
    # 1. Spacing is physical 1.5mm x 1.5mm x 1.5mm (up from 2.0mm) to capture fine nodule textures and rib fractures.
    # 2. Base crop shape is 160x160x112 (up from 128x128x96) to cover a large anatomical context at high resolution.
    base_transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear")),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=5.0, upper=95.0,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(160, 160, 112)),
    ]
    
    if mode == "train":
        train_transforms = [
            RandFlipd(keys=["image"], prob=0.0), # Caching marker
            # Crop at high-resolution 160x160x112 without warping anatomy
            RandSpatialCropd(keys=["image"], roi_size=(160, 160, 112), random_size=False),
            # Flips along all 3 axes independently
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            # Rotations and Scaling
            RandAffined(keys=["image"], prob=0.3, rotate_range=(0.15, 0.15, 0.15), scale_range=(0.1, 0.1, 0.1)),
            # Advanced Medical Noise Augmentations
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
            RandGibbsNoised(keys=["image"], prob=0.15, alpha=(0.1, 0.4)),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3)
        ]
        transforms = base_transforms + train_transforms
    else:
        # Validation uses a fixed center crop at 160x160x112 for consistent validation metrics
        transforms = base_transforms + [
            RandFlipd(keys=["image"], prob=0.0),
            CenterSpatialCropd(keys=["image"], roi_size=(160, 160, 112))
        ]

    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)

class NakedDataset(PersistentDataset):
    def __init__(self, data, transform, cache_dir):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
        
    def __getitem__(self, index):
        data = super().__getitem__(index)
        image = data["image"]
        label = data["label"]
        
        # Convert to raw PyTorch tensors to completely strip all MONAI MetaTensor wrappers
        if hasattr(image, "numpy"):
            image_np = image.numpy()
        else:
            image_np = np.array(image)
            
        if hasattr(label, "numpy"):
            label_np = label.numpy()
        else:
            label_np = np.array(label)
            
        return {
            "image": torch.tensor(image_np),
            "label": torch.tensor(label_np)
        }

def compute_sample_weights(data_dicts):
    """
    Compute per-sample weights inversely proportional to class frequency.
    Samples with rare pathologies (Hernia, Pneumothorax, Fracture etc.) get
    upweighted so the DataLoader sees them proportionally more often.
    """
    # Stack all labels into a matrix: (N, 18)
    label_matrix = np.stack([d["label"] for d in data_dicts], axis=0)
    N, C = label_matrix.shape

    # Class frequency: fraction of samples positive for each class
    class_freq = label_matrix.mean(axis=0)  # shape (C,)
    class_freq = np.clip(class_freq, 1e-6, 1.0)  # avoid div-by-zero

    # Inverse frequency weight per class
    class_weights = 1.0 / class_freq  # rare classes get higher weight

    # Per-sample weight = sum of weights for all positive labels in that sample
    # A sample with only common pathologies gets a low weight; rare pathology samples get high weight
    sample_weights = []
    for labels in label_matrix:
        positive_mask = labels > 0.5
        if positive_mask.any():
            w = class_weights[positive_mask].mean()
        else:
            w = 1.0  # healthy / no findings — normal weight
        sample_weights.append(w)

    return torch.DoubleTensor(sample_weights)


def get_dataloaders(train_csv, valid_csv, train_images_dir, valid_images_dir, batch_size=8, seed=42):
    set_determinism(seed=seed)
    
    train_data = parse_dataset_csv(train_csv, train_images_dir) if train_csv else []
    val_data = parse_dataset_csv(valid_csv, valid_images_dir) if valid_csv else []
    
    # Cache directory for high-res v4 persistent cache
    cache_dir = "/scratch/25208443/monai_cache_v4"
    os.makedirs(cache_dir, exist_ok=True)
        
    train_loader = None
    if train_data:
        train_dataset = NakedDataset(data=train_data, transform=get_transforms("train"), cache_dir=cache_dir)

        # WeightedRandomSampler: oversample rare-pathology scans to combat class imbalance
        # This replaces shuffle=True — sampler and shuffle are mutually exclusive
        sample_weights = compute_sample_weights(train_data)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"=> WeightedRandomSampler enabled: {len(sample_weights)} samples, "
              f"weight range [{sample_weights.min():.2f}, {sample_weights.max():.2f}]", flush=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=sampler,          # replaces shuffle=True
            num_workers=8, pin_memory=True,
            persistent_workers=True,  # keep workers alive between epochs (saves ~5-10s/epoch respawn)
            prefetch_factor=4         # pre-load 4 batches per worker (default=2) to keep GPU busy
        )
        
    val_loader = None
    if val_data:
        val_dataset = NakedDataset(data=val_data, transform=get_transforms("val"), cache_dir=cache_dir)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=8, pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)
        
    return train_loader, val_loader, train_data
