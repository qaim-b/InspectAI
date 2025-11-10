"""
Dataset and DataLoader utilities for InspectAI
Handles industrial defect image datasets
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import random


class DefectDataset(Dataset):
    """
    Dataset class for defect detection
    
    Expected directory structure:
    data/
        train/
            defect/
                img1.jpg
                img2.jpg
            good/
                img1.jpg
                img2.jpg
        val/
            defect/
            good/
        test/
            defect/
            good/
    """
    
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir: Root directory of dataset
            transform: Albumentations transforms
            split: 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.split = split
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        
        # Class 0: good (no defect), Class 1: defect
        for class_name, class_idx in [('good', 0), ('defect', 1)]:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.images.append(str(img_path))
                        self.labels.append(class_idx)
        
        print(f"{split.upper()} Dataset: {len(self.images)} images")
        print(f"  Good: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Defect: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def get_transforms(img_size=224, augment=True):
    """
    Get augmentation transforms
    
    Args:
        img_size: Target image size
        augment: Whether to apply augmentations (for training)
    
    Returns:
        Albumentations compose transform
    """
    if augment:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def create_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    """
    Create train, val, test dataloaders
    
    Args:
        data_dir: Root directory containing train/val/test splits
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers for data loading
    
    Returns:
        Dictionary with train, val, test dataloaders
    """
    # Transforms
    train_transform = get_transforms(img_size=img_size, augment=True)
    val_transform = get_transforms(img_size=img_size, augment=False)
    
    # Datasets
    train_dataset = DefectDataset(data_dir, transform=train_transform, split='train')
    val_dataset = DefectDataset(data_dir, transform=val_transform, split='val')
    test_dataset = DefectDataset(data_dir, transform=val_transform, split='test')
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


class InMemoryDataset(Dataset):
    """
    In-memory dataset for small datasets
    Loads all images into RAM for faster training
    """
    
    def __init__(self, root_dir, transform=None, split='train'):
        self.transform = transform
        self.data = []
        self.labels = []
        
        root_path = Path(root_dir) / split
        
        for class_name, class_idx in [('good', 0), ('defect', 1)]:
            class_dir = root_path / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        img = Image.open(img_path).convert('RGB')
                        img = np.array(img)
                        self.data.append(img)
                        self.labels.append(class_idx)
        
        print(f"{split.upper()} In-Memory Dataset: {len(self.data)} images loaded into RAM")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def download_sample_dataset():
    """
    Download a sample defect detection dataset
    Uses MVTec AD dataset as an example
    """
    print("To use this project, you can download datasets from:")
    print("1. MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad")
    print("2. Kaggle Casting Defects: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
    print("3. NEU Surface Defect: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html")
    print("\nOrganize your data in the following structure:")
    print("data/")
    print("  train/")
    print("    good/")
    print("    defect/")
    print("  val/")
    print("    good/")
    print("    defect/")
    print("  test/")
    print("    good/")
    print("    defect/")


if __name__ == "__main__":
    # Test dataset loading
    print("Dataset module loaded successfully")
    download_sample_dataset()
