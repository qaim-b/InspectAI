"""
Data preparation and augmentation utilities
Downloads and prepares datasets for training
"""

import os
import shutil
import requests
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


def create_directory_structure(base_dir='data'):
    """
    Create the required directory structure
    
    Args:
        base_dir: Base directory for data
    """
    base_path = Path(base_dir)
    
    for split in ['train', 'val', 'test']:
        for class_name in ['good', 'defect']:
            path = base_path / split / class_name
            path.mkdir(parents=True, exist_ok=True)
    
    (base_path / 'raw').mkdir(exist_ok=True)
    (base_path / 'processed').mkdir(exist_ok=True)
    (base_path / 'samples').mkdir(exist_ok=True)
    
    print(f"Directory structure created at {base_path}")


def split_dataset(source_dir, output_dir='data', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test
    
    Args:
        source_dir: Directory containing good/ and defect/ folders
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    create_directory_structure(output_dir)
    
    for class_name in ['good', 'defect']:
        class_dir = source_path / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist, skipping...")
            continue
        
        # Get all images
        images = list(class_dir.glob('*'))
        images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Shuffle
        np.random.shuffle(images)
        
        # Split
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Copy files
        for split, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            dest_dir = output_path / split / class_name
            for img_path in tqdm(split_images, desc=f'{class_name} - {split}'):
                shutil.copy2(img_path, dest_dir / img_path.name)
        
        print(f"{class_name}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")


def generate_synthetic_defects(good_dir, defect_dir, num_samples=100):
    """
    Generate synthetic defects from good images
    Useful for data augmentation when you have limited defect samples
    
    Args:
        good_dir: Directory with good images
        defect_dir: Output directory for synthetic defects
        num_samples: Number of synthetic samples to generate
    """
    good_path = Path(good_dir)
    defect_path = Path(defect_dir)
    defect_path.mkdir(parents=True, exist_ok=True)
    
    good_images = list(good_path.glob('*'))
    good_images = [img for img in good_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    print(f"Generating {num_samples} synthetic defects...")
    
    for i in tqdm(range(num_samples)):
        # Select random good image
        img_path = np.random.choice(good_images)
        img = cv2.imread(str(img_path))
        
        # Apply random defect simulation
        defect_type = np.random.choice(['scratch', 'spot', 'crack', 'discoloration'])
        
        if defect_type == 'scratch':
            # Add scratch
            h, w = img.shape[:2]
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            thickness = np.random.randint(2, 8)
            cv2.line(img, pt1, pt2, color, thickness)
        
        elif defect_type == 'spot':
            # Add spots
            h, w = img.shape[:2]
            num_spots = np.random.randint(3, 10)
            for _ in range(num_spots):
                center = (np.random.randint(0, w), np.random.randint(0, h))
                radius = np.random.randint(5, 20)
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.circle(img, center, radius, color, -1)
        
        elif defect_type == 'crack':
            # Add crack pattern
            h, w = img.shape[:2]
            pts = []
            for _ in range(np.random.randint(5, 15)):
                pts.append((np.random.randint(0, w), np.random.randint(0, h)))
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], False, (0, 0, 0), 2)
        
        elif defect_type == 'discoloration':
            # Add discoloration
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (np.random.randint(0, w), np.random.randint(0, h))
            radius = np.random.randint(30, min(h, w) // 3)
            cv2.circle(mask, center, radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            mask = mask / 255.0
            
            color_shift = np.array([
                np.random.randint(-50, 50),
                np.random.randint(-50, 50),
                np.random.randint(-50, 50)
            ])
            
            for c in range(3):
                img[:, :, c] = np.clip(
                    img[:, :, c] * (1 - mask) + (img[:, :, c] + color_shift[c]) * mask,
                    0, 255
                ).astype(np.uint8)
        
        # Save synthetic defect
        output_path = defect_path / f'synthetic_{i:04d}.jpg'
        cv2.imwrite(str(output_path), img)
    
    print(f"Generated {num_samples} synthetic defects at {defect_path}")


def balance_dataset(data_dir='data'):
    """
    Balance the dataset by oversampling minority class
    
    Args:
        data_dir: Data directory
    """
    data_path = Path(data_dir)
    
    for split in ['train']:  # Usually only balance training set
        good_dir = data_path / split / 'good'
        defect_dir = data_path / split / 'defect'
        
        good_images = list(good_dir.glob('*'))
        defect_images = list(defect_dir.glob('*'))
        
        n_good = len(good_images)
        n_defect = len(defect_images)
        
        print(f"{split}: Good={n_good}, Defect={n_defect}")
        
        if n_good > n_defect:
            # Oversample defects
            difference = n_good - n_defect
            samples_to_copy = np.random.choice(defect_images, difference, replace=True)
            
            for i, img_path in enumerate(tqdm(samples_to_copy, desc='Balancing defects')):
                img = Image.open(img_path)
                # Apply random flip
                if np.random.rand() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if np.random.rand() > 0.5:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                output_path = defect_dir / f'balanced_{i:04d}{img_path.suffix}'
                img.save(output_path)
        
        elif n_defect > n_good:
            # Oversample good
            difference = n_defect - n_good
            samples_to_copy = np.random.choice(good_images, difference, replace=True)
            
            for i, img_path in enumerate(tqdm(samples_to_copy, desc='Balancing good')):
                img = Image.open(img_path)
                if np.random.rand() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if np.random.rand() > 0.5:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                output_path = good_dir / f'balanced_{i:04d}{img_path.suffix}'
                img.save(output_path)
        
        # Print new counts
        good_images = list(good_dir.glob('*'))
        defect_images = list(defect_dir.glob('*'))
        print(f"After balancing: Good={len(good_images)}, Defect={len(defect_images)}")


def create_sample_dataset(output_dir='data', samples_per_class=50):
    """
    Create a small sample dataset for quick testing
    Generates synthetic images
    
    Args:
        output_dir: Output directory
        samples_per_class: Number of samples per class
    """
    create_directory_structure(output_dir)
    
    print("Creating sample dataset...")
    
    for split, n_samples in [('train', samples_per_class), ('val', samples_per_class//4), ('test', samples_per_class//4)]:
        # Generate good samples (clean backgrounds)
        good_dir = Path(output_dir) / split / 'good'
        for i in tqdm(range(n_samples), desc=f'{split} - good'):
            # Create clean image
            img = np.ones((224, 224, 3), dtype=np.uint8) * 200
            # Add some texture
            noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img)
            img.save(good_dir / f'good_{i:04d}.jpg')
        
        # Generate defect samples (with visible defects)
        defect_dir = Path(output_dir) / split / 'defect'
        for i in tqdm(range(n_samples), desc=f'{split} - defect'):
            # Create base image
            img = np.ones((224, 224, 3), dtype=np.uint8) * 200
            noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Add defect
            defect_type = np.random.choice(['scratch', 'spot', 'crack'])
            
            if defect_type == 'scratch':
                cv2.line(img, 
                        (np.random.randint(0, 224), np.random.randint(0, 224)),
                        (np.random.randint(0, 224), np.random.randint(0, 224)),
                        (0, 0, 0), np.random.randint(3, 8))
            
            elif defect_type == 'spot':
                for _ in range(np.random.randint(3, 8)):
                    cv2.circle(img,
                             (np.random.randint(0, 224), np.random.randint(0, 224)),
                             np.random.randint(10, 25),
                             (0, 0, 0), -1)
            
            elif defect_type == 'crack':
                pts = [(np.random.randint(0, 224), np.random.randint(0, 224)) 
                       for _ in range(np.random.randint(8, 15))]
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(img, [pts], False, (0, 0, 0), 3)
            
            img = Image.fromarray(img)
            img.save(defect_dir / f'defect_{i:04d}.jpg')
    
    print(f"Sample dataset created at {output_dir}")
    print("You can now train the model with: python src/train.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data preparation utilities')
    parser.add_argument('--action', type=str, required=True,
                       choices=['create_structure', 'split', 'balance', 'synthetic', 'sample'],
                       help='Action to perform')
    parser.add_argument('--source', type=str, help='Source directory')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    
    args = parser.parse_args()
    
    if args.action == 'create_structure':
        create_directory_structure(args.output)
    
    elif args.action == 'split':
        if not args.source:
            raise ValueError("--source required for split action")
        split_dataset(args.source, args.output)
    
    elif args.action == 'balance':
        balance_dataset(args.output)
    
    elif args.action == 'synthetic':
        if not args.source:
            raise ValueError("--source required for synthetic action")
        generate_synthetic_defects(args.source, args.output, args.samples)
    
    elif args.action == 'sample':
        create_sample_dataset(args.output, args.samples)
