"""
Configuration file for InspectAI
Centralized settings for training and inference
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Model configuration
MODEL_CONFIG = {
    'type': 'full',  # 'full' or 'light'
    'num_classes': 2,
    'input_size': 224,
    'pretrained': False,
}

# Training configuration
TRAIN_CONFIG = {
    # Data
    'data_dir': str(DATA_DIR),
    'batch_size': 32,
    'num_workers': 4,
    'img_size': 224,
    
    # Training
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'adamw',  # 'adam', 'sgd', 'adamw'
    'scheduler': 'cosine',  # 'cosine', 'plateau', None
    
    # Regularization
    'dropout': 0.5,
    'use_weighted_loss': False,
    'class_weights': [1.0, 1.0],
    'early_stop_patience': 10,
    
    # Augmentation
    'augmentation': {
        'horizontal_flip': 0.5,
        'vertical_flip': 0.3,
        'rotate90': 0.5,
        'gaussian_noise': 0.3,
        'brightness_contrast': 0.3,
        'shift_scale_rotate': 0.5,
        'coarse_dropout': 0.3,
    },
    
    # Output
    'output_dir': str(MODELS_DIR),
    'save_frequency': 5,  # Save checkpoint every N epochs
}

# Inference configuration
INFERENCE_CONFIG = {
    'model_path': str(MODELS_DIR / 'checkpoints' / 'best_acc.pth'),
    'device': 'cuda',  # 'cuda' or 'cpu'
    'batch_size': 16,
    'confidence_threshold': 0.5,
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'log_level': 'info',
    'workers': 1,
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
}

# Data preparation configuration
DATA_PREP_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'balance_dataset': True,
    'generate_synthetic': False,
    'synthetic_samples': 100,
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(LOGS_DIR / 'inspectai.log'),
}

# Class names
CLASS_NAMES = ['Good', 'Defect']

# Image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Model metadata
MODEL_METADATA = {
    'name': 'InspectAI',
    'version': '1.0.0',
    'description': 'Industrial Defect Detection System',
    'author': 'Qaim',
    'license': 'MIT',
}

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, MODELS_DIR / 'checkpoints']:
    dir_path.mkdir(parents=True, exist_ok=True)


def get_config(config_name):
    """
    Get configuration by name
    
    Args:
        config_name: Name of configuration
            ('model', 'train', 'inference', 'api', 'data_prep')
    
    Returns:
        Configuration dictionary
    """
    configs = {
        'model': MODEL_CONFIG,
        'train': TRAIN_CONFIG,
        'inference': INFERENCE_CONFIG,
        'api': API_CONFIG,
        'data_prep': DATA_PREP_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    
    return configs[config_name]


def update_config(config_name, updates):
    """
    Update configuration
    
    Args:
        config_name: Name of configuration to update
        updates: Dictionary of updates
    """
    config = get_config(config_name)
    config.update(updates)


if __name__ == "__main__":
    # Print all configurations
    print("InspectAI Configuration")
    print("=" * 60)
    
    print("\nModel Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in TRAIN_CONFIG.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\nInference Configuration:")
    for key, value in INFERENCE_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nAPI Configuration:")
    for key, value in API_CONFIG.items():
        print(f"  {key}: {value}")
