"""
InspectAI - Industrial Defect Detection System

A production-ready deep learning system for automated quality control
in manufacturing environments.
"""

__version__ = "1.0.0"
__author__ = "Qaim"
__license__ = "MIT"

from .model import DefectDetectionCNN, LightDefectDetector, get_model
from .inference import DefectPredictor, load_predictor
from .dataset import DefectDataset, create_dataloaders

__all__ = [
    'DefectDetectionCNN',
    'LightDefectDetector',
    'get_model',
    'DefectPredictor',
    'load_predictor',
    'DefectDataset',
    'create_dataloaders',
]
