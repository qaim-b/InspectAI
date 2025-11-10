"""
Inference utilities for InspectAI
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json

from model import get_model


class DefectPredictor:
    """
    Predictor class for defect detection
    """
    
    def __init__(self, model_path, device='cuda', img_size=224):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            img_size: Input image size
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Initialize model
        self.model = get_model(
            model_type=config.get('model_type', 'full'),
            num_classes=config.get('num_classes', 2)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Class names
        self.class_names = ['Good', 'Defect']
        
        # Preprocessing
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        print(f"Model loaded successfully on {self.device}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image, return_probs=True):
        """
        Make prediction on single image
        
        Args:
            image: PIL Image or numpy array
            return_probs: Whether to return probabilities
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        # Results
        pred_class = predicted.item()
        pred_label = self.class_names[pred_class]
        confidence_score = confidence.item()
        
        result = {
            'class': pred_class,
            'label': pred_label,
            'confidence': float(confidence_score),
        }
        
        if return_probs:
            result['probabilities'] = {
                self.class_names[i]: float(probs[0][i])
                for i in range(len(self.class_names))
            }
        
        return result
    
    def predict_batch(self, images):
        """
        Make predictions on batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def predict_from_path(self, image_path):
        """
        Make prediction from image file path
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with prediction results
        """
        image = Image.open(image_path).convert('RGB')
        return self.predict(image)
    
    def visualize_prediction(self, image, save_path=None):
        """
        Visualize prediction with overlay
        
        Args:
            image: PIL Image or numpy array
            save_path: Optional path to save visualization
        
        Returns:
            Visualization image as numpy array
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image.copy()
        
        # Make prediction
        result = self.predict(image)
        
        # Create visualization
        h, w = image_np.shape[:2]
        vis_image = image_np.copy()
        
        # Add prediction text
        label = result['label']
        confidence = result['confidence']
        text = f"{label}: {confidence:.2%}"
        
        # Color based on prediction
        color = (0, 255, 0) if label == 'Good' else (255, 0, 0)
        
        # Add text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Draw rectangle background
        cv2.rectangle(vis_image, 
                     (10, 10), 
                     (text_size[0] + 20, text_size[1] + 30),
                     color, -1)
        
        # Draw text
        cv2.putText(vis_image, text, (15, 40), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Add border based on prediction
        border_width = 10
        vis_image = cv2.copyMakeBorder(
            vis_image, 
            border_width, border_width, border_width, border_width,
            cv2.BORDER_CONSTANT, 
            value=color
        )
        
        # Save if path provided
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image
    
    def get_model_info(self):
        """Get model information"""
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            'device': str(self.device),
            'parameters': num_params,
            'classes': self.class_names,
            'input_size': self.img_size
        }


def load_predictor(model_path='models/best_acc.pth', device='cuda'):
    """
    Convenience function to load predictor
    
    Args:
        model_path: Path to model checkpoint
        device: Device to use
    
    Returns:
        DefectPredictor instance
    """
    return DefectPredictor(model_path, device=device)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [model_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'models/best_acc.pth'
    
    # Load predictor
    predictor = load_predictor(model_path)
    
    # Make prediction
    result = predictor.predict_from_path(image_path)
    
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2))
    
    # Create visualization
    image = Image.open(image_path)
    vis = predictor.visualize_prediction(image, save_path='prediction_result.jpg')
    print("\nVisualization saved to: prediction_result.jpg")
