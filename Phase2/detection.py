"""
YOLO-based Blood Cell Detection Module
Handles cell detection and localization using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch


class BloodCellDetector:
    """
    Blood cell detector using YOLOv8
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """
        Initialize the detector
        
        Args:
            model_path: Path to trained YOLO model (if None, uses pretrained)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        
        # Load model
        if model_path and Path(model_path).exists():
            print(f"Loading custom model from {model_path}")
            self.model = YOLO(model_path)
        else:
            # Start with pretrained YOLOv8 nano model (fastest)
            # You'll need to train this on blood cell dataset later
            print("Loading YOLOv8n pretrained model (you'll need to train on blood cells)")
            self.model = YOLO('yolov8n.pt')
        
        # Class names for blood cells
        self.class_names = {
            0: 'RBC',      # Red Blood Cell
            1: 'WBC',      # White Blood Cell
            2: 'Platelet'  # Platelet
        }
    
    def detect(self, image, return_crops=False):
        """
        Detect blood cells in an image
        
        Args:
            image: Input image (numpy array)
            return_crops: If True, also return cropped cell images
            
        Returns:
            list: Detections with bbox, class, confidence
            list: Cropped images (if return_crops=True)
        """
        # Run YOLO inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        crops = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = self.class_names.get(class_id, f'Class_{class_id}')
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class': class_name,
                    'class_id': class_id
                }
                
                detections.append(detection)
                
                # Extract crop if requested
                if return_crops:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    crop = image[y1:y2, x1:x2]
                    crops.append(crop)
        
        if return_crops:
            return detections, crops
        return detections
    
    def detect_and_count(self, image):
        """
        Detect cells and count by type
        
        Args:
            image: Input image
            
        Returns:
            dict: Cell counts by type
            list: All detections
        """
        detections = self.detect(image)
        
        # Count by class
        counts = {
            'RBC': 0,
            'WBC': 0,
            'Platelet': 0,
            'Total': len(detections)
        }
        
        for det in detections:
            class_name = det['class']
            if class_name in counts:
                counts[class_name] += 1
        
        return counts, detections
    
    def train(self, data_yaml, epochs=50, img_size=640, batch_size=16):
        """
        Train YOLO model on blood cell dataset
        
        Args:
            data_yaml: Path to dataset configuration file
            epochs: Number of training epochs
            img_size: Input image size
            batch_size: Training batch size
            
        Returns:
            Training results
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Image size: {img_size}, Batch size: {batch_size}")
        
        # Train the model
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            project='models',
            name='blood_cell_yolo'
        )
        
        print("Training complete!")
        return results
    
    def evaluate(self, data_yaml):
        """
        Evaluate model on validation dataset
        
        Args:
            data_yaml: Path to dataset configuration
            
        Returns:
            Validation metrics
        """
        print("Evaluating model...")
        metrics = self.model.val(data=data_yaml)
        
        print(f"mAP50: {metrics.box.map50:.3f}")
        print(f"mAP50-95: {metrics.box.map:.3f}")
        
        return metrics
    
    def save_model(self, save_path):
        """
        Save trained model
        
        Args:
            save_path: Where to save the model
        """
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def visualize_detections(self, image, detections):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with drawn boxes
        """
        result_img = image.copy()
        
        # Colors for each class
        colors = {
            'RBC': (0, 0, 255),      # Red
            'WBC': (255, 0, 0),      # Blue
            'Platelet': (0, 255, 0)  # Green
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class']
            confidence = det['confidence']
            
            # Get color
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(result_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_img


# Example usage
if __name__ == "__main__":
    print("Blood Cell Detector Module")
    print("=" * 50)
    print("Using YOLOv8 for cell detection")
    print("Supported classes: RBC, WBC, Platelet")
    print("=" * 50)
    
    # Initialize detector
    detector = BloodCellDetector()
    print("Detector initialized successfully!")