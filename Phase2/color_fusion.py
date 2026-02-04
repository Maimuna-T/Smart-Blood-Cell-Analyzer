"""
Color Space Fusion for Blood Cell Classification
This module implements the core algorithm: RGB + HSV + LAB feature fusion
"""

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler


class ColorSpaceFusion:
    """
    Extract and fuse features from multiple color spaces (RGB, HSV, LAB)
    for improved blood cell classification
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_rgb_features(self, image):
        """
        Extract features from RGB color space
        
        Args:
            image: Input image in BGR format
            
        Returns:
            dict: RGB-based features
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        features = {}
        
        # Mean color values for each channel
        features['r_mean'] = np.mean(rgb[:, :, 0])
        features['g_mean'] = np.mean(rgb[:, :, 1])
        features['b_mean'] = np.mean(rgb[:, :, 2])
        
        # Standard deviation (color uniformity)
        features['r_std'] = np.std(rgb[:, :, 0])
        features['g_std'] = np.std(rgb[:, :, 1])
        features['b_std'] = np.std(rgb[:, :, 2])
        
        # Color histogram features (simplified)
        for i, color in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([rgb], [i], None, [16], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features[f'{color}_hist_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return features
    
    def extract_hsv_features(self, image):
        """
        Extract features from HSV color space
        Particularly useful for nucleus detection (Hue) and cell segmentation
        
        Args:
            image: Input image in BGR format
            
        Returns:
            dict: HSV-based features
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        features = {}
        
        # Hue features (important for nucleus color - purple/blue in stained cells)
        features['hue_mean'] = np.mean(hsv[:, :, 0])
        features['hue_std'] = np.std(hsv[:, :, 0])
        
        # Saturation features (color purity)
        features['sat_mean'] = np.mean(hsv[:, :, 1])
        features['sat_std'] = np.std(hsv[:, :, 1])
        
        # Value features (brightness)
        features['val_mean'] = np.mean(hsv[:, :, 2])
        features['val_std'] = np.std(hsv[:, :, 2])
        
        # Nucleus detection: pixels in purple/blue range
        # Typical nucleus hue range: 120-150 (purple-blue in HSV)
        nucleus_mask = ((hsv[:, :, 0] >= 120) & (hsv[:, :, 0] <= 150) & 
                       (hsv[:, :, 1] > 50))  # High saturation
        features['nucleus_ratio'] = np.sum(nucleus_mask) / (image.shape[0] * image.shape[1])
        
        return features
    
    def extract_lab_features(self, image):
        """
        Extract features from LAB color space
        LAB is perceptually uniform - good for cytoplasm texture and subtle color differences
        
        Args:
            image: Input image in BGR format
            
        Returns:
            dict: LAB-based features
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        features = {}
        
        # Lightness channel
        features['l_mean'] = np.mean(lab[:, :, 0])
        features['l_std'] = np.std(lab[:, :, 0])
        
        # A channel (green to red)
        features['a_mean'] = np.mean(lab[:, :, 1])
        features['a_std'] = np.std(lab[:, :, 1])
        
        # B channel (blue to yellow)
        features['b_mean'] = np.mean(lab[:, :, 2])
        features['b_std'] = np.std(lab[:, :, 2])
        
        # Texture features using gradient magnitude on L channel
        sobelx = cv2.Sobel(lab[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(lab[:, :, 0], cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features['texture_mean'] = np.mean(gradient_magnitude)
        features['texture_std'] = np.std(gradient_magnitude)
        
        return features
    
    def extract_shape_features(self, image):
        """
        Extract basic shape features (size, aspect ratio)
        
        Args:
            image: Input image
            
        Returns:
            dict: Shape features
        """
        features = {}
        
        height, width = image.shape[:2]
        features['area'] = height * width
        features['aspect_ratio'] = width / height if height > 0 else 1.0
        features['perimeter_approx'] = 2 * (height + width)
        
        return features
    
    def fuse_features(self, image):
        """
        Main feature fusion function
        Combines RGB + HSV + LAB features into a single feature vector
        
        Args:
            image: Input image (cropped cell region)
            
        Returns:
            numpy.ndarray: Fused feature vector
            dict: Individual feature dictionaries for analysis
        """
        # Extract features from each color space
        rgb_features = self.extract_rgb_features(image)
        hsv_features = self.extract_hsv_features(image)
        lab_features = self.extract_lab_features(image)
        shape_features = self.extract_shape_features(image)
        
        # Combine all features
        all_features = {
            **rgb_features,
            **hsv_features,
            **lab_features,
            **shape_features
        }
        
        # Convert to feature vector
        feature_vector = np.array(list(all_features.values()))
        
        return feature_vector, all_features
    
    def classify_cell(self, image, feature_weights=None):
        """
        Classify blood cell based on fused features
        This is a rule-based classifier - you can replace with ML model later
        
        Args:
            image: Input cell image
            feature_weights: Optional feature weighting scheme
            
        Returns:
            str: Predicted class ('RBC', 'WBC', or 'Platelet')
            float: Confidence score
        """
        feature_vector, features = self.fuse_features(image)
        
        # Rule-based classification logic
        # These thresholds are approximate - tune based on your dataset
        
        # WBC: Has nucleus (high nucleus_ratio), larger size
        if features['nucleus_ratio'] > 0.15 and features['area'] > 1000:
            return 'WBC', 0.85
        
        # Platelet: Very small, irregular, purple fragments
        elif features['area'] < 500 and features['hue_mean'] > 100:
            return 'Platelet', 0.75
        
        # RBC: Medium size, no nucleus, reddish (low hue value)
        elif features['nucleus_ratio'] < 0.05 and 500 < features['area'] < 3000:
            return 'RBC', 0.80
        
        # Default fallback
        else:
            return 'Unknown', 0.50
    
    def get_feature_importance(self, features):
        """
        Analyze which features are most discriminative
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            dict: Feature importance scores
        """
        importance = {}
        
        # Higher variance = more discriminative
        importance['nucleus_detection'] = features['nucleus_ratio'] * 10
        importance['color_variation'] = (features['hue_std'] + features['sat_std']) / 2
        importance['texture_complexity'] = features['texture_std']
        importance['size'] = np.log(features['area'] + 1)
        
        return importance


# Example usage and testing
if __name__ == "__main__":
    print("Color Space Fusion Module")
    print("=" * 50)
    print("This module extracts features from:")
    print("  - RGB: Color information")
    print("  - HSV: Nucleus detection, cell segmentation")
    print("  - LAB: Cytoplasm texture, perceptual differences")
    print("=" * 50)