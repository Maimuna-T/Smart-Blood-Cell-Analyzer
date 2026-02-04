"""
Utility functions for blood cell detection and color space processing
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    Load an image from file path
    
    Args:
        image_path: Path to the image file
    
    Returns:
        numpy.ndarray: Image in BGR format (OpenCV default)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def convert_to_hsv(image):
    """
    Convert RGB/BGR image to HSV color space
    
    Args:
        image: Input image in BGR format
    
    Returns:
        numpy.ndarray: Image in HSV format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convert_to_lab(image):
    """
    Convert RGB/BGR image to LAB color space
    
    Args:
        image: Input image in BGR format
    
    Returns:
        numpy.ndarray: Image in LAB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def convert_to_rgb(image):
    """
    Convert BGR image to RGB (for displaying with matplotlib)
    
    Args:
        image: Input image in BGR format
    
    Returns:
        numpy.ndarray: Image in RGB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_roi(image, bbox):
    """
    Extract Region of Interest (ROI) from image using bounding box
    
    Args:
        image: Input image
        bbox: Bounding box as [x1, y1, x2, y2]
    
    Returns:
        numpy.ndarray: Cropped image region
    """
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]


def visualize_color_spaces(image, save_path=None):
    """
    Visualize image in different color spaces side by side
    
    Args:
        image: Input image in BGR format
        save_path: Optional path to save the visualization
    """
    # Convert to different color spaces
    rgb = convert_to_rgb(image)
    hsv = convert_to_hsv(image)
    lab = convert_to_lab(image)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Color Space Comparison', fontsize=16)
    
    # RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # RGB channels
    axes[1, 0].imshow(rgb[:, :, 0], cmap='Reds')
    axes[1, 0].set_title('R Channel')
    axes[1, 0].axis('off')
    
    # HSV
    axes[0, 1].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    axes[0, 1].set_title('HSV')
    axes[0, 1].axis('off')
    
    # HSV - Hue
    axes[1, 1].imshow(hsv[:, :, 0], cmap='hsv')
    axes[1, 1].set_title('Hue (H)')
    axes[1, 1].axis('off')
    
    # HSV - Saturation
    axes[0, 2].imshow(hsv[:, :, 1], cmap='gray')
    axes[0, 2].set_title('Saturation (S)')
    axes[0, 2].axis('off')
    
    # HSV - Value
    axes[1, 2].imshow(hsv[:, :, 2], cmap='gray')
    axes[1, 2].set_title('Value (V)')
    axes[1, 2].axis('off')
    
    # LAB - L
    axes[0, 3].imshow(lab[:, :, 0], cmap='gray')
    axes[0, 3].set_title('Lightness (L)')
    axes[0, 3].axis('off')
    
    # LAB - A
    axes[1, 3].imshow(lab[:, :, 1], cmap='RdYlGn')
    axes[1, 3].set_title('A Channel (Green-Red)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def draw_detections(image, detections, class_names):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: Input image
        detections: List of detections with bbox, confidence, class
        class_names: Dictionary mapping class IDs to names
    
    Returns:
        numpy.ndarray: Image with drawn detections
    """
    result_image = image.copy()
    
    colors = {
        'RBC': (0, 0, 255),      # Red
        'WBC': (255, 0, 0),      # Blue
        'Platelet': (0, 255, 0)  # Green
    }
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Get color for this class
        color = colors.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
    
    return result_image


def save_image(image, save_path):
    """
    Save image to file
    
    Args:
        image: Image to save
        save_path: Path where to save the image
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)
    print(f"Image saved to {save_path}")


def resize_image(image, width=None, height=None):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
    
    Returns:
        numpy.ndarray: Resized image
    """
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    
    if width is not None:
        ratio = width / w
        new_size = (width, int(h * ratio))
    else:
        ratio = height / h
        new_size = (int(w * ratio), height)
    
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def create_grid_visualization(images, titles, rows=2, cols=3, figsize=(15, 10)):
    """
    Create a grid visualization of multiple images
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        rows: Number of rows in grid
        cols: Number of columns in grid
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        if idx < len(axes):
            if len(img.shape) == 2:  # Grayscale
                axes[idx].imshow(img, cmap='gray')
            else:  # Color
                axes[idx].imshow(convert_to_rgb(img))
            axes[idx].set_title(title)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_image_statistics(image):
    """
    Calculate basic statistics of an image
    
    Args:
        image: Input image
    
    Returns:
        dict: Image statistics
    """
    stats = {
        'shape': image.shape,
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'median': np.median(image)
    }
    
    if len(image.shape) == 3:
        stats['channels'] = image.shape[2]
        for i in range(image.shape[2]):
            stats[f'channel_{i}_mean'] = np.mean(image[:, :, i])
    
    return stats