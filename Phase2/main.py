"""
Blood Cell Detection & Classification System
Main Pipeline: YOLO Detection + Color Space Fusion Classification
"""
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
from detection import BloodCellDetector
from color_fusion import ColorSpaceFusion
from utils import load_image, visualize_color_spaces, save_image, convert_to_rgb
from gui import BloodCellApp


class BloodCellAnalyzer:
    """
    Complete blood cell analysis pipeline:
    1. YOLO detects and localizes cells
    2. Color Space Fusion refines classification
    """
    
    def __init__(self, yolo_model_path=None):
        """
        Initialize the analyzer
        
        Args:
            yolo_model_path: Path to trained YOLO model (optional)
        """
        print("Initializing Blood Cell Analyzer...")
        self.detector = BloodCellDetector(yolo_model_path)
        self.fusion = ColorSpaceFusion()
        print("✓ Analyzer ready!")
    
    def analyze_image(self, image_path, visualize=True, save_results=True):
        """
        Complete analysis of a blood smear image
        
        Args:
            image_path: Path to input image
            visualize: Whether to display results
            save_results: Whether to save output images
            
        Returns:
            dict: Analysis results
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {image_path}")
        print(f"{'='*60}")
        
        # Load image
        image = load_image(image_path)
        print(f"✓ Image loaded: {image.shape}")
        
        # Step 1: YOLO Detection
        print("\n[1/3] Running YOLO detection...")
        detections, crops = self.detector.detect(image, return_crops=True)
        print(f"✓ Detected {len(detections)} cells")
        
        # Count by type
        counts = {'RBC': 0, 'WBC': 0, 'Platelet': 0}
        for det in detections:
            cell_class = det.get('class', 'Unknown')
            counts[cell_class] = counts.get(cell_class, 0) + 1
        
        print(f"  - RBCs: {counts.get('RBC', 0)}")
        print(f"  - WBCs: {counts.get('WBC', 0)}")
        print(f"  - Platelets: {counts.get('Platelet', 0)}")
        
        # Step 2: Color Space Fusion Refinement
        print("\n[2/3] Applying Color Space Fusion...")
        refined_detections = []
        
        for i, (det, crop) in enumerate(zip(detections, crops)):
            if crop is None or crop.size == 0:
                continue
            
            # Extract fused features
            feature_vector, features = self.fusion.fuse_features(crop)
            
            # Refine classification
            refined_class, confidence = self.fusion.classify_cell(crop)
            
            # Update detection with refined result
            refined_det = det.copy()
            refined_det['refined_class'] = refined_class
            refined_det['refined_confidence'] = confidence
            refined_det['features'] = features
            
            refined_detections.append(refined_det)
        
        print(f"✓ Refined {len(refined_detections)} cell classifications")
        
        # Step 3: Visualization
        print("\n[3/3] Creating visualizations...")
        
        # Draw detections
        result_image = self._draw_refined_detections(image, refined_detections)
        
        # Prepare results
        results = {
            'image_path': str(image_path),
            'total_cells': len(refined_detections),
            'counts': counts,
            'detections': refined_detections,
            'result_image': result_image
        }
        
        # Save results
        if save_results:
            output_dir = Path('results/visualizations')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"result_{Path(image_path).name}"
            save_image(result_image, str(output_path))
            print(f"✓ Results saved to: {output_path}")
        
        # Visualize
        if visualize:
            self._display_results(image, result_image, refined_detections)
        
        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"{'='*60}\n")
        
        return results
    
    def _draw_refined_detections(self, image, detections):
        """Draw bounding boxes with refined classifications"""
        result = image.copy()
        
        colors = {
            'RBC': (0, 0, 255),      # Red
            'WBC': (255, 0, 0),      # Blue
            'Platelet': (0, 255, 0), # Green
            'Unknown': (128, 128, 128)  # Gray
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Use refined class if available
            class_name = det.get('refined_class', det.get('class', 'Unknown'))
            confidence = det.get('refined_confidence', det.get('confidence', 0.0))
            
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def _display_results(self, original, result, detections):
        """Display original and result images side by side"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original
        axes[0].imshow(convert_to_rgb(original))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Result
        axes[1].imshow(convert_to_rgb(result))
        axes[1].set_title(f'Detected Cells: {len(detections)}')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_color_spaces(self, image_path):
        """
        Analyze and visualize different color spaces for an image
        
        Args:
            image_path: Path to input image
        """
        image = load_image(image_path)
        visualize_color_spaces(image, save_path='results/color_spaces.png')
    
    def batch_analyze(self, image_dir, output_csv='results/analysis_results.csv'):
        """
        Analyze multiple images in a directory
        
        Args:
            image_dir: Directory containing blood smear images
            output_csv: Path to save CSV results
            
        Returns:
            list: Results for all images
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        print(f"\nBatch Analysis: {len(image_files)} images")
        print("=" * 60)
        
        all_results = []
        
        for img_path in image_files:
            try:
                results = self.analyze_image(img_path, visualize=False, save_results=True)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save summary CSV
        if all_results:
            self._save_results_csv(all_results, output_csv)
        
        return all_results
    
    def _save_results_csv(self, results, output_path):
        """Save analysis results to CSV"""
        import pandas as pd
        
        data = []
        for result in results:
            data.append({
                'image': Path(result['image_path']).name,
                'total_cells': result['total_cells'],
                'rbc_count': result['counts'].get('RBC', 0),
                'wbc_count': result['counts'].get('WBC', 0),
                'platelet_count': result['counts'].get('Platelet', 0)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")


def main():
    """
    Main function - Launch GUI or run CLI mode
    """
    print("\n" + "="*60)
    print("BLOOD CELL DETECTION & CLASSIFICATION")
    print("YOLO + Color Space Fusion")
    print("="*60 + "\n")
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # CLI Mode
        run_cli_mode()
    else:
        # GUI Mode (default)
        run_gui_mode()


def run_gui_mode():
    """Launch the GUI application"""
    print("Launching GUI application...")
    try:
        app = BloodCellApp()
        app.run()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("\nTrying CLI mode instead...")
        run_cli_mode()


def run_cli_mode():
    """Run in command-line mode"""
    print("Running in CLI mode...")
    
    # Initialize analyzer
    analyzer = BloodCellAnalyzer()
    
    # Check for sample images
    data_dir = Path('data/images')
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_images = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
    
    if sample_images:
        print(f"\nFound {len(sample_images)} image(s) in data/images/")
        print("-" * 60)
        
        # Analyze first image
        print("\nAnalyzing first image...")
        results = analyzer.analyze_image(sample_images[0], visualize=False)
        
        # If multiple images, offer batch processing
        if len(sample_images) > 1:
            print(f"\nFound {len(sample_images)} total images.")
            response = input("Run batch analysis on all images? (y/n): ")
            if response.lower() == 'y':
                print("\nStarting batch analysis...")
                analyzer.batch_analyze('data/images')
    else:
        print("⚠ No images found in data/images/")
        print("\nPlease add blood smear images to the 'data/images' folder.")
        print("\nYou can download datasets from:")
        print("  - Kaggle: Blood Cell Images Dataset")
        print("  - Roboflow: Blood Cell Detection Dataset")
        print("\nOr use the GUI to browse and select images.")


if __name__ == "__main__":
    main()