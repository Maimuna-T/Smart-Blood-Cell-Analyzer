"""
BloodCell Detection Package
YOLO + Color Space Fusion for Blood Cell Classification
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main classes for easier access
from .detection import BloodCellDetector
from .color_fusion import ColorSpaceFusion
from .main import BloodCellAnalyzer

__all__ = [
    'BloodCellDetector',
    'ColorSpaceFusion',
    'BloodCellAnalyzer'
]