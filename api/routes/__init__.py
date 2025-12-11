"""
Routes Package
"""

from .main import main_bp
from .detection import detection_bp
from .logs import logs_bp
from .webcam import webcam_bp

__all__ = ['main_bp', 'detection_bp', 'logs_bp', 'webcam_bp']
