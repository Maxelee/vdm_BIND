"""
BIND (Baryonic Inference from N-body Data) package.

This package provides:
- BIND: Main class for the inference pipeline
- ConfigLoader: Load model configurations
- ModelManager: Initialize and load trained models
- sample: Generate samples from trained models
- load_normalization_stats: Load normalization statistics
- Analysis functions for evaluation

The package supports two import styles:
1. Modular (preferred): 
   `from bind.config_loader import ConfigLoader, load_normalization_stats`
   `from bind.model_manager import ModelManager`
   `from bind.sampling import sample`
2. Legacy (from workflow_utils - still works for backward compatibility):
   `from bind.workflow_utils import ConfigLoader, ModelManager, sample`
"""

from .bind import BIND, HaloPaster2D, HaloPaster3D

# Import from new modular files (canonical locations)
from .config_loader import ConfigLoader, load_normalization_stats
from .model_manager import ModelManager
from .sampling import sample

__all__ = [
    'BIND',
    'HaloPaster2D',
    'HaloPaster3D',
    'ConfigLoader',
    'ModelManager',
    'sample',
    'load_normalization_stats',
]

