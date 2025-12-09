"""
VDM (Variational Diffusion Models) package for cosmological field generation.

This package provides:
- UNetVDM: Score network architecture with Fourier features
- LightCleanVDM: Lightning module for training 3-channel models
- LightTripleVDM: Lightning module for training 3 independent single-channel models
- AstroDataset: Dataset class for cosmological simulation data
"""

from .networks_clean import UNetVDM
from .vdm_model_clean import CleanVDM, LightCleanVDM
from .astro_dataset import get_astro_data

__all__ = [
    'UNetVDM',
    'CleanVDM', 
    'LightCleanVDM',
    'get_astro_data',
]
