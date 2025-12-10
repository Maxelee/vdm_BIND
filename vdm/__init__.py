"""
VDM (Variational Diffusion Models) package for cosmological field generation.

This package provides:
- UNetVDM: Score network architecture with Fourier features
- LightCleanVDM: Lightning module for training 3-channel models
- LightTripleVDM: Lightning module for training 3 independent single-channel models
- LightDDPM: Lightning module for DDPM/NCSNpp using score_models package
- AstroDataset: Dataset class for cosmological simulation data
"""

from .networks_clean import UNetVDM
from .vdm_model_clean import CleanVDM, LightCleanVDM
from .astro_dataset import get_astro_data

# Try to import DDPM module (requires score_models package)
try:
    from .ddpm_model import LightDDPM
    _HAS_DDPM = True
except ImportError:
    _HAS_DDPM = False

__all__ = [
    'UNetVDM',
    'CleanVDM', 
    'LightCleanVDM',
    'get_astro_data',
]

if _HAS_DDPM:
    __all__.append('LightDDPM')
