"""
VDM (Variational Diffusion Models) package for cosmological field generation.

This package provides:
- UNetVDM: Score network architecture with Fourier features
- LightCleanVDM: Lightning module for training 3-channel models
- LightTripleVDM: Lightning module for training 3 independent single-channel models
- LightDDPM: Lightning module for DDPM/NCSNpp using score_models package
- LightInterpolant: Lightning module for flow matching / stochastic interpolants
- LightDSM: Lightning module for DSM using custom UNet (fair comparison with VDM)
- AstroDataset: Dataset class for cosmological simulation data

Verbosity Control
-----------------
Control output verbosity across the package:

>>> import vdm
>>> vdm.set_verbosity('silent')   # No output
>>> vdm.set_verbosity('summary')  # Minimal output (default)
>>> vdm.set_verbosity('debug')    # Full verbose output

>>> with vdm.quiet():
...     # All output suppressed in this block
...     model = load_model()
"""

from .networks_clean import UNetVDM
from .vdm_model_clean import CleanVDM, LightCleanVDM
from .astro_dataset import get_astro_data

# Verbosity control
from .verbosity import (
    set_verbosity,
    get_verbosity,
    quiet,
    verbosity,
    vprint,
    vprint_summary,
    vprint_debug,
    SILENT,
    SUMMARY,
    DEBUG,
)

# Try to import DDPM module (requires score_models package)
try:
    from .ddpm_model import LightDDPM
    _HAS_DDPM = True
except ImportError:
    _HAS_DDPM = False

# Import interpolant module (no external dependencies)
from .interpolant_model import LightInterpolant, Interpolant

# Import DSM module (uses custom UNet, no external dependencies)
from .dsm_model import LightDSM

__all__ = [
    # Core models
    'UNetVDM',
    'CleanVDM', 
    'LightCleanVDM',
    'get_astro_data',
    'LightInterpolant',
    'Interpolant',
    'LightDSM',
    # Verbosity control
    'set_verbosity',
    'get_verbosity',
    'quiet',
    'verbosity',
    'vprint',
    'vprint_summary',
    'vprint_debug',
    'SILENT',
    'SUMMARY',
    'DEBUG',
]

if _HAS_DDPM:
    __all__.append('LightDDPM')
