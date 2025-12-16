# API Reference

Detailed API documentation for all VDM-BIND modules.

```{toctree}
:maxdepth: 2

vdm
bind
```

## Package Overview

### vdm Package

The `vdm` package contains all generative model implementations and utilities:

```python
import vdm

# Verbosity control
vdm.set_verbosity('silent')  # or 'summary', 'debug'
vdm.quiet()  # Context manager for silent execution

# Import models
from vdm.vdm_model_clean import LightCleanVDM
from vdm.interpolant_model import LightInterpolant
from vdm.ddpm_model import LightDDPM
```

### bind Package

The `bind` package contains the BIND inference pipeline:

```python
from bind import BIND
from bind.workflow_utils import ConfigLoader, ModelManager, sample

# Load configuration
config = ConfigLoader('configs/clean_vdm_aggressive_stellar.ini')

# Initialize model
data_module, model = ModelManager.initialize(config)

# Run BIND on simulation
bind_runner = BIND(model, config)
result = bind_runner.process_simulation(dmo_field, halo_catalog)
```

## Module Index

### vdm

| Module | Description |
|--------|-------------|
| [`vdm.vdm_model_clean`](vdm.md#vdm-model) | VDM (3-channel) model |
| [`vdm.interpolant_model`](vdm.md#interpolant-model) | Flow Matching / Interpolants |
| [`vdm.ddpm_model`](vdm.md#ddpm-model) | DDPM with NCSNpp |
| [`vdm.dsm_model`](vdm.md#dsm-model) | Denoising Score Matching |
| [`vdm.consistency_model`](vdm.md#consistency-model) | Consistency Models |
| [`vdm.dit_model`](vdm.md#dit-model) | Diffusion Transformer |
| [`vdm.networks_clean`](vdm.md#networks) | UNet architecture |
| [`vdm.uncertainty`](vdm.md#uncertainty) | Uncertainty quantification |
| [`vdm.ensemble`](vdm.md#ensemble) | Model ensembles |
| [`vdm.benchmark`](vdm.md#benchmark) | Evaluation metrics |

### bind

| Module | Description |
|--------|-------------|
| [`bind.bind`](bind.md#bind-class) | Main BIND class |
| [`bind.workflow_utils`](bind.md#workflow-utils) | ConfigLoader, ModelManager |
| [`bind.power_spec`](bind.md#power-spectrum) | Power spectrum analysis |
| [`bind.analyses`](bind.md#analyses) | Evaluation utilities |
