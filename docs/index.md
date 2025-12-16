# VDM-BIND Documentation

Welcome to the VDM-BIND documentation! VDM-BIND is a framework for generating baryonic fields from dark matter-only simulations using diffusion models.

## Overview

VDM-BIND provides:

- **Multiple diffusion model architectures** - VDM, DDPM, Flow Matching, DSM, Consistency Models, DiT
- **BIND inference pipeline** - Apply trained models to full N-body simulations
- **Uncertainty quantification** - MC Dropout, multi-realization, ensemble methods
- **Benchmarking tools** - Standardized metrics for model comparison

```{toctree}
:maxdepth: 2
:caption: Getting Started

quickstart
installation
```

```{toctree}
:maxdepth: 2
:caption: Model Guide

models/index
models/vdm
models/ddpm
models/interpolant
models/dsm
models/consistency
models/ot_flow
models/dit
models/triple_vdm
```

```{toctree}
:maxdepth: 2
:caption: Key Concepts

concepts/index
concepts/uncertainty
concepts/ensemble
concepts/benchmark
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/vdm
api/bind
```

## Quick Example

```python
import vdm
from bind.workflow_utils import ConfigLoader, ModelManager, sample

# Set verbosity (optional)
vdm.set_verbosity('summary')  # 'silent', 'summary', or 'debug'

# Load model
config = ConfigLoader('configs/interpolant.ini')
_, model = ModelManager.initialize(config)
model = model.to('cuda').eval()

# Generate samples
samples = sample(model, condition, batch_size=4)
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
