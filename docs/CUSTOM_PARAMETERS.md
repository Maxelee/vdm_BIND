# Custom Parameter Conditioning Guide

This guide explains how to configure VDM-BIND with custom parameter conditioning for your own simulations or datasets.

## Overview

VDM-BIND can be trained with **conditional** or **unconditional** generation:

- **Conditional**: Model learns to generate baryonic fields given cosmological/astrophysical parameters
- **Unconditional**: Model learns a general mapping from dark matter to baryons without explicit parameters

## Quick Start

### Unconditional Training (No Parameters)

```ini
# In your config file
n_params = 0
# OR
param_norm_path = none
```

### Custom Parameters (Inline)

```ini
# 6 custom parameters with min/max bounds
param_min = 0.1, 0.5, 0.0, 0.0, 0.1, 1e4
param_max = 0.5, 1.2, 1.0, 1.0, 10.0, 1e6
```

### Custom Parameters (File)

```ini
# CSV or JSON file
param_norm_path = /path/to/custom_params.json
```

## Parameter Specification Methods

### 1. Unconditional (`n_params = 0`)

For datasets without cosmological parameters or when you want a general model:

```ini
[TRAINING]
n_params = 0
use_param_prediction = False  # Disable auxiliary prediction task
```

The model will still use:
- Dark matter density field conditioning
- Large-scale environment context
- Time/noise level (intrinsic to diffusion)

### 2. Inline Specification

Directly in your `.ini` config file:

```ini
[TRAINING]
# Comma-separated min values
param_min = 0.1, 0.5, 0.0, 0.0, 0.1, 10000.0
# Comma-separated max values (must have same count)
param_max = 0.5, 1.2, 1.0, 1.0, 10.0, 1000000.0
```

Parameters are normalized to [0, 1] using: `(value - min) / (max - min)`

### 3. CSV File

Create a CSV with `MinVal` and `MaxVal` columns:

```csv
Parameter,MinVal,MaxVal
Omega_m,0.1,0.5
sigma_8,0.5,1.2
AGN_eff,0.0,1.0
SN_eff,0.0,1.0
SF_thresh,0.1,10.0
BH_seed,10000.0,1000000.0
```

Then in config:
```ini
param_norm_path = /path/to/params.csv
```

### 4. JSON File

More expressive format with optional metadata:

```json
{
    "param_names": ["Omega_m", "sigma_8", "AGN_eff", "SN_eff", "SF_thresh", "BH_seed"],
    "param_min": [0.1, 0.5, 0.0, 0.0, 0.1, 10000.0],
    "param_max": [0.5, 1.2, 1.0, 1.0, 10.0, 1000000.0],
    "param_units": ["dimensionless", "dimensionless", "dimensionless", "dimensionless", "cm^-3", "M_sun"],
    "description": "Custom simulation parameters"
}
```

Then in config:
```ini
param_norm_path = /path/to/params.json
```

## Training Data Format

Your `.npz` training files must include a `params` array:

```python
import numpy as np

# For 6 parameters
np.savez(
    'halo_000_000.npz',
    condition=dm_field,           # (128, 128) dark matter
    target=target_fields,          # (3, 128, 128) [DM_hydro, Gas, Stars]
    large_scale=large_scale,       # (3, 128, 128) or (4, 128, 128)
    params=np.array([0.3, 0.8, 0.5, 0.3, 1.0, 1e5])  # (n_params,)
)
```

**Important**: The `params` array must have the same number of values as your `param_min`/`param_max` specification.

## Example Configurations

### Example 1: CAMELS Default (35 Parameters)

```ini
[TRAINING]
param_norm_path = /path/to/SB35_param_minmax.csv
use_param_prediction = True
param_prediction_weight = 0.01
```

### Example 2: Unconditional

```ini
[TRAINING]
n_params = 0
use_param_prediction = False
```

### Example 3: Cosmology Only (2 Parameters)

```ini
[TRAINING]
param_min = 0.1, 0.5
param_max = 0.5, 1.0
use_param_prediction = True
```

### Example 4: Full Custom (6 Parameters)

```ini
[TRAINING]
param_norm_path = configs/custom_params_example.json
use_param_prediction = True
param_prediction_weight = 0.01
```

## Adding New Parameters to Existing Dataset

If you want to add parameters to an existing dataset:

1. **Compute parameter bounds** for your new parameters:
```python
import numpy as np
import glob

all_params = []
for f in glob.glob('data/*.npz'):
    data = np.load(f)
    all_params.append(data['params'])

all_params = np.stack(all_params)
param_min = all_params.min(axis=0)
param_max = all_params.max(axis=0)
```

2. **Create JSON file** with bounds:
```python
import json

config = {
    'param_names': ['param1', 'param2', 'param3'],
    'param_min': param_min.tolist(),
    'param_max': param_max.tolist()
}

with open('custom_params.json', 'w') as f:
    json.dump(config, f, indent=2)
```

3. **Update config** to use new file:
```ini
param_norm_path = custom_params.json
```

## Parameter Prediction Head

When training with parameters, you can optionally enable a parameter prediction auxiliary task:

```ini
use_param_prediction = True
param_prediction_weight = 0.01
```

This:
- Adds a small auxiliary loss that predicts parameters from generated fields
- Can improve conditioning quality
- Provides interpretability (which parameters does the model "understand"?)

For unconditional training, this should be disabled:
```ini
use_param_prediction = False
```

## Inference with Custom Parameters

When running inference (`bind_predict.py`), provide parameters that match your training configuration:

```bash
# Default (uses params from halo catalog if available)
python bind_predict.py --checkpoint model.ckpt --input_halo_catalog halos.csv

# Override with custom parameters
python bind_predict.py \
    --checkpoint model.ckpt \
    --input_halo_catalog halos.csv \
    --cosmo_params 0.3,0.8,0.5,0.3,1.0,1e5
```

For unconditional models, parameters are ignored:
```bash
python bind_predict.py --checkpoint unconditional_model.ckpt --input_halo_catalog halos.csv
```

## Troubleshooting

### "Expected N parameters, got M"

Your training data has a different number of parameters than your config specifies. Check:
- `param_min`/`param_max` length matches `params` array in `.npz` files
- All training files have consistent `params` shape

### "Parameter norm path not found"

The CSV/JSON file doesn't exist at the specified path. Use absolute paths or paths relative to where you run the command.

### Parameters Not Affecting Generation

If parameter conditioning seems to have no effect:
1. Ensure `use_param_conditioning` is implicitly True (happens when params are provided)
2. Check if param bounds are reasonable (very narrow ranges may cause issues)
3. Try increasing `param_prediction_weight` to emphasize parameter learning

## Reference Files

- `configs/vdm_unconditional.ini` - Example unconditional config
- `configs/vdm_custom_params.ini` - Example custom parameter config
- `configs/custom_params_example.json` - Example JSON parameter specification
