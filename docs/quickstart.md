# Quick Start Guide

This guide will help you get started with VDM-BIND in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/Maxelee/vdm_BIND.git
cd vdm_BIND

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Loading a Pre-trained Model

```python
import vdm
from bind.workflow_utils import ConfigLoader, ModelManager, sample
import torch

# Silence verbose output
vdm.set_verbosity('silent')

# Load a trained model
config = ConfigLoader('configs/interpolant.ini')
_, model = ModelManager.initialize(config, verbose=False, skip_data_loading=True)
model = model.to('cuda').eval()

print(f"Loaded model: {config.model_name}")
print(f"Checkpoint: {config.best_ckpt}")
```

## Generating Samples

```python
from vdm.astro_dataset import get_astro_data

# Load test data
test_root = '/path/to/test/data/'
datamodule = get_astro_data('IllustrisTNG', test_root, batch_size=4, stage='test')
datamodule.setup(stage='test')

# Get a batch
batch = next(iter(datamodule.test_dataloader()))
condition, large_scale, target, params = batch

# Full conditioning = DM + large-scale context
full_cond = torch.cat([condition, large_scale], dim=1)

# Generate samples
with torch.no_grad():
    samples = sample(model, full_cond, batch_size=4, n_sampling_steps=250)

print(f"Generated samples shape: {samples.shape}")
```

## Training a New Model

```bash
# Activate environment
source /path/to/venv/bin/activate

# Train with a config file
python train_unified.py --model interpolant --config configs/interpolant.ini

# Available model types: vdm, triple, ddpm, dsm, interpolant, ot_flow, consistency, dit
```

## Running BIND Inference

```bash
# Apply trained model to a full simulation
python run_bind_unified.py --suite cv --sim_nums 0 --batch_size 4 --realizations 2
```

## Verbosity Control

```python
import vdm

# Three verbosity levels
vdm.set_verbosity('silent')   # No output
vdm.set_verbosity('summary')  # Minimal output (default)
vdm.set_verbosity('debug')    # Full verbose output

# Context manager for temporary silence
with vdm.quiet():
    # All output suppressed here
    samples = sample(model, condition)
```

## Next Steps

- Read the [Model Guide](models/index.md) to understand each model type
- Learn about [Uncertainty Quantification](concepts/uncertainty.md)
- Explore the [API Reference](api/vdm.md)
