# VDM-BIND: Variational Diffusion Models for Baryonic Inference from N-body Data

A unified framework for training variational diffusion models on cosmological simulations and using them for baryonic field inference (BIND).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: BSD-2](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

## 📚 Documentation

**[Full Documentation](docs/index.md)** — Comprehensive guides, API reference, and tutorials.

Quick links:
- [Quick Start Guide](docs/quickstart.md) — Get started in 5 minutes
- [Model Guide](docs/models/index.md) — Detailed documentation for all 8 model architectures
- [API Reference](docs/api/index.md) — Complete API documentation for `vdm` and `bind` packages
- [Uncertainty & Ensembles](docs/concepts/index.md) — Uncertainty quantification and model ensembles

## Overview

VDM-BIND provides a complete pipeline for:
1. **Training** generative models to learn the mapping from dark-matter-only (DMO) simulations to hydrodynamic fields
2. **Generating** baryonic fields (dark matter, gas, stars) from DMO conditions
3. **Benchmarking** models with standardized metrics (SSIM, power spectrum, integrated mass)
4. **Quantifying uncertainty** via multi-realization sampling and ensemble methods
5. **Comparing** different model architectures (VDM, DDPM, Flow Matching, DiT, etc.)
6. **Applying BIND** to full N-body simulations

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start Tutorial](#2-quick-start-tutorial)
3. [Training a Model](#3-training-a-model)
4. [Generating Samples](#4-generating-samples)
5. [Benchmarking & Uncertainty](#5-benchmarking--uncertainty)
6. [Model Comparison](#6-model-comparison)
7. [Running BIND Inference](#7-running-bind-inference)
8. [Project Structure](#8-project-structure)
9. [Testing](#9-testing)
10. [Contributing](#10-contributing)

---

## 1. Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- Access to training data (CAMELS simulations or your own data)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Maxelee/vdm_BIND.git
cd vdm_BIND

# Create conda environment (recommended)
conda create -n vdm_bind python=3.10
conda activate vdm_bind

# Install package and dependencies
pip install -e .
```

### Additional Dependencies

```bash
# Pylians3 for power spectrum analysis
pip install Pylians

# einops for DiT models
pip install einops

# For development
pip install -e ".[dev]"
```

### Environment Variables (Optional)

Configure paths via environment variables in `~/.bashrc`:

```bash
export VDM_BIND_ROOT=/path/to/vdm_BIND
export TRAIN_DATA_ROOT=/path/to/training_data
export BIND_OUTPUT_ROOT=/path/to/outputs
export TB_LOGS_ROOT=/path/to/tensorboard_logs
```

### Verify Installation

```bash
# Quick check
python -c "from vdm import networks_clean; print('✓ VDM-BIND installed')"

# Run validation
python run_tests.py --validate

# Run full test suite
python -m pytest tests/ -v
```

---

## 2. Quick Start Tutorial

We provide an interactive tutorial notebook that walks through the entire pipeline on a small subset of data:

📓 **[Tutorial Notebook: `analysis/notebooks/00_quickstart_tutorial.ipynb`](analysis/notebooks/00_quickstart_tutorial.ipynb)**

The tutorial covers:
1. Loading and exploring training data
2. Training a model on 1000 samples
3. Generating samples from the trained model
4. Computing benchmark metrics
5. Uncertainty quantification
6. Model comparison

### Minimal Example

```python
import vdm
from vdm.vdm_model_clean import LightCleanVDM
from vdm.networks_clean import UNetVDM
from vdm.astro_dataset import get_astro_data
import torch

# Control output verbosity (optional)
vdm.set_verbosity('silent')  # or 'summary', 'debug'

# 1. Load data
train_loader, val_loader = get_astro_data(batch_size=32, num_workers=4)

# 2. Create model
score_model = UNetVDM(
    input_channels=3,
    embedding_dim=64,
    n_blocks=4,
    conditioning_channels=1,
)
model = LightCleanVDM(score_model=score_model, learning_rate=1e-4)

# 3. Train (using PyTorch Lightning)
import pytorch_lightning as pl
trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(model, train_loader, val_loader)

# 4. Generate samples
condition = next(iter(val_loader))['dm'].cuda()
params = next(iter(val_loader))['params'].cuda()
samples = model.draw_samples(condition, params, n_samples=1)
```

---

## 3. Training a Model

### 3.1 Supported Model Types

| Model | Type | Config | Description |
|-------|------|--------|-------------|
| `vdm` | Diffusion | `clean_vdm_aggressive_stellar.ini` | 3-channel Variational Diffusion Model |
| `triple` | Diffusion | `clean_vdm_triple.ini` | 3 independent single-channel VDMs |
| `ddpm` | Diffusion | `ddpm.ini` | Denoising Diffusion Probabilistic Model |
| `dsm` | Score | `dsm.ini` | Denoising Score Matching |
| `interpolant` | Flow | `interpolant.ini` | Flow Matching |
| `ot_flow` | Flow | `ot_flow.ini` | Optimal Transport Flow Matching |
| `consistency` | Consistency | `consistency.ini` | Consistency Models |
| `dit` | Transformer | `dit.ini` | Diffusion Transformer (DiT) |

### 3.2 Training Commands

```bash
# Train VDM model
python train_unified.py --model vdm --config configs/clean_vdm_aggressive_stellar.ini

# Train DiT model
python train_unified.py --model dit --config configs/dit.ini

# Train with CPU only (for testing)
python train_unified.py --model vdm --config configs/clean_vdm_aggressive_stellar.ini --cpu_only
```

### 3.3 SLURM Training

```bash
# Single model on cluster
sbatch scripts/run_train_unified.sh

# All models as array job
sbatch --array=0-8 scripts/run_train_unified.sh

# Monitor progress
./scripts/monitor_training.sh
tensorboard --logdir=/path/to/tb_logs
```

### 3.4 Key Configuration Options

Edit the INI config file to customize:

```ini
[TRAINING]
# Architecture
embedding_dim = 96
n_blocks = 5
n_attention_heads = 8

# Training
batch_size = 128
learning_rate = 5e-5
max_epochs = 250

# Diffusion
gamma_min = -13.3
gamma_max = 13.0
noise_schedule = learned_nn

# Loss weights
channel_weights = 1.0, 1.0, 3.0  # DM, Gas, Stars
```

### 3.5 Training Data Requirements

Training data should be `.npz` files with:
- `dm`: DMO density cutout (128×128)
- `dm_hydro`: Hydro DM density
- `gas`: Gas density
- `star`: Stellar density
- `conditional_params`: Cosmological parameters (optional)
- `large_scale_dm_*`: Multi-scale context (optional)

See `data_generation/README.md` for data generation instructions.

---

## 4. Generating Samples

### 4.1 From a Trained Model

```python
from bind.workflow_utils import ConfigLoader, ModelManager
import torch

# Load model
config = ConfigLoader('configs/clean_vdm_aggressive_stellar.ini')
config.best_ckpt = '/path/to/checkpoint.ckpt'
model, _ = ModelManager.initialize(config, skip_data_loading=True)
model = model.cuda().eval()

# Prepare inputs
condition = torch.randn(1, 1, 128, 128).cuda()  # DMO density
params = torch.zeros(1, 6).cuda()  # Cosmological parameters

# Generate samples
with torch.no_grad():
    # Single sample
    sample = model.draw_samples(condition, params, n_samples=1)
    
    # Multiple realizations for uncertainty
    samples = model.draw_samples(condition, params, n_samples=10)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
```

### 4.2 Using the sample() Function

```python
from bind.workflow_utils import sample

# Generate with default settings
samples = sample(
    model,
    condition,
    params,
    n_samples=5,
    batch_size=16,
)
```

### 4.3 Channel Interpretation

The model outputs 3 channels:
- **Channel 0**: Dark matter (hydro)
- **Channel 1**: Gas
- **Channel 2**: Stars

All outputs are in log10(density + 1) normalized space. Use normalization stats to convert back:

```python
import numpy as np
from config import DATA_DIR

# Load stats
stats = np.load(f'{DATA_DIR}/stellar_normalization_stats.npz')
mean, std = stats['stellar_mag_mean'], stats['stellar_mag_std']

# Denormalize
physical = samples[..., 2] * std + mean  # Stellar channel
density = 10**physical - 1
```

---

## 5. Benchmarking & Uncertainty

### 5.1 Benchmark Suite

The `BenchmarkSuite` provides standardized evaluation metrics:

```python
from vdm.benchmark import BenchmarkSuite

# Initialize benchmark
benchmark = BenchmarkSuite(
    channel_names=['DM', 'Gas', 'Stars'],
    box_size=6.25,  # Mpc/h for power spectrum
)

# Compute all metrics
metrics = benchmark.compute_all_metrics(
    predictions=predicted_samples,  # (N, 3, 128, 128)
    targets=ground_truth,           # (N, 3, 128, 128)
)

# Print summary
benchmark.print_summary(metrics)
```

**Available Metrics:**
- **Pixel metrics**: SSIM, MAE, MSE, PSNR
- **Mass metrics**: Integrated mass error, mass conservation ratio
- **Power spectrum**: Ratio at different k-scales, correlation coefficient

### 5.2 Quick Benchmark

For fast evaluation of a single sample:

```python
from vdm.benchmark import quick_benchmark

results = quick_benchmark(prediction, target)
print(f"SSIM: {results['ssim']:.4f}")
print(f"Power ratio: {results['power_ratio']:.4f}")
```

### 5.3 Uncertainty Quantification

Generate uncertainty estimates with multiple realizations:

```python
from vdm.uncertainty import UncertaintyEstimator

# Initialize estimator
estimator = UncertaintyEstimator(model, device='cuda')

# Generate samples with uncertainty
mean, std = estimator.estimate_uncertainty(
    condition=dm_condition,
    params=cosmo_params,
    n_realizations=20,
)

# Get coefficient of variation
cv = std / (mean.abs() + 1e-8)
```

### 5.4 MC Dropout Uncertainty

For models with dropout, use MC Dropout:

```python
mean, std = estimator.estimate_uncertainty(
    condition=dm_condition,
    params=cosmo_params,
    n_realizations=50,
    use_mc_dropout=True,
)
```

### 5.5 Calibration Analysis

Check if uncertainty estimates are well-calibrated:

```python
from vdm.uncertainty import compute_calibration

# predictions: (N, 3, H, W)
# uncertainties: (N, 3, H, W) 
# targets: (N, 3, H, W)
calibration = compute_calibration(
    predictions, 
    uncertainties, 
    targets,
    n_bins=10,
)

print(f"Expected coverage vs actual: {calibration['expected']} vs {calibration['observed']}")
```

### 5.6 Uncertainty Visualization

```python
from vdm.uncertainty import create_uncertainty_maps

# Create visualization
fig = create_uncertainty_maps(
    mean=mean_prediction,
    std=uncertainty,
    target=ground_truth,
    channel_names=['DM', 'Gas', 'Stars'],
)
fig.savefig('uncertainty_maps.png')
```

---

## 6. Model Comparison

### 6.1 Model Ensemble

Combine predictions from multiple models:

```python
from vdm.ensemble import ModelEnsemble, WeightedEnsemble

# Load multiple models
models = [model1, model2, model3]

# Simple averaging
ensemble = ModelEnsemble(models)
prediction = ensemble.predict(condition, params)

# Weighted ensemble (learned weights)
weighted_ensemble = WeightedEnsemble(models)
weighted_ensemble.fit(val_predictions, val_targets)  # Learn weights
prediction = weighted_ensemble.predict(condition, params)
```

### 6.2 Channel-Wise Ensemble

Use different models for different channels:

```python
from vdm.ensemble import ChannelWiseEnsemble

# Different weights per channel
ensemble = ChannelWiseEnsemble(models, n_channels=3)
ensemble.fit(val_predictions, val_targets)

# Model 1 might be better for gas, Model 2 for stars, etc.
prediction = ensemble.predict(condition, params)
```

### 6.3 Diversity-Promoting Ensemble

Select models for maximum diversity:

```python
from vdm.ensemble import DiversityEnsemble

# Select K most diverse models from pool
ensemble = DiversityEnsemble(models, n_select=3)
ensemble.select_diverse_models(val_predictions)
prediction = ensemble.predict(condition, params)
```

### 6.4 Compare Multiple Models

```python
from vdm.benchmark import BenchmarkSuite

benchmark = BenchmarkSuite()

# Compare models
results = {}
for name, model in [('VDM', vdm_model), ('DiT', dit_model), ('Flow', flow_model)]:
    predictions = model.draw_samples(condition, params, n_samples=10)
    results[name] = benchmark.compute_all_metrics(predictions, targets)

# Create comparison table
benchmark.compare_models(results)
```

---

## 7. Running BIND Inference

BIND (Baryonic Inference from N-body Data) applies trained models to full simulations.

### 7.1 User-Friendly Interface

```bash
# Apply BIND to your simulation
python bind_predict.py \
    --dmo_path /path/to/snap_090.hdf5 \
    --halo_catalog /path/to/halos.hdf5 \
    --output_dir ./my_bind_output \
    --mass_threshold 5e12 \
    --n_realizations 5
```

**Supported halo formats:** SubFind/FOF (`.hdf5`), Rockstar (`.ascii`), CSV

### 7.2 CAMELS Simulations

```bash
# Process CAMELS CV simulation
python run_bind_unified.py \
    --suite cv \
    --sim_nums 0,1,2 \
    --batch_size 16 \
    --realizations 5
```

### 7.3 Programmatic Usage

```python
from bind.bind import BIND

# Initialize pipeline
bind = BIND(
    simulation_path='/path/to/dmo_simulation',
    snapnum=90,
    boxsize=50.0,
    gridsize=1024,
    subimage_size=128,
    mass_threshold=1e13,
    config_path='configs/clean_vdm_aggressive_stellar.ini',
    output_dir='/path/to/output',
)

# Run full pipeline
bind.run_pipeline(batch_size=16, n_realizations=5)

# Or step-by-step
bind.voxelize()        # Create density grid
bind.extract_halos()   # Extract halo cutouts
bind.generate()        # Apply diffusion model
bind.paste_halos()     # Combine into final field
```

### 7.4 Output Structure

```
output_dir/
└── sim_XXX/
    ├── bind_dm.npy          # Generated DM field
    ├── bind_gas.npy         # Generated gas field
    ├── bind_star.npy        # Generated stellar field
    ├── power_spectra.npz    # Power spectrum data
    └── plots/
        └── power_spec_*.png
```

---

## 8. Project Structure

```
vdm_BIND/
├── config.py                 # Path configuration
├── train_unified.py          # Training script for all models
├── run_bind_unified.py       # BIND inference script
├── bind_predict.py           # User-friendly BIND CLI
│
├── vdm/                      # Core model package
│   ├── networks_clean.py     # UNet architecture
│   ├── vdm_model_clean.py    # VDM Lightning module
│   ├── dit.py                # Diffusion Transformer
│   ├── dit_model.py          # DiT Lightning module
│   ├── ddpm_model.py         # DDPM model
│   ├── interpolant_model.py  # Flow matching
│   ├── consistency_model.py  # Consistency models
│   ├── uncertainty.py        # Uncertainty quantification
│   ├── benchmark.py          # Benchmark suite
│   ├── ensemble.py           # Model ensembles
│   └── astro_dataset.py      # Data loading
│
├── bind/                     # BIND inference
│   ├── bind.py               # Main BIND class
│   ├── workflow_utils.py     # ConfigLoader, ModelManager
│   └── power_spec.py         # Power spectrum analysis
│
├── configs/                  # Training configurations
├── scripts/                  # SLURM scripts
├── tests/                    # Test suite (314 tests)
├── data/                     # Normalization statistics
├── analysis/                 # Analysis notebooks
│   └── notebooks/
│       ├── 00_quickstart_tutorial.ipynb  # Tutorial
│       ├── 01_bind_overview.ipynb
│       └── ...
└── data_generation/          # Training data processing
```

---

## 9. Testing

```bash
# Run all tests (314 tests)
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=vdm --cov=bind

# Run specific test file
python -m pytest tests/test_vdm.py -v

# Quick validation
python run_tests.py --validate
```

---

## 10. Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

### Development Setup

```bash
git clone https://github.com/Maxelee/vdm_BIND.git
cd vdm_BIND
pip install -e ".[dev]"
pytest tests/ -v
```

### Branch Naming

- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation

---

## Citation

If you use this code, please cite:
```
[Citation to be added upon publication]
```

## License

BSD 2-Clause License
