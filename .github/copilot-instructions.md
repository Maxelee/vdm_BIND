# VDM-BIND Development Guidelines

## Python Environment

**Always use this virtual environment for running Python commands:**
```bash
source /mnt/home/mlee1/venvs/torch3/bin/activate
```

This environment has PyTorch, Lightning, and all required dependencies installed.

## Project Architecture

This project has **two main pipelines** that share the `vdm/` core package:

### Training Pipeline (generative models)
```
train_unified.py → vdm/*_model.py → vdm/networks_clean.py
```
**Supported model types** (via `--model` flag):
- `vdm` → 3-channel VDM (LightCleanVDM)
- `triple` → 3 independent single-channel VDMs (LightTripleVDM)
- `ddpm` → DDPM/NCSNpp (score_models package)
- `dsm` → Denoising Score Matching
- `interpolant` → Flow Matching / Stochastic Interpolants
- `ot_flow` → Optimal Transport Flow Matching
- `consistency` → Consistency Models

**Purpose**: Train generative models to learn DMO → Hydro mapping
**Input**: Halo-centric cutouts (128×128) with conditional parameters
**Output**: 3-channel predictions [DM_hydro, Gas, Stars]

### Inference Pipeline (BIND)
```
run_bind_unified.py → bind/bind.py → bind/workflow_utils.py → vdm/
bind_predict.py → bind/bind.py (user-friendly CLI)
```
- **Purpose**: Apply trained models to full N-body simulations
- **Key steps**: Voxelize → Extract halos → Generate → Paste back

## Project Structure

```
vdm_BIND/
├── config.py                 # Centralized path configuration (env vars)
├── train_unified.py          # Unified training for ALL model types
├── train_model_clean.py      # Legacy VDM training (kept for reference)
├── run_bind_unified.py       # BIND inference on simulation suites
├── bind_predict.py           # User-friendly BIND CLI for custom simulations
├── vdm/                      # Core model package
│   ├── networks_clean.py     # UNet architecture
│   ├── vdm_model_clean.py    # 3-channel LightCleanVDM
│   ├── vdm_model_triple.py   # 3 independent VDMs
│   ├── ddpm_model.py         # DDPM/NCSNpp using score_models
│   ├── dsm_model.py          # Denoising Score Matching
│   ├── interpolant_model.py  # Flow matching / stochastic interpolants
│   ├── ot_flow_model.py      # Optimal transport flow matching
│   ├── consistency_model.py  # Consistency models
│   ├── astro_dataset.py      # Data loading
│   ├── io_utils.py           # I/O utilities (load_simulation, project_particles_2d)
│   └── validation_plots.py   # Validation plotting utilities
├── bind/                     # BIND inference package
│   ├── bind.py               # Main BIND class
│   ├── workflow_utils.py     # ConfigLoader, ModelManager, sample()
│   ├── analyses.py           # Evaluation utilities
│   └── power_spec.py         # Power spectrum functions
├── configs/                  # Training configurations (one per model type)
├── scripts/                  # SLURM job scripts
│   ├── run_train_unified.sh  # Array job for all models
│   ├── run_all_models.sh     # Submit all training jobs
│   └── monitor_training.sh   # Monitor running jobs
├── data/                     # Normalization stats & quantile transformer
├── tests/                    # Unit & integration tests
├── analysis/                 # Paper plots and notebooks
│   ├── paper_utils.py        # Shared analysis functions
│   └── notebooks/            # Analysis notebooks
└── data_generation/          # Training data processing
    ├── README.md             # Data generation documentation
    ├── process_simulations.py
    └── add_large_scale.py
```

## Consolidated Utility Modules

### `vdm/io_utils.py` - I/O Utilities
Centralized simulation loading and projection functions:
```python
from vdm.io_utils import load_simulation, project_particles_2d, load_halo_catalog

# Load simulation particle data
dm_pos, dm_mass, hydro_dm_pos, hydro_dm_mass, gas_pos, gas_mass, star_pos, star_mass = \
    load_simulation(nbody_path, hydro_snapdir)

# Project 3D particles to 2D grid
field_2d = project_particles_2d(positions, masses, box_size=50.0, resolution=1024, axis=2)

# Load halo catalog
halo_pos, halo_mass, halo_radii = load_halo_catalog(fof_path, mass_threshold=1e13)
```

### `bind/power_spec.py` - Power Spectrum Analysis
Consolidated power spectrum functions using Pk_library:
```python
from bind.power_spec import (
    compute_power_spectrum_simple,    # Single 2D field → (Pk, k)
    compute_power_spectrum_batch,     # Batch of fields → (k, Pk, Nmodes)
    compute_cross_power_spectrum      # Cross-correlation → (k, r, Pk)
)

# Single field power spectrum
Pk, k = compute_power_spectrum_simple(field_2d, BoxSize=50.0, MAS='CIC')

# Batch power spectrum (e.g., for multiple halos)
k, Pk_batch, Nmodes = compute_power_spectrum_batch(fields_batch, BoxSize=6.25)

# Cross-correlation
k, r, Pk_cross = compute_cross_power_spectrum(field1, field2, BoxSize=50.0)
```

## Environment Variables

All paths can be configured via environment variables (see `config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VDM_BIND_ROOT` | Auto-detected | Project root directory |
| `TRAIN_DATA_ROOT` | `/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/` | Training data |
| `BIND_OUTPUT_ROOT` | `/mnt/home/mlee1/ceph/BIND_outputs/` | BIND inference outputs |
| `CHECKPOINT_ROOT` | `/mnt/home/mlee1/ceph/tb_logs/` | Model checkpoints |
| `CAMELS_DATA_ROOT` | `/mnt/ceph/users/camels/PUBLIC_RELEASE/` | CAMELS simulations |

**Usage:**
```bash
export VDM_BIND_ROOT=/path/to/vdm_BIND
export TRAIN_DATA_ROOT=/path/to/training_data
python train_model_clean.py --config configs/clean_vdm_aggressive_stellar.ini
```

## Critical Data Flow

### Normalization (MUST be consistent)
Training and inference MUST use identical normalization:
```python
# Located in: config.py → NORMALIZATION_STATS_DIR (data/)
# DM condition: log10(field + 1), then Z-score normalize
# Target channels: log10(field + 1), then per-channel Z-score
# Stats loaded from: data/*_normalization_stats.npz files
```

### Multi-scale Conditioning
The model uses multi-scale context (6.25, 12.5, 25.0, 50.0 Mpc/h scales):
- `conditioning_channels=1`: Base DM condition only
- `large_scale_channels=3`: Additional 3 large-scale context maps
- Total input channels = 1 + large_scale_channels (auto-detected from checkpoint)

## Code Conventions

### Configuration Files (`configs/*.ini`)
- All training hyperparameters in INI format
- Key sections: `[TRAINING]` with model, data, and loss parameters
- Use relative paths (e.g., `data/quantile_normalizer_stellar.pkl`) for project files
- Use absolute paths or environment variables for external data

### SLURM Scripts (`scripts/*.sh`)
- All jobs use A100 GPUs with `--constraint=a100-40gb`
- PyTorch Lightning handles DDP internally - use `ntasks=1`
- Always unset `SLURM_NTASKS` to prevent Lightning conflicts

### Import Patterns
```python
# In bind/ modules, import vdm directly (package in same repo):
from vdm import networks_clean as networks
from vdm import vdm_model_clean as vdm_module
from vdm.astro_dataset import get_astro_data

# For paths, import from config:
from config import PROJECT_ROOT, DATA_DIR, NORMALIZATION_STATS_DIR
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `config.py` | Centralized path configuration with environment variable support |
| `train_unified.py` | **Unified training script for all model types** |
| `vdm/networks_clean.py` | UNet architecture with Fourier features, attention, cross-attention |
| `vdm/vdm_model_clean.py` | 3-channel VDM with focal loss, per-channel weighting |
| `vdm/vdm_model_triple.py` | 3 independent single-channel VDMs |
| `vdm/ddpm_model.py` | DDPM/NCSNpp wrapper using score_models package |
| `vdm/interpolant_model.py` | Flow matching / stochastic interpolants (LightInterpolant) |
| `vdm/dsm_model.py` | Denoising Score Matching |
| `vdm/ot_flow_model.py` | Optimal Transport Flow Matching |
| `vdm/consistency_model.py` | Consistency Models |
| `vdm/io_utils.py` | Consolidated I/O: `load_simulation()`, `project_particles_2d()`, `load_halo_catalog()` |
| `bind/bind.py` | BIND class: voxelize, extract, generate, paste |
| `bind/workflow_utils.py` | ConfigLoader, ModelManager, sample() function |
| `bind/power_spec.py` | Power spectrum: `compute_power_spectrum_simple/batch()`, `compute_cross_power_spectrum()` |
| `bind_predict.py` | User-friendly CLI for custom simulations (SubFind, Rockstar, CSV) |
| `run_bind_unified.py` | BIND inference on CAMELS simulation suites |
| `analysis/paper_utils.py` | Shared analysis functions: `compute_radial_profile()`, plotting utilities |

## Testing

### Running Tests
```bash
# Run all tests (142 tests)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_vdm.py -v

# Run with coverage
python -m pytest tests/ --cov=vdm --cov=bind

# Validate configuration/paths without running full tests
python run_tests.py --validate
```

### Test Structure
- `tests/test_config.py` - Path configuration and environment variables
- `tests/test_vdm.py` - UNet architecture, VDM model, noise schedules
- `tests/test_bind.py` - Normalization, ConfigLoader, halo pasting
- `tests/test_integration.py` - End-to-end pipeline tests
- `tests/test_data_generation.py` - Data generation consistency tests

### Training Pipeline
```bash
# Quick test (CPU, small batches)
python train_unified.py --model vdm --config configs/clean_vdm_aggressive_stellar.ini --cpu_only

# Train specific model type
python train_unified.py --model ddpm --config configs/ddpm.ini

# Full training on SLURM (all 8 models as array job)
sbatch scripts/run_train_unified.sh
```

### BIND Pipeline
```bash
# Test single simulation
python run_bind_unified.py --suite cv --sim_nums 0 --batch_size 2 --realizations 1
```

## Checkpoint Formats

Training scripts save checkpoints in two formats (both backward compatible):

| Format | Pattern | Created by |
|--------|---------|------------|
| **New (files)** | `epoch-epoch=XXX-step=YYY.ckpt` | `epoch_checkpoint` |
| **Legacy (dirs)** | `epoch=X-step=Y-val/metric=Z.ZZZ.ckpt` | `val_checkpoint` |
| **Latest** | `latest-epoch=X-step=Y.ckpt` | `latest_checkpoint` |

`ConfigLoader._state_initialization()` auto-detects the format and loads the appropriate checkpoint.

## Common Gotchas

1. **Checkpoint conditioning channels**: Auto-detected from `conv_in.weight` shape. Don't manually override unless debugging.

2. **Fourier features**: Two modes exist (`fourier_legacy=True/False`). Auto-detected from checkpoint keys.

3. **Mass conservation**: BIND normalizes generated fields so total mass matches DMO input (`conserve_mass=True`).

4. **Quantile normalization**: Optional for stellar channel - requires `quantile_path` in config.

5. **Periodic boundaries**: All halo extraction and pasting uses periodic boundary conditions.

6. **Environment variables**: Override paths via `VDM_BIND_ROOT`, `TRAIN_DATA_ROOT`, etc. See `config.py`.

7. **Checkpoint formats**: Two formats exist (see above). Both are auto-detected and loaded correctly.

## Simulation Suites

| Suite | Sims | Description | DMO Path Pattern |
|-------|------|-------------|------------------|
| CV | 25 | Cosmic Variance | `/mnt/ceph/.../CV/CV_{n}` |
| SB35 | 1024 | Latin Hypercube | `/mnt/ceph/.../SB35/SB35_{n}` |
| 1P | 61 | Single Parameter | `/mnt/ceph/.../1P/1P_{name}` |

## Adding New Features

1. **New model type**: Add to `vdm/` as `{model}_model.py`, update `train_unified.py` MODEL_TYPES
2. **New loss function**: Add to `vdm/vdm_model_clean.py` in `get_diffusion_loss()`
3. **New architecture component**: Add to `vdm/networks_clean.py`
4. **New analysis**: Add notebook to `analysis/notebooks/`, use `analysis/paper_utils.py`
5. **New simulation suite**: Add `get_{suite}_simulation_info()` in `run_bind_unified.py`

## Training Quick Reference

### Run Training for Any Model
```bash
# Activate environment
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Train a specific model (interactive)
python train_unified.py --model MODEL_TYPE --config configs/CONFIG.ini

# MODEL_TYPE options: vdm, triple, ddpm, dsm, interpolant, ot_flow, consistency
```

### Config Files
| Model | Config File |
|-------|-------------|
| VDM (3-channel) | `configs/clean_vdm_aggressive_stellar.ini` |
| Triple VDM | `configs/clean_vdm_triple.ini` |
| DDPM | `configs/ddpm.ini` |
| DSM | `configs/dsm.ini` |
| Interpolant | `configs/interpolant.ini` |
| Stochastic Interpolant | `configs/stochastic_interpolant.ini` |
| OT Flow | `configs/ot_flow.ini` |
| Consistency | `configs/consistency.ini` |

### SLURM Training
```bash
# Submit all 8 models as array job
sbatch scripts/run_train_unified.sh

# Monitor running jobs
./scripts/monitor_training.sh
```

