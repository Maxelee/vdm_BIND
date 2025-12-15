# VDM-BIND: Variational Diffusion Models for Baryonic Inference from N-body Data

A unified framework for training variational diffusion models on cosmological simulations and using them for baryonic field inference (BIND).

## Table of Contents

1. [Installation](#1-installation)
2. [Configuration Setup](#2-configuration-setup)
3. [Training the Diffusion Model](#3-training-the-diffusion-model)
4. [Loading a Trained Model](#4-loading-a-trained-model)
5. [Running BIND Inference](#5-running-bind-inference)
6. [Test Coverage](#6-test-coverage)

---

## 1. Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- Access to CAMELS simulations (for training data generation)

### Setting Up the Environment

```bash
# Clone the repository
git clone https://github.com/your-org/vdm_BIND.git
cd vdm_BIND

# Option 1: Use existing virtual environment (recommended for cluster)
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Option 2: Create new environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Required Dependencies

```
torch >= 2.0
lightning >= 2.0
numpy
h5py
pandas
scipy
matplotlib
joblib
```

### Optional Dependencies (for full functionality)

```
MAS_library (pylians3)    # For mesh assignment
Pk_library (pylians3)     # For power spectrum analysis
pytest                    # For running tests
```

### Verify Installation

```bash
# Run validation to check paths and imports
python run_tests.py --validate

# Run full test suite
python -m pytest tests/ -v
```

---

## 2. Configuration Setup

### Environment Variables

All paths can be configured via environment variables. Edit `~/.bashrc` or set them in your SLURM scripts:

```bash
# Project root (auto-detected if not set)
export VDM_BIND_ROOT=/path/to/vdm_BIND

# Training data location (halo cutouts with rotations)
export TRAIN_DATA_ROOT=/path/to/train_data_rotated2_128_cpu

# BIND output directory
export BIND_OUTPUT_ROOT=/path/to/bind_outputs

# TensorBoard logs directory
export TB_LOGS_ROOT=/path/to/tb_logs

# CAMELS simulations base directory
export CAMELS_SIMS_ROOT=/path/to/camels
```

### Configuration File (`config.py`)

The centralized path configuration is in `config.py`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | Auto-detected | Root directory of the project |
| `DATA_DIR` | `PROJECT_ROOT/data` | Normalization statistics location |
| `TRAIN_DATA_ROOT` | `/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu` | Training data |
| `BIND_OUTPUT_ROOT` | `/mnt/home/mlee1/ceph/BIND2d` | BIND inference outputs |
| `TB_LOGS_ROOT` | `/mnt/home/mlee1/ceph/tb_logs` | TensorBoard logs |
| `CAMELS_ROOT` | `/mnt/ceph/users/camels` | CAMELS simulations |

### Normalization Files

The `data/` directory contains essential normalization statistics:

```
data/
├── dark_matter_normalization_stats.npz   # DM field stats (mean, std)
├── gas_normalization_stats.npz           # Gas field stats
├── stellar_normalization_stats.npz       # Stellar field stats
└── quantile_normalizer_stellar.pkl       # Optional quantile transformer
```

These files are **required** for both training and inference to ensure consistent normalization.

---

## 3. Training the Diffusion Model

### 3.1 Training Data Generation

Training data is generated from paired DMO and hydrodynamic CAMELS simulations. The pipeline extracts halo-centric cutouts with consistent multi-scale conditioning.

**For complete documentation on training data generation, see:** `data_generation/README.md`

**Quick Overview:**
```bash
# Generate training data from CAMELS simulations using MPI
mpirun -n 64 python data_generation/process_simulations.py \
    --output_dir /path/to/train_data \
    --box_size 50.0 \
    --resolution 128
```

**Key extraction parameters:**
- **Multi-scale cutouts**: [6.25, 12.5, 25.0, 50.0] Mpc/h (must match training and inference)
- **Resolution**: 128×128 pixels per cutout
- **CIC interpolation**: Uses MAS_library for mass assignment
- **Periodic boundaries**: Full periodic handling with minimum image convention

### 3.2 Training Data Requirements

Training requires halo-centric cutouts from CAMELS simulations:

**Directory Structure:**
```
TRAIN_DATA_ROOT/
├── train/
│   ├── SB35_0_halo_0.npz
│   ├── SB35_0_halo_0_rot90.npz
│   ├── SB35_0_halo_0_rot180.npz
│   └── ...
└── test/
    ├── SB35_900_halo_0.npz
    └── ...
```

**Each `.npz` file contains:**
- `dm`: DMO density cutout (128×128)
- `dm_hydro`: Hydro DM density (128×128)  
- `gas`: Gas density (128×128)
- `star`: Stellar density (128×128)
- `conditional_params`: Cosmological/astrophysical parameters (up to 35 for CAMELS LH)
- `large_scale_dm_*`: Multi-scale context maps (optional)

**Parameter Conditioning:**
The model can be conditioned on cosmological and astrophysical parameters. CAMELS provides:
- 6 parameters for CV/1P sets: Ωm, σ8, A_SN1, A_AGN1, A_SN2, A_AGN2
- Up to 35 parameters for SB35 Latin Hypercube set

Set `n_params` in your config to match your data (or 0 for unconditional generation).

### 3.2 Configuration Files

Training is controlled via INI files in `configs/`. Each model type has its own configuration file:

| Model Type | Config File | Description |
|------------|-------------|-------------|
| VDM | `clean_vdm_aggressive_stellar.ini` | 3-channel Variational Diffusion Model |
| Triple VDM | `clean_vdm_triple.ini` | 3 independent single-channel VDMs |
| DDPM | `ddpm.ini` | Denoising Diffusion Probabilistic Model |
| DSM | `dsm.ini` | Denoising Score Matching |
| Interpolant | `interpolant.ini` | Flow Matching |
| Stochastic Interpolant | `stochastic_interpolant.ini` | Stochastic Flow Matching |
| OT Flow | `ot_flow.ini` | Optimal Transport Flow Matching |
| Consistency | `consistency.ini` | Consistency Models |

#### Training Hyperparameters

| Parameter | Example | Description |
|-----------|---------|-------------|
| `seed` | 8 | Random seed for reproducibility |
| `batch_size` | 128 | Training batch size |
| `max_epochs` | 250 | Maximum training epochs |
| `learning_rate` | 5e-5 | Initial learning rate |
| `lr_scheduler` | `cosine_warmup` | LR schedule: `cosine`, `cosine_warmup`, `reduce_on_plateau` |
| `num_workers` | 40 | DataLoader workers |

#### Model Architecture

| Parameter | Example | Description |
|-----------|---------|-------------|
| `embedding_dim` | 96 | Base channel dimension (doubles each encoder block) |
| `n_blocks` | 5 | Number of UNet encoder/decoder blocks |
| `n_attention_heads` | 8 | Self-attention heads (must divide bottleneck channels) |
| `norm_groups` | 8 | GroupNorm groups |
| `add_attention` | True | Enable self-attention at bottleneck |
| `use_fourier_features` | True | Enable Fourier positional encoding |
| `legacy_fourier` | False | Use legacy (True) or multi-scale (False) Fourier |
| `large_scale_channels` | 3 | Number of multi-scale context maps |

#### Diffusion Settings

| Parameter | Example | Description |
|-----------|---------|-------------|
| `noise_schedule` | `learned_nn` | Schedule type: `fixed_linear`, `learned_linear`, `learned_nn` |
| `gamma_min` | -13.3 | Minimum log-SNR (high noise) |
| `gamma_max` | 13.0 | Maximum log-SNR (low noise) |
| `data_noise` | `5e-4, 5e-4, 5e-4` | Per-channel data noise |
| `antithetic_time_sampling` | True | Use antithetic sampling for variance reduction |

#### Loss Configuration

| Parameter | Example | Description |
|-----------|---------|-------------|
| `channel_weights` | `1.0, 1.0, 3.0` | Per-channel loss weights [DM, Gas, Stars] |
| `use_focal_loss` | False | Enable focal loss for hard examples |
| `focal_gamma` | 3.0 | Focal loss focusing parameter |
| `lambdas` | `1.0, 1.0, 1.0` | VDM loss term weights (diffusion, latent, recons) |

#### Parameter Conditioning

| Parameter | Example | Description |
|-----------|---------|-------------|
| `use_param_prediction` | True | Enable auxiliary parameter prediction |
| `param_prediction_weight` | 0.01 | Weight for parameter prediction loss |
| `param_norm_path` | `path/to/SB35_param_minmax.csv` | Parameter normalization file |

#### Output Paths

| Parameter | Example | Description |
|-----------|---------|-------------|
| `model_name` | `clean_vdm_aggressive_stellar` | Name for logging |
| `tb_logs` | `/mnt/home/mlee1/ceph/tb_logs` | TensorBoard directory |
| `version` | 3 | Experiment version number |

### 3.3 Running Training

All model types are trained using the unified training script `train_unified.py`.

**Interactive Training (single model):**
```bash
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Train VDM
python train_unified.py --model vdm --config configs/clean_vdm_aggressive_stellar.ini

# Train DDPM
python train_unified.py --model ddpm --config configs/ddpm.ini

# Train with CPU only (for testing)
python train_unified.py --model vdm --config configs/clean_vdm_aggressive_stellar.ini --cpu_only
```

**Available model types:** `vdm`, `triple`, `ddpm`, `dsm`, `interpolant`, `ot_flow`, `consistency`

**On SLURM cluster (single model):**
```bash
# Edit scripts/run_train_unified.sh to set MODEL and CONFIG
sbatch scripts/run_train_unified.sh
```

**Train all models as SLURM array job:**
```bash
# Submits all 8 model types as an array job
sbatch scripts/run_train_unified.sh
```

**Monitor training progress:**
```bash
# Check job status and training metrics
./scripts/monitor_training.sh

# View TensorBoard logs
tensorboard --logdir=/mnt/home/mlee1/ceph/tb_logs3/
```

### 3.4 Training Outputs

Training creates the following directory structure:

```
TB_LOGS_ROOT/
└── model_name/
    └── version_X/
        ├── hparams.yaml              # Hyperparameters
        ├── events.out.tfevents.*     # TensorBoard logs
        └── checkpoints/
            ├── epoch=X-step=Y-val/elbo=Z.ckpt   # Best checkpoint
            └── latest-epoch=X-step=Y.ckpt       # Periodic checkpoint
```

**Checkpoint contents:**
- Model state dict (score_model weights)
- Optimizer state
- LR scheduler state
- Training step/epoch
- Hyperparameters

**Monitoring training:**
```bash
tensorboard --logdir=/mnt/home/mlee1/ceph/tb_logs/model_name
```

---

## 4. Loading a Trained Model

### 4.1 Using ModelManager (Recommended)

```python
from bind.workflow_utils import ConfigLoader, ModelManager

# Load configuration
config = ConfigLoader('configs/clean_vdm_aggressive_stellar.ini')

# Set checkpoint path
config.best_ckpt = '/path/to/checkpoint.ckpt'

# Initialize model (auto-detects architecture from checkpoint)
model, dataloader = ModelManager.initialize(
    config, 
    verbose=True,
    skip_data_loading=True  # Skip data loading for inference
)

# Move to GPU
model = model.cuda()
model.eval()
```

### 4.2 Direct Loading

```python
import torch
from vdm.networks_clean import UNetVDM
from vdm.vdm_model_clean import LightCleanVDM

# Create score model with same architecture as training
score_model = UNetVDM(
    input_channels=3,
    gamma_min=-13.3,
    gamma_max=13.0,
    embedding_dim=96,
    n_blocks=5,
    n_attention_heads=8,
    norm_groups=8,
    add_attention=True,
    use_fourier_features=True,
    conditioning_channels=1,
    large_scale_channels=3,
)

# Create Lightning model
light_vdm = LightCleanVDM(
    score_model=score_model,
    learning_rate=5e-5,
    gamma_min=-13.3,
    gamma_max=13.0,
    image_shape=(3, 128, 128),
    noise_schedule="learned_nn",
)

# Load checkpoint
checkpoint = torch.load('checkpoint.ckpt', map_location='cuda')
light_vdm.load_state_dict(checkpoint['state_dict'])
light_vdm.eval()
```

### 4.3 Auto-Detection Features

When loading from checkpoint, `ModelManager` automatically detects:

- **Fourier features mode**: Legacy vs multi-scale (from checkpoint keys)
- **Conditioning channels**: From `conv_in.weight` shape
- **Large-scale channels**: Computed from total input channels
- **Cross-attention**: From presence of cross-attention keys

---

## 5. Running BIND Inference

### 5.1 Required Parameters

To run BIND on a simulation, you need:

1. **Trained model checkpoint** (`--checkpoint` or set in config)
2. **Simulation suite** (`--suite`: cv, sb35, or 1p)
3. **Simulation number(s)** (`--sim_nums`: comma-separated list)
4. **Output directory** (set via `BIND_OUTPUT_ROOT` or `--output_dir`)

### 5.2 Basic Usage

```bash
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Process single CV simulation
python run_bind_unified.py \
    --suite cv \
    --sim_nums 0 \
    --checkpoint /path/to/checkpoint.ckpt \
    --batch_size 16 \
    --realizations 1

# Process multiple simulations
python run_bind_unified.py \
    --suite sb35 \
    --sim_nums 900,901,902 \
    --realizations 5
```

### 5.3 Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--suite` | Required | Simulation suite: `cv`, `sb35`, `1p` |
| `--sim_nums` | Required | Comma-separated simulation numbers |
| `--checkpoint` | Auto | Path to model checkpoint |
| `--config` | Auto | Path to config INI file |
| `--output_dir` | `BIND_OUTPUT_ROOT` | Output directory |
| `--batch_size` | 16 | Inference batch size |
| `--realizations` | 1 | Number of stochastic realizations |
| `--gridsize` | 1024 | Output grid resolution |
| `--mass_threshold` | 1e13 | Minimum halo mass for extraction |
| `--dim` | `2d` | Dimension mode: `2d` or `3d` |
| `--axis` | 2 | Projection axis for 2D (0=x, 1=y, 2=z) |

### 5.4 SLURM Array Jobs

For processing many simulations:

```bash
# Process CV suite (25 simulations)
sbatch --array=0-24 scripts/run_bind_array.sh

# Process SB35 test set with max 10 concurrent
sbatch --array=0-101%10 --export=ALL,SUITE="sb35" scripts/run_bind_array.sh
```

### 5.5 Output Structure

```
BIND_OUTPUT_ROOT/
└── suite_name/
    └── sim_XXX/
        ├── bind_dm.npy           # Generated DM field
        ├── bind_gas.npy          # Generated gas field
        ├── bind_star.npy         # Generated stellar field
        ├── true_dm.npy           # Ground truth hydro DM
        ├── true_gas.npy          # Ground truth gas
        ├── true_star.npy         # Ground truth stellar
        ├── dmo_dm.npy            # Input DMO field
        ├── power_spectra.npz     # Power spectrum data
        └── plots/
            ├── power_spec_dm.png
            ├── power_spec_gas.png
            └── power_spec_star.png
```

### 5.6 User-Friendly BIND Prediction (bind_predict.py)

For users who want to apply BIND to their own DMO simulations, we provide a simplified command-line interface:

```bash
# Basic usage - specify your simulation, halo catalog, and output directory
python bind_predict.py \
    --dmo_path /path/to/your/snap_090.hdf5 \
    --halo_catalog /path/to/your/fof_halos.hdf5 \
    --output_dir ./my_bind_output

# With custom parameters
python bind_predict.py \
    --dmo_path /path/to/snap_090.hdf5 \
    --halo_catalog /path/to/halos.hdf5 \
    --output_dir ./output \
    --mass_threshold 5e12 \
    --n_realizations 5 \
    --device cuda

# Using Rockstar halo catalog format
python bind_predict.py \
    --dmo_path /path/to/snap_090.hdf5 \
    --halo_catalog /path/to/halos_0.0.ascii \
    --halo_format rockstar \
    --output_dir ./output

# Using CSV halo catalog (columns: x, y, z, mass)
python bind_predict.py \
    --dmo_path /path/to/snap_090.hdf5 \
    --halo_catalog /path/to/halos.csv \
    --halo_format csv \
    --output_dir ./output
```

**Supported Halo Catalog Formats:**

| Format | Extension | Description |
|--------|-----------|-------------|
| SubFind/FOF | `.hdf5`, `.h5` | AREPO/Gadget format with `Group/` or `Subhalo/` groups |
| Rockstar | `.ascii`, `.list`, `.txt` | ASCII format with standard Rockstar columns |
| CSV | `.csv` | Generic CSV with columns: `x, y, z, mass` (optionally `radius`) |

**Output:**

For each halo and realization, `bind_predict.py` generates:
- `halo_{idx}_real_{n}.npz` containing:
  - `dm_hydro`: Dark matter density from hydro simulation approximation
  - `gas`: Gas density field
  - `stars`: Stellar density field
  - `condition`: Input DMO condition map
  - `halo_position`, `halo_mass`: Halo metadata
- `generation_summary.csv`: Summary table of all generated samples

### 5.7 Programmatic BIND Usage

```python
from bind.bind import BIND

# Initialize BIND pipeline
bind = BIND(
    simulation_path='/path/to/dmo_simulation',
    snapnum=90,
    boxsize=50.0,
    gridsize=1024,
    subimage_size=128,
    mass_threshold=1e13,
    config_path='configs/clean_vdm_aggressive_stellar.ini',
    output_dir='/path/to/output',
    device='cuda',
    dim='2d',
    axis=2,
)

# Run full pipeline
bind.run_pipeline(
    batch_size=16,
    n_realizations=5,
)

# Or run step-by-step
bind.voxelize()           # Create density grid from particles
bind.extract_halos()      # Extract halo-centric cutouts
bind.generate()           # Apply diffusion model
bind.paste_halos()        # Combine into final field
```

---

## 6. Test Coverage

The test suite (`tests/`) provides comprehensive coverage of core functionality.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=vdm --cov=bind --cov-report=html

# Run specific test file
python -m pytest tests/test_vdm.py -v

# Quick validation (no tests, just path checks)
python run_tests.py --validate
```

### Test Structure

#### `tests/test_config.py` - Configuration Tests (10 tests)

| Test | Description |
|------|-------------|
| `test_project_root_exists` | Verify PROJECT_ROOT is valid directory |
| `test_data_dir_exists` | Verify DATA_DIR is valid directory |
| `test_normalization_stats_exist` | All normalization .npz files exist |
| `test_validate_paths_required` | `validate_paths()` function works |
| `test_get_config_paths` | `get_config_paths()` returns correct dict |
| `test_dm_stats_content` | DM stats contain dm_mag_mean, dm_mag_std |
| `test_gas_stats_content` | Gas stats contain correct keys |
| `test_stellar_stats_content` | Stellar stats contain correct keys |
| `test_train_data_root_override` | Environment variable override works |

#### `tests/test_data_generation.py` - Data Generation Consistency Tests (15 tests)

| Test | Description |
|------|-------------|
| `test_scale_sizes_match` | Training and BIND use same physical scales |
| `test_target_resolution_match` | Consistent 128x128 resolution |
| `test_voxel_resolution_calculation` | Grid resolution math is correct |
| `test_normalization_stats_exist` | Normalization files exist |
| `test_normalization_transform` | log10(x+1) transform is correct |
| `test_zscore_normalization` | Z-score normalization works |
| `test_normalization_roundtrip` | Normalize/denormalize is reversible |
| `test_minimum_image_convention` | Periodic boundary handling |
| `test_cic_mass_conservation` | CIC interpolation conserves mass |

#### `tests/test_vdm.py` - UNet Architecture Tests (30 tests)

Tests cover UNet forward pass, conditioning, attention mechanisms, Fourier features, and cross-attention.

#### `tests/test_vdm_model.py` - VDM Model Tests (56 tests)

Comprehensive tests for noise schedules, diffusion loss, focal loss, per-channel noise, and Lightning module integration.

#### `tests/test_bind.py` - BIND Pipeline Tests (7 tests)

| Test | Description |
|------|-------------|
| `test_load_normalization_stats` | Load stats from data/ directory |
| `test_normalization_values_reasonable` | Stats values are physically reasonable |
| `test_config_loader_basic` | ConfigLoader parses INI files |
| `test_config_loader_channel_weights` | Channel weights parsed correctly |
| `test_weight_function_2d` | 2D halo pasting weight function |
| `test_weight_function_3d` | 3D halo pasting weight function |
| `test_extract_region_periodic` | Periodic boundary extraction |

#### `tests/test_integration.py` - End-to-End Tests (5 tests)

| Test | Description |
|------|-------------|
| `test_model_forward_pass` | Complete forward pass through model |
| `test_loss_computation` | Loss computation with Lightning model |
| `test_draw_samples_shape` | Sampling produces correct output shape |
| `test_normalization_roundtrip` | Normalize/denormalize consistency |
| `test_load_single_sample` | Load real training sample from disk |

### Test Coverage Summary

```
Module                    Coverage
---------------------------------------
config.py                 ~83%
vdm/networks_clean.py     ~75%
vdm/vdm_model_clean.py    ~86%
vdm/utils.py              ~77%
bind/workflow_utils.py    ~45%
bind/bind.py              ~30%
---------------------------------------
Total: 142 tests, all passing
```

### Test Categories

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_config.py` | 10 | Path configuration, environment variables |
| `test_vdm.py` | 30 | UNet architecture, forward pass, attention |
| `test_vdm_model.py` | 56 | VDM loss functions, noise schedules, Lightning module |
| `test_bind.py` | 13 | Normalization, ConfigLoader, halo pasting |
| `test_data_generation.py` | 15 | Multi-scale extraction, CIC interpolation, consistency |
| `test_integration.py` | 5 | End-to-end pipeline tests |
| `test_utils.py` | 13 | Utility functions |

---

## Project Structure

```
vdm_BIND/
├── config.py                 # Centralized path configuration
├── train_unified.py          # Unified training for ALL model types
├── train_model_clean.py      # Legacy VDM training (kept for reference)
├── run_bind_unified.py       # BIND inference entry point
├── run_tests.py              # Test runner with validation
├── setup.py                  # Package installation
├── vdm/                      # Core model package
│   ├── networks_clean.py     # UNet architecture
│   ├── vdm_model_clean.py    # LightCleanVDM Lightning module
│   ├── vdm_model_triple.py   # Triple independent VDMs
│   ├── ddpm_model.py         # DDPM/NCSNpp model
│   ├── dsm_model.py          # Denoising Score Matching
│   ├── interpolant_model.py  # Flow Matching
│   ├── ot_flow_model.py      # OT Flow Matching
│   ├── consistency_model.py  # Consistency Models
│   ├── astro_dataset.py      # PyTorch dataset
│   ├── augmentation.py       # Data augmentation
│   ├── callbacks.py          # Training callbacks
│   ├── constants.py          # Normalization constants
│   ├── metrics.py            # Evaluation metrics
│   ├── utils.py              # Utility functions
│   └── validation_plots.py   # Visualization
├── bind/                     # BIND inference package
│   ├── bind.py               # Main BIND class
│   ├── workflow_utils.py     # ConfigLoader, ModelManager
│   ├── power_spec.py         # Power spectrum analysis
│   └── analyses.py           # Evaluation utilities
├── configs/                  # Configuration files
├── scripts/                  # SLURM job scripts
├── data/                     # Normalization statistics
├── docs/                     # Additional documentation
├── tests/                    # Test suite
├── analysis/                 # Paper plots and notebooks
└── data_generation/          # Training data processing
```

---

## 8. Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/Maxelee/vdm_BIND.git
cd vdm_BIND
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Branch Naming Convention

- `feature/*` - New features
- `fix/*` - Bug fixes  
- `docs/*` - Documentation updates
- `test/*` - Testing improvements

---

## Citation

If you use this code, please cite:
```
[Citation to be added upon publication]
```

## License

BSD 2-Clause License
