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

### 3.1 Training Data Requirements

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
- `conditional_params`: 15 cosmological/astrophysical parameters
- `large_scale_dm_*`: Multi-scale context maps (optional)

### 3.2 Configuration Files

Training is controlled via INI files in `configs/`. Key configuration file: `configs/clean_vdm_aggressive_stellar.ini`

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

**Interactive (for testing):**
```bash
source /mnt/home/mlee1/venvs/torch3/bin/activate
python train_model_clean.py --config configs/clean_vdm_aggressive_stellar.ini
```

**On SLURM cluster:**
```bash
sbatch scripts/run_train.sh
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

### 5.6 Programmatic BIND Usage

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

#### `tests/test_vdm.py` - VDM Model Tests (7 tests)

| Test | Description |
|------|-------------|
| `test_unet_forward_shape` | UNet output shape matches input |
| `test_unet_deterministic` | UNet is deterministic in eval mode |
| `test_unet_with_large_scale` | UNet handles large-scale conditioning |
| `test_variance_preserving_map` | Diffusion forward process |
| `test_sample_times` | Time sampling in [0, 1] |
| `test_snr_computation` | SNR is positive for valid gamma |
| `test_fixed_linear_schedule` | Noise schedule interpolation |

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
config.py                 ~95%
vdm/networks_clean.py     ~60%
vdm/vdm_model_clean.py    ~50%
vdm/utils.py              ~40%
bind/workflow_utils.py    ~45%
bind/bind.py              ~30%
---------------------------------------
Total: 28 tests, all passing
```

---

## Project Structure

```
vdm_BIND/
├── config.py                 # Centralized path configuration
├── train_model_clean.py      # Training entry point (3-channel)
├── train_triple_model.py     # Training entry point (triple)
├── run_bind_unified.py       # BIND inference entry point
├── run_tests.py              # Test runner with validation
├── setup.py                  # Package installation
├── vdm/                      # Core VDM package
│   ├── networks_clean.py     # UNet architecture
│   ├── vdm_model_clean.py    # LightCleanVDM Lightning module
│   ├── vdm_model_triple.py   # Triple independent VDMs
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
├── tests/                    # Test suite
├── analysis/                 # Paper plots and notebooks
└── data_generation/          # Training data processing
```

---

## Branch Strategy

- **main**: Stable production code
- **astro_params**: 3-channel VDM with cosmological parameter conditioning
- **seperate_training**: Triple independent VDMs (one per output channel)

---

## Citation

If you use this code, please cite:
```
[Citation to be added upon publication]
```

## License

BSD 2-Clause License
