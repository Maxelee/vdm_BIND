# VDM-BIND: Variational Diffusion Models for Baryonic Inference from N-body Data

A unified framework for training variational diffusion models on cosmological simulations and using them for baryonic field inference (BIND).

## Project Overview

This project combines:
1. **Training Pipeline** (`vdm/`): Train diffusion models to learn the mapping from dark matter-only (DMO) simulations to full hydrodynamic simulations
2. **Inference Pipeline** (`bind/`): Use trained models to "paint" baryonic physics onto N-body simulations
3. **Data Generation** (`data_generation/`): Process raw simulations into training datasets
4. **Analysis** (`analysis/`): Notebooks and tools for evaluating model performance

## Quick Start

### Training a Model
```bash
# Single 3-channel model (DM, Gas, Stars together)
sbatch scripts/run_train.sh

# Triple model (3 independent single-channel models)
sbatch scripts/run_train_triple.sh
```

### Running BIND Inference
```bash
# Process a single simulation
sbatch --array=0 scripts/run_bind_array.sh

# Process CV suite (25 simulations)
sbatch --array=0-24 scripts/run_bind_array.sh

# Process SB35 test set with max 10 concurrent jobs
sbatch --array=0-101%10 --export=ALL,SUITE="sb35" scripts/run_bind_array.sh
```

### Generating Training Data
```bash
# MPI-parallel training data generation
sbatch scripts/run_data_generation.sh
```

## Directory Structure

```
vdm_BIND/
├── vdm/                     # Core diffusion model package
│   ├── __init__.py
│   ├── networks_clean.py    # UNet architecture with Fourier features
│   ├── vdm_model_clean.py   # Single 3-channel VDM (LightCleanVDM)
│   ├── vdm_model_triple.py  # Triple independent model training
│   ├── astro_dataset.py     # PyTorch dataset for training data
│   ├── augmentation.py      # Data augmentation transforms
│   ├── callbacks.py         # Training callbacks (FID, gradients, etc.)
│   ├── constants.py         # Normalization constants
│   ├── metrics.py           # Evaluation metrics
│   ├── utils.py             # Utility functions
│   └── validation_plots.py  # Visualization utilities
├── bind/                    # BIND inference package
│   ├── __init__.py
│   ├── bind.py              # Main BIND class
│   ├── workflow_utils.py    # Model loading, sampling utilities
│   ├── power_spec.py        # Power spectrum analysis
│   └── analyses.py          # Comprehensive analysis functions
├── data_generation/         # Training data creation
│   ├── process_simulations.py    # Main data processing script
│   └── add_large_scale.py        # Large-scale conditioning extraction
├── configs/                 # Configuration files
│   ├── clean_vdm_aggressive_stellar.ini  # Standard 3-channel config
│   └── clean_vdm_triple.ini              # Triple model config
├── scripts/                 # SLURM job scripts
│   ├── run_train.sh         # Train 3-channel model
│   ├── run_train_triple.sh  # Train triple model
│   ├── run_bind_array.sh    # BIND inference array job
│   └── run_data_generation.sh  # Data generation job
├── analysis/                # Analysis notebooks
│   ├── notebooks/           # Jupyter notebooks for paper figures
│   └── paper_utils.py       # Shared plotting utilities
├── setup.py                 # Package installation
└── README.md                # This file
```

## Model Architecture

### Single 3-Channel Model (`astro_params` branch style)
- **Input**: DMO density field + (optional) large-scale context
- **Output**: 3 channels [DM_hydro, Gas, Stars]
- **Conditioning**: Cosmological + astrophysical parameters (15 params)
- **Features**: Fourier position encoding, multi-scale conditioning

### Triple Independent Models (`seperate_training` branch style)
- Three completely independent single-channel VDMs
- Each model has its own optimizer and gradients
- Enables per-channel hyperparameter tuning (e.g., focal loss for stars)

## Data Requirements

Training data is generated from CAMELS simulations:
- **DMO simulations**: Dark matter-only N-body runs
- **Hydro simulations**: Full hydrodynamic simulations (IllustrisTNG physics)
- **Halo catalogs**: FOF/Subfind catalogs for halo-centric extraction

### Training Data Format
Each training sample contains:
- `dm`: DMO density cutout (128×128)
- `dm_hydro`: Hydro DM density (128×128)
- `gas`: Gas density (128×128)
- `star`: Stellar density (128×128)
- `conditional_params`: Array of 15 cosmological/astrophysical parameters

## Key Dependencies

```
torch >= 2.0
lightning >= 2.0
numpy
h5py
pandas
MAS_library (pylians3)
Pk_library (pylians3)
scipy
matplotlib
```

## Configuration

Model training is controlled via INI config files. Key parameters:

```ini
[TRAINING]
# Model architecture
embedding_dim = 96
n_blocks = 5
n_attention_heads = 8

# Diffusion process
gamma_min = -10.0
gamma_max = 10.0
noise_schedule = fixed_linear

# Data
cropsize = 128
batch_size = 64
large_scale_channels = 3  # Number of multi-scale context maps

# Loss
channel_weights = 1.0,1.0,3.0  # Weight stars 3x
use_focal_loss = True
focal_gamma = 3.0
```

## Citation

If you use this code, please cite:
```
[Citation to be added upon publication]
```

## License

BSD 2-Clause License
