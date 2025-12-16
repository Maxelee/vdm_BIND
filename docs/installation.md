# Installation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

## Basic Installation

```bash
# Clone the repository
git clone https://github.com/Maxelee/vdm_BIND.git
cd vdm_BIND

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Dependencies

Core dependencies are listed in `requirements.txt`:

```
torch>=2.0.0
pytorch-lightning>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
h5py>=3.0.0
pandas>=1.3.0
scikit-image>=0.19.0
tqdm>=4.60.0
tensorboard>=2.10.0
```

### Optional Dependencies

For DDPM/NCSNpp models:
```bash
pip install score_models  # Required for ddpm model type
```

For documentation:
```bash
pip install sphinx sphinx-rtd-theme myst-parser sphinx-copybutton
```

## Environment Variables

Configure paths via environment variables (optional):

```bash
export VDM_BIND_ROOT=/path/to/vdm_BIND
export TRAIN_DATA_ROOT=/path/to/training_data
export BIND_OUTPUT_ROOT=/path/to/outputs
export CHECKPOINT_ROOT=/path/to/checkpoints
```

See `config.py` for all configurable paths.

## Verifying Installation

```python
import vdm
print(vdm.__all__)  # Should list available exports

# Test verbosity system
vdm.set_verbosity('debug')
print(f"Verbosity level: {vdm.get_verbosity()}")
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=vdm --cov=bind

# Quick validation
python run_tests.py --validate
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config or use gradient checkpointing:
```ini
[TRAINING]
batch_size = 2
accumulate_grad_batches = 4
```

### Module Not Found

Ensure the package is installed in development mode:
```bash
pip install -e .
```

### score_models Import Error

The `ddpm` model type requires the `score_models` package:
```bash
pip install git+https://github.com/yang-song/score_sde_pytorch.git
```
