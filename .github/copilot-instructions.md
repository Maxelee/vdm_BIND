# VDM-BIND Development Guidelines

## Project Architecture

This project has **two main pipelines** that share the `vdm/` core package:

### Training Pipeline (variational diffusion model)
```
train_model_clean.py → vdm/vdm_model_clean.py → vdm/networks_clean.py
train_triple_model.py → vdm/vdm_model_triple.py → vdm/networks_clean.py
```
- **Purpose**: Train diffusion models to learn DMO → Hydro mapping
- **Input**: Halo-centric cutouts (128×128) with conditional parameters
- **Output**: 3-channel predictions [DM_hydro, Gas, Stars]

### Inference Pipeline (BIND)
```
run_bind_unified.py → bind/bind.py → bind/workflow_utils.py → vdm/
```
- **Purpose**: Apply trained models to full N-body simulations
- **Key steps**: Voxelize → Extract halos → Generate → Paste back

## Critical Data Flow

### Normalization (MUST be consistent)
Training and inference MUST use identical normalization:
```python
# Located in: vdm/constants.py and workflow_utils.py
# DM condition: log10(field + 1), then Z-score normalize
# Target channels: log10(field + 1), then per-channel Z-score
# Stats loaded from: *_normalization_stats.npz files
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
- Paths should use absolute paths on the cluster

### SLURM Scripts (`scripts/*.sh`)
- All jobs use A100 GPUs with `--constraint=a100-40gb`
- PyTorch Lightning handles DDP internally - use `ntasks=1`
- Always unset `SLURM_NTASKS` to prevent Lightning conflicts

### Import Patterns
```python
# In bind/ modules, import vdm dynamically to handle different install paths:
from vdm import networks_clean as networks
from vdm import vdm_model_clean as vdm_module
from vdm.astro_dataset import get_astro_data
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `vdm/networks_clean.py` | UNet architecture with Fourier features, attention, cross-attention |
| `vdm/vdm_model_clean.py` | 3-channel VDM with focal loss, per-channel weighting |
| `vdm/vdm_model_triple.py` | 3 independent single-channel VDMs |
| `bind/bind.py` | BIND class: voxelize, extract, generate, paste |
| `bind/workflow_utils.py` | ConfigLoader, ModelManager, sample() function |
| `run_bind_unified.py` | Main CLI for BIND inference on simulation suites |

## Testing Changes

### Training Pipeline
```bash
# Quick test (CPU, small batches)
python train_model_clean.py --config configs/clean_vdm_aggressive_stellar.ini --cpu_only

# Full training
sbatch scripts/run_train.sh
```

### BIND Pipeline
```bash
# Test single simulation
python run_bind_unified.py --suite cv --sim_nums 0 --batch_size 2 --realizations 1
```

## Common Gotchas

1. **Checkpoint conditioning channels**: Auto-detected from `conv_in.weight` shape. Don't manually override unless debugging.

2. **Fourier features**: Two modes exist (`fourier_legacy=True/False`). Auto-detected from checkpoint keys.

3. **Mass conservation**: BIND normalizes generated fields so total mass matches DMO input (`conserve_mass=True`).

4. **Quantile normalization**: Optional for stellar channel - requires `quantile_path` in config.

5. **Periodic boundaries**: All halo extraction and pasting uses periodic boundary conditions.

## Simulation Suites

| Suite | Sims | Description | DMO Path Pattern |
|-------|------|-------------|------------------|
| CV | 25 | Cosmic Variance | `/mnt/ceph/.../CV/CV_{n}` |
| SB35 | 1024 | Latin Hypercube | `/mnt/ceph/.../SB35/SB35_{n}` |
| 1P | 61 | Single Parameter | `/mnt/ceph/.../1P/1P_{name}` |

## Adding New Features

1. **New loss function**: Add to `vdm/vdm_model_clean.py` in `get_diffusion_loss()`
2. **New architecture component**: Add to `vdm/networks_clean.py`
3. **New analysis**: Add notebook to `analysis/notebooks/`, use `analysis/paper_utils.py`
4. **New simulation suite**: Add `get_{suite}_simulation_info()` in `run_bind_unified.py`
