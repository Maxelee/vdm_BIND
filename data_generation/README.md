# Training Data Generation

This directory contains scripts for generating training data from paired DMO (Dark Matter Only) and Hydro simulations.

## Overview

The VDM-BIND model learns to predict baryonic fields (gas, stars) from dark matter distributions. Training requires:

1. **DMO simulation**: Provides the N-body dark matter distribution
2. **Hydro simulation**: Provides the "ground truth" baryonic fields (gas, stars, hydro-DM)
3. **Halo catalog**: FOF halos from the DMO simulation (to center cutouts)

## Data Generation Pipeline

### Input Data Structure

The pipeline expects CAMELS-style simulation organization:

```
# DMO simulations
/path/to/DMO/SB35_{sim_id}/
├── snap_090.hdf5              # Single-file snapshot (z=0)
└── (or snapdir_090/           # Multi-file snapshots)
    ├── snap_090.0.hdf5
    ├── snap_090.1.hdf5
    └── ...

# Hydro simulations  
/path/to/Hydro/SB35_{sim_id}/
└── snapdir_090/
    ├── snap_090.0.hdf5
    ├── snap_090.1.hdf5
    └── ...

# FOF catalogs (from DMO)
/path/to/FOF/SB35_{sim_id}/
└── fof_subhalo_tab_090.hdf5
```

### Output Data Structure

Each training sample is a `.npz` file containing:

```python
{
    'condition': np.ndarray,     # DM condition at 6.25 Mpc/h scale (128, 128)
    'target': np.ndarray,        # [DM_hydro, Gas, Stars] (3, 128, 128)
    'large_scale': np.ndarray,   # Multi-scale context (3, 128, 128) at [12.5, 25.0, 50.0] Mpc/h
    'params': np.ndarray,        # Cosmological + astrophysical parameters (15,)
    'halo_mass': float,          # Halo M200c mass in M_sun/h
    'halo_center': np.ndarray,   # Halo center position (3,) in Mpc/h
}
```

## Processing Steps

### 1. Load Simulation Data

```python
# Load DMO particles
dm_pos, dm_mass = load_dmo_particles(nbody_path)

# Load Hydro particles
hydro_dm_pos, hydro_dm_mass, gas_pos, gas_mass, star_pos, star_mass = load_hydro_particles(hydro_path)

# Load halo catalog
halo_pos, halo_mass = load_halos(fof_file, mass_threshold=1e13)  # M_sun/h
```

### 2. Process Each Halo with Random Rotations

For each halo:

1. **Apply periodic boundary conditions** - Center particles on halo
2. **Create periodic copies** - Fill edges for rotation
3. **Apply random 3D rotation** - Data augmentation
4. **Project to 2D** - Z-projection using CIC interpolation
5. **Extract multi-scale cutouts** - 6.25, 12.5, 25.0, 50.0 Mpc/h scales

```python
# Key function: process_halo_with_full_periodic_tiling()
mass_map, rot_matrix = process_halo_with_full_periodic_tiling(
    dm_pos, dm_mass, halo_center, 
    box_size=50.0,      # Mpc/h
    npix=1024,          # Voxel resolution for full box
    seed=unique_seed    # For reproducible rotations
)

# Extract multi-scale cutouts
multiscale = extract_multiscale_cutouts(
    mass_map, 
    box_size=50.0,
    target_resolution=128
)
# Returns shape (4, 128, 128) for scales [6.25, 12.5, 25.0, 50.0] Mpc/h
```

### 3. Multi-Scale Extraction

The multi-scale extraction creates 4 maps at different physical scales, all resampled to the same pixel resolution:

| Scale Index | Physical Size | What It Captures |
|-------------|---------------|------------------|
| 0 | 6.25 Mpc/h | Halo-scale structure (condition) |
| 1 | 12.5 Mpc/h | Local environment |
| 2 | 25.0 Mpc/h | Intermediate structure |
| 3 | 50.0 Mpc/h | Full box (cosmic web) |

**Resolution calculation:**
- Full box at 1024 pixels → 50.0 / 1024 = 0.0488 Mpc/h per pixel
- 6.25 Mpc cutout at 128 pixels → 6.25 / 128 = 0.0488 Mpc/h per pixel ✓ (same!)
- Larger scales are downsampled by averaging

### 4. Normalization

All fields are normalized identically during training AND inference:

```python
# Transform: log10(field + 1) then Z-score normalize
def normalize(field, mean, std):
    log_field = np.log10(field + 1)
    return (log_field - mean) / std

# Statistics are stored in data/*.npz files:
# - dark_matter_normalization_stats.npz
# - gas_normalization_stats.npz  
# - stellar_normalization_stats.npz
```

## Running Data Generation

### Single-node (for testing)

```bash
python process_simulations.py \
    --resolution 128 \
    --total_sims 10 \
    --start_sim 0 \
    --end_sim 10 \
    --num_rotations 10 \
    --output_base_root /path/to/output
```

### MPI Parallel (for production)

```bash
mpirun -n 64 python process_simulations.py \
    --resolution 128 \
    --total_sims 1024 \
    --num_rotations 10 \
    --hydro_base /path/to/hydro \
    --nbody_base /path/to/dmo \
    --fof_nbody_base /path/to/fof \
    --output_base_root /path/to/output
```

### SLURM (on cluster)

```bash
sbatch scripts/run_data_generation.sh
```

## Consistency with BIND Inference

**CRITICAL**: The BIND inference pipeline MUST use identical extraction methods:

| Aspect | Training (process_simulations.py) | Inference (bind.py) |
|--------|-----------------------------------|---------------------|
| CIC interpolation | `MASL.MA(..., MAS='CIC')` | `MASL.MA(..., MAS='CIC')` |
| Multi-scale sizes | [6.25, 12.5, 25.0, 50.0] Mpc/h | [6.25, 12.5, 25.0, 50.0] Mpc/h |
| Target resolution | 128 × 128 | 128 × 128 |
| Periodic boundaries | Minimum image convention | Minimum image convention |
| Normalization | log10(x+1), then Z-score | log10(x+1), then Z-score |
| Stats files | `data/*_normalization_stats.npz` | `data/*_normalization_stats.npz` |

## Verifying Consistency

Run the test suite to verify training/inference consistency:

```bash
python -m pytest tests/test_data_generation.py -v
```

Key tests:
- `test_multiscale_extraction_consistency` - Verify scale extraction matches
- `test_normalization_consistency` - Verify normalization is identical
- `test_periodic_boundary_handling` - Verify periodic boundaries match

## File Descriptions

- `process_simulations.py` - Main MPI-parallel data generation script
- `add_large_scale.py` - Utility to add large-scale context to existing data
