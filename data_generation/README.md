# Training Data Generation

This directory contains the exact scripts used to generate training data from paired DMO (Dark Matter Only) and Hydro simulations for the BIND model.

> **Source of truth**: These scripts are copied from `/mnt/home/mlee1/make_train_data/` where the production data generation was run. The canonical scripts are `process_simulations2_cpu.py` (renamed here to `process_simulations.py`) and `process_simulations2_cpu_lowmass.py` (renamed to `process_simulations_lowmass.py`).

## Overview

The VDM-BIND model learns to predict baryonic fields (gas, stars) from dark matter distributions. Training requires:

1. **DMO simulation**: Provides the N-body dark matter distribution
2. **Hydro simulation**: Provides the "ground truth" baryonic fields (gas, stars, hydro-DM)
3. **Halo catalog**: FOF halos from the DMO simulation (to center cutouts)

## Two-Stage Data Generation

Training data is generated in two stages with different halo mass thresholds:

| Stage | Script | Mass Range | Rotations | Halos/Sim | SLURM Script |
|-------|--------|-----------|-----------|-----------|--------------|
| **1. High-mass** | `process_simulations.py` | M > 10^13 M☉ | 10 | ~3-10 | `run_mpi_cpu.sh` |
| **2. Low-mass** | `process_simulations_lowmass.py` | 10^12 < M ≤ 10^13 M☉ | 1 | ~50-200 | `run_mpi_cpu_lowmass.sh` |

Both stages save into the **same output directory** (`train_data_rotated2_128_cpu/train/` and `test/`), so the training pipeline picks up all `.npz` files seamlessly.

### File naming convention

- High-mass halos: `sim_{sim_id}_halo_{halo_idx}_rot_{rot_idx}.npz`
- Low-mass halos: `sim_{sim_id}_lowmass_halo_{halo_idx}_rot_{rot_idx}.npz`

## Input Data Structure

The pipeline expects CAMELS-style simulation organization:

```
# DMO simulations
/path/to/DMO/SB35_{sim_id}/
└── snap_090.hdf5              # Single-file snapshot (z=0)

# Hydro simulations  
/path/to/Hydro/SB35_{sim_id}/
└── snapdir_090/
    ├── snap_090.0.hdf5        # Multi-file snapshots (16 chunks)
    ├── snap_090.1.hdf5
    └── ...

# FOF catalogs (from DMO)
/path/to/FOF/SB35_{sim_id}/
└── fof_subhalo_tab_090.hdf5
```

Default paths (configurable via CLI args):
- Hydro: `/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/`
- DMO: `/mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35/`
- FOF: `/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/SB35/`
- Params: `/mnt/home/mlee1/50Mpc_boxes/data/param_df.csv`

## Output Data Structure

Each training sample is a `.npz` file containing:

```python
{
    'condition': np.ndarray,     # DM condition at 6.25 Mpc/h scale (128, 128)
    'target': np.ndarray,        # [DM_hydro, Gas, Stars] (3, 128, 128)
    'large_scale': np.ndarray,   # Multi-scale context (3, 128, 128) at [12.5, 25.0, 50.0] Mpc/h
    'params': np.ndarray,        # Cosmological + astrophysical parameters (35,)
    'halo_mass': float,          # Halo M200c mass in M_sun
    'halo_center': np.ndarray,   # Halo center position (3,) in Mpc/h
}
```

Output directory structure:
```
train_data_rotated2_128_cpu/
├── train/
│   ├── sim_0/
│   │   ├── sim_0_halo_0_rot_0.npz          # High-mass, rotation 0
│   │   ├── sim_0_halo_0_rot_1.npz          # High-mass, rotation 1
│   │   ├── ...
│   │   ├── sim_0_halo_0_rot_9.npz          # High-mass, rotation 9
│   │   ├── sim_0_lowmass_halo_0_rot_0.npz  # Low-mass, 1 rotation
│   │   ├── sim_0_lowmass_halo_1_rot_0.npz
│   │   └── ...
│   ├── sim_1/
│   └── ...
└── test/
    ├── sim_42/
    └── ...
```

## Processing Steps

### 1. Load Simulation Data

```python
dm_pos, dm_mass, hydro_dm_pos, hydro_dm_mass, gas_pos, gas_mass, star_pos, star_mass = \
    load_simulation(nbody_path, hydro_snapdir)
```

Unit conversions applied:
- Positions: kpc/h → Mpc/h (divide by 1000)
- Masses: 10^10 M☉ → M☉ (multiply by 1e10)

### 2. Load Halo Catalog

```python
# High-mass (Stage 1)
halo_pos, halo_mass = load_halos(fof_file, mass_threshold=1e13)

# Low-mass (Stage 2)
halo_pos, halo_mass, indices = load_halos_in_mass_range(fof_file, mass_low=1e12, mass_high=1e13)
```

### 3. Process Each Halo with Random Rotations

For each halo, the processing pipeline:

1. **Center on halo** — Apply periodic minimum image convention
2. **Create periodic copies** — Buffer of 10 Mpc/h near edges to ensure full coverage after rotation
3. **Apply random 3D rotation** — Euler angles uniformly sampled from [0, 2π)
4. **Shift to box frame** — Place halo at box center
5. **Project to 2D** — Z-axis projection using CIC interpolation at 1024² resolution
6. **Extract multi-scale cutouts** — 4 scales, all resampled to 128×128

```python
mass_map, rot_matrix = process_halo_with_full_periodic_tiling(
    dm_pos, dm_mass, halo_center, 
    box_size=50.0,      # Mpc/h
    npix=1024,          # Full-box voxel resolution
    seed=unique_seed    # Reproducible rotations
)

multiscale = extract_multiscale_cutouts(mass_map, box_size=50.0, target_resolution=128)
# Returns shape (4, 128, 128) for scales [6.25, 12.5, 25.0, 50.0] Mpc/h
```

**Seed strategy:**
- High-mass: `seed = sim_id * 1000 + halo_idx * 100 + rotation_idx`
- Low-mass: `seed = 50000 + sim_id * 1000 + halo_idx * 100 + rotation_idx` (offset to avoid collision)

### 4. Multi-Scale Extraction

| Scale Index | Physical Size | What It Captures | Saved As |
|-------------|---------------|------------------|----------|
| 0 | 6.25 Mpc/h | Halo-scale structure | `condition` |
| 1 | 12.5 Mpc/h | Local environment | `large_scale[0]` |
| 2 | 25.0 Mpc/h | Intermediate structure | `large_scale[1]` |
| 3 | 50.0 Mpc/h | Full box (cosmic web) | `large_scale[2]` |

Resolution calculation:
- Full box at 1024 pixels → 50.0 / 1024 = 0.0488 Mpc/h per pixel
- 6.25 Mpc cutout at 128 pixels → 6.25 / 128 = 0.0488 Mpc/h per pixel (same native resolution)
- Larger scales are downsampled by block-averaging

### 5. Train/Test Split

- **Seed**: 1993 (fixed for reproducibility)
- **Test fraction**: 10% (102 out of 1024 simulations)
- Split is simulation-level (all halos from a sim go to the same set)
- Both high-mass and low-mass use the **same seed**, producing identical splits

## Running Data Generation

### Stage 1: High-Mass Halos (M > 10^13 M☉)

```bash
# SLURM submission
sbatch data_generation/run_mpi_cpu.sh

# Or manually with MPI (128 ranks across 8 nodes)
srun -n 128 python3 -u data_generation/process_simulations.py \
    --start_sim 0 --end_sim 1024 --num_rotations 10
```

**Resources**: 8 nodes × 16 cores, ~7 days, ~50 GB output

### Stage 2: Low-Mass Halos (10^12 < M ≤ 10^13 M☉)

```bash
# SLURM array job (103 tasks × 10 sims each)
sbatch data_generation/run_mpi_cpu_lowmass.sh

# Or manually
srun -n 16 python3 -u data_generation/process_simulations_lowmass.py \
    --start_sim 0 --end_sim 1024 --num_rotations 1
```

**Resources**: 1 node × 16 cores per array task, 103 array tasks, ~1 day each, ~150 GB output

### Adding Large-Scale Context (Legacy)

The `add_large_scale.py` script was used to add multi-scale context to an older data format. It is **not needed** for the current pipeline — `process_simulations.py` already generates multi-scale cutouts inline via `extract_multiscale_cutouts()`.

```bash
# Legacy only - not needed for current pipeline
mpirun -n 16 python3 add_large_scale.py \
    --data_dir /path/to/train_2d \
    --projected_images_dir /path/to/projected_images
```

## Resume Capability

Both scripts support resuming interrupted runs:
- On startup, rank 0 scans for incomplete simulations
- Only incomplete simulations are distributed to MPI ranks
- Per-halo resume: completed halos (all rotations present) are skipped

## Consistency with BIND Inference

**CRITICAL**: The BIND inference pipeline MUST use identical methods:

| Aspect | Training (process_simulations.py) | Inference (bind.py) |
|--------|-----------------------------------|---------------------|
| CIC interpolation | `MASL.MA(..., MAS='CIC')` | `MASL.MA(..., MAS='CIC')` |
| Multi-scale sizes | [6.25, 12.5, 25.0, 50.0] Mpc/h | [6.25, 12.5, 25.0, 50.0] Mpc/h |
| Target resolution | 128 × 128 | 128 × 128 |
| Periodic boundaries | Minimum image convention | Minimum image convention |
| Normalization | log10(x+1), then Z-score | log10(x+1), then Z-score |
| Stats files | `data/*_normalization_stats.npz` | `data/*_normalization_stats.npz` |

## Normalization

All fields are normalized identically during training AND inference:

```python
# Transform: log10(field + 1) then Z-score normalize
log_field = np.log10(field + 1)
normalized = (log_field - mean) / std

# Statistics stored in data/*.npz:
# - dark_matter_normalization_stats.npz
# - gas_normalization_stats.npz  
# - stellar_normalization_stats.npz
```

## Verifying Consistency

```bash
python -m pytest tests/test_data_generation.py -v
```

Key tests:
- `test_multiscale_extraction_consistency` - Verify scale extraction matches
- `test_normalization_consistency` - Verify normalization is identical
- `test_periodic_boundary_handling` - Verify periodic boundaries match

## File Descriptions

| File | Description |
|------|-------------|
| `process_simulations.py` | Main MPI-parallel script for high-mass halos (M > 10^13 M☉, 10 rotations) |
| `process_simulations_lowmass.py` | MPI-parallel script for low-mass halos (10^12 < M ≤ 10^13 M☉, 1 rotation) |
| `run_mpi_cpu.sh` | SLURM script for Stage 1 (8 nodes, 128 MPI ranks) |
| `run_mpi_cpu_lowmass.sh` | SLURM array job for Stage 2 (103 tasks × 16 MPI ranks) |
| `add_large_scale.py` | Legacy utility to add multi-scale context (not needed for current pipeline) |
