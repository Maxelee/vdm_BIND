# VDM-BIND Generalization Roadmap

## Goal
Make VDM-BIND usable with ANY N-body simulation and halo finder, not just CAMELS.

## Key Changes Required

### 1. Abstract Data Interface (`vdm/data_interface.py`)

Create an abstract base class for simulation data:

```python
from abc import ABC, abstractmethod

class SimulationLoader(ABC):
    """Abstract interface for loading simulation data."""
    
    @abstractmethod
    def load_particles(self, particle_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load positions and masses for a particle type."""
        pass
    
    @abstractmethod
    def get_box_size(self) -> float:
        """Return simulation box size in Mpc/h."""
        pass
    
    @abstractmethod
    def get_cosmological_parameters(self) -> Optional[Dict]:
        """Return cosmological parameters (optional)."""
        pass

class HaloCatalogLoader(ABC):
    """Abstract interface for halo catalogs."""
    
    @abstractmethod
    def load_halos(self, mass_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return positions, masses, and radii."""
        pass
```

### 2. Configuration-Based Parameter Conditioning

Make the parameter conditioning flexible:

```python
# In config file:
[CONDITIONING]
# Number of conditional parameters (0 = unconditional)
n_params = 0  

# OR specify custom parameters
n_params = 6
param_names = Omega_m, sigma_8, A_SN1, A_AGN1, z, halo_mass
param_file = /path/to/my_params.csv
```

### 3. On-the-fly Normalization Computation

Instead of hardcoded stats, compute during data loading:

```python
# First pass: compute stats
python compute_normalization.py --data_dir /path/to/your/data --output data/my_norm_stats.npz

# Training uses these stats
python train_unified.py --config my_config.ini --norm_stats data/my_norm_stats.npz
```

### 4. Flexible Training Data Format

Support multiple input formats:

```yaml
# config.yaml (new format)
data:
  format: "hdf5"  # or "npz", "fits", "npy"
  condition_key: "dm_density"  # key in file
  target_keys: ["dm_hydro", "gas", "stars"]
  large_scale_keys: ["dm_12.5", "dm_25.0", "dm_50.0"]  # optional
  params_key: "parameters"  # optional
```

## Implementation Priority

1. **Phase 1: Config-driven paths** (Easy)
   - Replace all hardcoded paths with config/env vars ✅ (mostly done)
   
2. **Phase 2: Abstract data loaders** (Medium)
   - Create `SimulationLoader` and `HaloCatalogLoader` interfaces
   - Implement for CAMELS, Illustris, SIMBA, user-custom
   
3. **Phase 3: Flexible conditioning** (Medium)
   - Allow 0-N parameters
   - Support custom parameter schemas
   
4. **Phase 4: Auto-normalization** (Easy)
   - Script to compute normalization from user data
   - Store in standard format

## Example: Using with Custom Data

```bash
# 1. Prepare your data in the expected format
python scripts/convert_to_bind_format.py \
    --input /path/to/my_sim \
    --output /path/to/bind_format \
    --halo_finder rockstar

# 2. Compute normalization statistics
python scripts/compute_normalization.py \
    --data_dir /path/to/bind_format/train \
    --output data/custom_norm_stats.npz

# 3. Train (unconditional or with custom params)
python train_unified.py \
    --model vdm \
    --config configs/custom.ini \
    --norm_stats data/custom_norm_stats.npz

# 4. Apply to new simulations
python bind_predict.py \
    --dmo_path /path/to/new_sim.hdf5 \
    --halo_catalog /path/to/halos.hdf5 \
    --output_dir ./predictions
```
