# bind Package API

## BIND Class

```{eval-rst}
.. module:: bind.bind
```

### BIND

```python
class BIND:
    def __init__(
        self,
        model: nn.Module,
        config: ConfigLoader,
        device: str = 'cuda',
        conserve_mass: bool = True,
    )
```

Main class for applying trained models to full N-body simulations.

**Key Methods:**

#### process_simulation

```python
def process_simulation(
    self,
    dmo_field: np.ndarray,
    halo_catalog: dict,
    n_realizations: int = 1,
) -> dict:
    """
    Process a full simulation through BIND pipeline.
    
    Parameters:
        dmo_field: (N, N, N) Dark matter density field
        halo_catalog: Dict with 'positions', 'masses', 'radii'
        n_realizations: Number of stochastic realizations
    
    Returns:
        Dict with 'dm_hydro', 'gas', 'stars' fields
    """
```

#### extract_halos

```python
def extract_halos(
    self,
    field: np.ndarray,
    halo_catalog: dict,
    cutout_size: int = 128,
) -> np.ndarray:
    """
    Extract halo-centric cutouts from field.
    
    Parameters:
        field: Full 3D field
        halo_catalog: Halo positions and properties
        cutout_size: Size of extracted cutouts
    
    Returns:
        Array of shape (n_halos, cutout_size, cutout_size, cutout_size)
    """
```

#### paste_halos

```python
def paste_halos(
    self,
    generated: np.ndarray,
    halo_catalog: dict,
    output_shape: tuple,
) -> np.ndarray:
    """
    Paste generated cutouts back to full field.
    
    Parameters:
        generated: Generated halo cutouts
        halo_catalog: Halo positions
        output_shape: Shape of output field
    
    Returns:
        Full field with pasted halos
    """
```

---

## Workflow Utilities

```{eval-rst}
.. module:: bind.workflow_utils
```

### ConfigLoader

```python
class ConfigLoader:
    def __init__(
        self,
        config_path: str,
        verbosity: str | int = 'summary',
    )
```

Load and parse training configuration files.

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_type` | str | Model architecture type |
| `checkpoint_path` | str | Path to model checkpoint |
| `normalization_stats` | dict | Channel normalization parameters |
| `img_size` | int | Input image size |
| `channels` | int | Number of output channels |
| `conditioning_channels` | int | Number of conditioning channels |

**Example:**

```python
from bind.workflow_utils import ConfigLoader

config = ConfigLoader('configs/clean_vdm_aggressive_stellar.ini')
print(config.model_type)  # 'vdm'
print(config.checkpoint_path)  # '/path/to/checkpoint.ckpt'
```

### ModelManager

```python
class ModelManager:
    @staticmethod
    def initialize(
        config: ConfigLoader,
        skip_data_loading: bool = False,
        device: str = None,
    ) -> tuple[DataModule, nn.Module]:
        """
        Initialize model from configuration.
        
        Parameters:
            config: Loaded configuration
            skip_data_loading: Skip creating data loaders
            device: Target device
        
        Returns:
            (data_module, model) tuple
        """
```

**Example:**

```python
from bind.workflow_utils import ConfigLoader, ModelManager

config = ConfigLoader('configs/interpolant.ini')
data_module, model = ModelManager.initialize(config)
model = model.cuda().eval()
```

### sample

```python
def sample(
    model: nn.Module,
    condition: torch.Tensor,
    num_steps: int = 100,
    device: str = 'cuda',
    return_trajectory: bool = False,
) -> torch.Tensor | tuple:
    """
    Generate samples from model.
    
    Parameters:
        model: Trained generative model
        condition: Conditioning input
        num_steps: Number of diffusion/integration steps
        device: Target device
        return_trajectory: Return full sampling trajectory
    
    Returns:
        Generated samples, or (samples, trajectory) if return_trajectory
    """
```

---

## Power Spectrum Analysis

```{eval-rst}
.. module:: bind.power_spec
```

### compute_power_spectrum_simple

```python
def compute_power_spectrum_simple(
    field: np.ndarray,
    BoxSize: float = 50.0,
    MAS: str = 'CIC',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D power spectrum.
    
    Parameters:
        field: 2D density field
        BoxSize: Physical box size in Mpc/h
        MAS: Mass assignment scheme ('NGP', 'CIC', 'TSC')
    
    Returns:
        (Pk, k) - Power spectrum and wavenumbers
    """
```

### compute_power_spectrum_batch

```python
def compute_power_spectrum_batch(
    fields: np.ndarray,
    BoxSize: float = 6.25,
    MAS: str = 'CIC',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power spectrum for batch of fields.
    
    Parameters:
        fields: (N, H, W) batch of 2D fields
        BoxSize: Physical box size in Mpc/h
        MAS: Mass assignment scheme
    
    Returns:
        (k, Pk, Nmodes) - Wavenumbers, power spectra, mode counts
    """
```

### compute_cross_power_spectrum

```python
def compute_cross_power_spectrum(
    field1: np.ndarray,
    field2: np.ndarray,
    BoxSize: float = 50.0,
    MAS: str = 'CIC',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cross-power spectrum and correlation coefficient.
    
    Parameters:
        field1, field2: 2D density fields
        BoxSize: Physical box size
        MAS: Mass assignment scheme
    
    Returns:
        (k, r, Pk_cross) - Wavenumbers, correlation, cross-power
    """
```

---

## Analysis Utilities

```{eval-rst}
.. module:: bind.analyses
```

### evaluate_halo

```python
def evaluate_halo(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    metrics: list[str] = ['mse', 'ssim', 'pk_error'],
) -> dict:
    """
    Evaluate single halo prediction.
    
    Parameters:
        prediction: Predicted halo fields
        ground_truth: True halo fields
        metrics: List of metrics to compute
    
    Returns:
        Dict of metric values
    """
```

### evaluate_simulation

```python
def evaluate_simulation(
    predicted_fields: dict,
    true_fields: dict,
    box_size: float = 50.0,
) -> dict:
    """
    Evaluate full simulation prediction.
    
    Parameters:
        predicted_fields: Dict with 'dm_hydro', 'gas', 'stars'
        true_fields: Ground truth fields
        box_size: Simulation box size
    
    Returns:
        Dict with per-field metrics
    """
```

### compute_radial_profile

```python
def compute_radial_profile(
    field: np.ndarray,
    center: tuple = None,
    n_bins: int = 50,
    r_max: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radial density profile.
    
    Parameters:
        field: 2D or 3D density field
        center: Center point (defaults to field center)
        n_bins: Number of radial bins
        r_max: Maximum radius
    
    Returns:
        (r, profile) - Radii and mean density in each bin
    """
```

---

## Usage Example

Complete BIND workflow:

```python
import vdm
from bind import BIND
from bind.workflow_utils import ConfigLoader, ModelManager
from bind.power_spec import compute_power_spectrum_simple
from vdm.io_utils import load_simulation, load_halo_catalog

# Suppress output
vdm.set_verbosity('silent')

# Load model
config = ConfigLoader('configs/clean_vdm_aggressive_stellar.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
model = model.cuda().eval()

# Load simulation
dmo_path = '/path/to/dmo/simulation'
hydro_path = '/path/to/hydro/simulation'

dmo_field = load_simulation(dmo_path, hydro_path)[0]  # DMO particles
halo_catalog = load_halo_catalog(f'{dmo_path}/fof_subhalo_tab_090.hdf5')

# Run BIND
bind = BIND(model, config, conserve_mass=True)
result = bind.process_simulation(
    dmo_field, 
    halo_catalog,
    n_realizations=5,
)

# Analyze results
from bind.analyses import evaluate_simulation

metrics = evaluate_simulation(
    result,
    true_fields={'dm_hydro': ..., 'gas': ..., 'stars': ...},
)
print(f"Gas power spectrum error: {metrics['gas']['pk_error']:.2%}")
```
