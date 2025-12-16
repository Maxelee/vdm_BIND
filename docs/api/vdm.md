# vdm Package API

## Verbosity Control

```{eval-rst}
.. module:: vdm.verbosity
```

### Functions

#### set_verbosity
```python
vdm.set_verbosity(level: str | int) -> None
```
Set the global verbosity level.

**Parameters:**
- `level`: One of `'silent'`/`0`, `'summary'`/`1`, `'debug'`/`2`

**Example:**
```python
import vdm
vdm.set_verbosity('silent')  # Suppress all output
vdm.set_verbosity('debug')   # Maximum verbosity
```

#### get_verbosity
```python
vdm.get_verbosity() -> int
```
Get the current verbosity level.

#### quiet
```python
vdm.quiet() -> contextmanager
```
Context manager for temporarily silencing output.

**Example:**
```python
with vdm.quiet():
    # All vdm output suppressed here
    model.sample(...)
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `vdm.SILENT` | 0 | No output |
| `vdm.SUMMARY` | 1 | Key information only |
| `vdm.DEBUG` | 2 | Detailed debugging |

---

## VDM Model

```{eval-rst}
.. module:: vdm.vdm_model_clean
```

### LightCleanVDM

```python
class LightCleanVDM(pl.LightningModule):
    def __init__(
        self,
        channels: int = 3,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16,),
        use_ema: bool = True,
        ema_decay: float = 0.999,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        learning_rate: float = 1e-4,
        # ... additional parameters
    )
```

3-channel Variational Diffusion Model for DMO → Hydro mapping.

**Key Methods:**

```python
# Generate samples
samples = model.sample(condition, num_steps=100)

# Get loss for training
loss = model.get_diffusion_loss(batch)

# Forward pass (for Lightning)
loss = model(batch)
```

---

## Interpolant Model

```{eval-rst}
.. module:: vdm.interpolant_model
```

### LightInterpolant

```python
class LightInterpolant(pl.LightningModule):
    def __init__(
        self,
        channels: int = 3,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        interpolant_type: str = 'linear',  # 'linear', 'trig', 'vp'
        sigma_min: float = 0.001,
        learning_rate: float = 1e-4,
        # ... additional parameters
    )
```

Flow Matching / Stochastic Interpolant model.

**Key Methods:**

```python
# Generate samples (ODE integration)
samples = model.sample(condition, num_steps=50)

# Velocity prediction
velocity = model.predict_velocity(x_t, t, condition)
```

---

## DDPM Model

```{eval-rst}
.. module:: vdm.ddpm_model
```

### LightDDPM

```python
class LightDDPM(pl.LightningModule):
    def __init__(
        self,
        channels: int = 3,
        conditioning_channels: int = 1,
        beta_schedule: str = 'linear',
        n_timesteps: int = 1000,
        learning_rate: float = 1e-4,
        # ... additional parameters
    )
```

DDPM with NCSNpp architecture from score_models package.

---

## DSM Model

```{eval-rst}
.. module:: vdm.dsm_model
```

### LightDSM

```python
class LightDSM(pl.LightningModule):
    def __init__(
        self,
        channels: int = 3,
        conditioning_channels: int = 1,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        learning_rate: float = 1e-4,
        # ... additional parameters
    )
```

Denoising Score Matching model.

---

## DiT Model

```{eval-rst}
.. module:: vdm.dit_model
```

### LightDiTVDM

```python
class LightDiTVDM(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 4,
        in_channels: int = 3,
        condition_channels: int = 4,
        hidden_size: int = 384,  # DiT-S
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        learning_rate: float = 1e-4,
        # ... additional parameters
    )
```

Diffusion Transformer model.

**Variants:**

| Variant | hidden_size | depth | num_heads |
|---------|-------------|-------|-----------|
| DiT-S | 384 | 12 | 6 |
| DiT-B | 768 | 12 | 12 |
| DiT-L | 1024 | 24 | 16 |
| DiT-XL | 1152 | 28 | 16 |

---

## Consistency Model

```{eval-rst}
.. module:: vdm.consistency_model
```

### LightConsistency

```python
class LightConsistency(pl.LightningModule):
    def __init__(
        self,
        channels: int = 3,
        conditioning_channels: int = 1,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        learning_rate: float = 1e-4,
        # ... additional parameters
    )
```

Consistency Model for single-step generation.

---

## Networks

```{eval-rst}
.. module:: vdm.networks_clean
```

### UNet

```python
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16,),
        time_embed_dim: int = 256,
        cond_embed_dim: int = 64,
        use_fourier_features: bool = True,
        use_cross_attention: bool = True,
        # ... additional parameters
    )
```

U-Net architecture with Fourier features and attention.

---

## Uncertainty

```{eval-rst}
.. module:: vdm.uncertainty
```

### UncertaintyEstimator

```python
class UncertaintyEstimator:
    def __init__(
        self,
        model: nn.Module,
        n_realizations: int = 10,
        mc_dropout: bool = False,
        dropout_rate: float = None,
    )
    
    def predict_with_uncertainty(
        self,
        condition: torch.Tensor,
        return_samples: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns (mean, std, [samples])"""
```

### calibration_analysis

```python
def calibration_analysis(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    ground_truth: torch.Tensor,
) -> dict:
    """
    Returns:
        - ece: Expected Calibration Error
        - coverage_1sigma: Fraction within 1 std
        - coverage_2sigma: Fraction within 2 std
    """
```

---

## Ensemble

```{eval-rst}
.. module:: vdm.ensemble
```

### ModelEnsemble

```python
class ModelEnsemble:
    def __init__(
        self,
        models: list[nn.Module],
        weights: list[float] | str = 'uniform',
    )
    
    def predict(
        self,
        condition: torch.Tensor,
        method: str = 'mean',
    ) -> torch.Tensor:
        """Generate ensemble prediction."""
    
    def predict_with_uncertainty(
        self,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) across ensemble members."""
```

### create_ensemble_from_checkpoints

```python
def create_ensemble_from_checkpoints(
    checkpoint_paths: list[str],
    device: str = 'cuda',
    weights: list[float] | str = 'uniform',
) -> ModelEnsemble:
    """Load models from checkpoints and create ensemble."""
```

---

## Benchmark

```{eval-rst}
.. module:: vdm.benchmark
```

### BenchmarkSuite

```python
class BenchmarkSuite:
    def __init__(
        self,
        metrics: list[str] = ['ssim', 'psnr', 'mse'],
        device: str = 'cuda',
        per_channel: bool = False,
        pk_config: dict = None,
    )
    
    def evaluate(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> dict:
        """Evaluate single batch."""
    
    def evaluate_dataset(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> dict:
        """Evaluate entire dataset."""
```

### quick_benchmark

```python
def quick_benchmark(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> dict:
    """One-line model evaluation."""
```

### compare_models

```python
def compare_models(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    metrics: list[str] = ['ssim', 'psnr', 'pk_error'],
    n_realizations: int = 5,
) -> ModelComparison:
    """Compare multiple models on same test set."""
```
