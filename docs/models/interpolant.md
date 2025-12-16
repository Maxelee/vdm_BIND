# Flow Matching / Stochastic Interpolants

Flow Matching provides an efficient alternative to diffusion models by learning velocity fields that transport samples along straight paths.

## Overview

Stochastic Interpolants (also called Flow Matching) learn a velocity field that transforms noise into data along interpolated paths. This approach often requires fewer sampling steps than traditional diffusion.

**When to use Interpolants:**
- ✅ Fast, efficient generation
- ✅ Fewer sampling steps (100-250 vs 1000)
- ✅ Theoretically elegant formulation
- ✅ Good for large-scale inference (BIND pipeline)

## Mathematical Background

### Interpolation Path

Instead of a noising process, we define an interpolation between noise $z_0 \sim \mathcal{N}(0, I)$ and data $x$:

$$z_t = \alpha_t x + \sigma_t z_0$$

where $\alpha_0 = 0, \alpha_1 = 1$ and $\sigma_0 = 1, \sigma_1 = 0$.

Common choices:
- **Linear**: $\alpha_t = t$, $\sigma_t = 1 - t$
- **Trigonometric**: $\alpha_t = \sin(\frac{\pi t}{2})$, $\sigma_t = \cos(\frac{\pi t}{2})$

### Velocity Field

The velocity field $v_\theta(z_t, t)$ describes how samples move along the interpolation:

$$\frac{dz_t}{dt} = v_\theta(z_t, t)$$

### Training Objective

The model learns to predict the velocity:

$$\mathcal{L} = \mathbb{E}_{t, x, z_0} \left[ \| v_\theta(z_t, t) - \dot{z}_t \|^2 \right]$$

where $\dot{z}_t = \dot{\alpha}_t x + \dot{\sigma}_t z_0$ is the true velocity.

### Stochastic vs Deterministic

Two modes are available:

1. **Deterministic** (`x0_mode='x0'`): Predict $x$ directly from $z_t$
2. **Stochastic** (`x0_mode='stochastic'`): Add noise during sampling for diversity

## Key Features

1. **Straight paths** - Efficient transport geometry
2. **Fewer steps** - 100-250 steps vs 1000 for DDPM
3. **Flexible interpolation** - Linear, trigonometric, or custom
4. **Both ODE and SDE sampling** - Deterministic or stochastic

## Usage Example

### Loading a Pre-trained Model

```python
from bind.workflow_utils import ConfigLoader, ModelManager, sample
import vdm

vdm.set_verbosity('summary')

# Load Interpolant model
config = ConfigLoader('configs/interpolant.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
model = model.cuda().eval()

# Generate samples (fewer steps than VDM!)
with torch.no_grad():
    samples = sample(
        model, 
        condition,
        batch_size=4,
        n_sampling_steps=100,  # Faster than diffusion!
    )
```

### Training a New Model

```bash
# Deterministic flow matching
python train_unified.py --model interpolant --config configs/interpolant.ini

# Stochastic interpolant
python train_unified.py --model interpolant --config configs/stochastic_interpolant.ini
```

## Configuration Options

Key parameters in `configs/interpolant.ini`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x0_mode` | `x0` | Prediction mode: 'x0' (deterministic) or 'stochastic' |
| `interpolant_type` | `linear` | Path type: 'linear' or 'trigonometric' |
| `n_sampling_steps` | 100 | ODE/SDE integration steps |
| `beta_min` | 0.1 | Min diffusion coefficient (for SDE) |
| `beta_max` | 20.0 | Max diffusion coefficient (for SDE) |
| `learning_rate` | 3e-4 | Optimizer learning rate |

## Architecture Details

Uses the same `UNetVDM` backbone as VDM, but wrapped in `VelocityNetWrapper`:

```
VelocityNetWrapper(
    net=UNetVDM(...),  # Base score network
    # Wrapper handles velocity → score conversion
)
```

The wrapper converts between velocity and score parameterizations as needed.

## Comparison with VDM

| Aspect | VDM | Interpolant |
|--------|-----|-------------|
| Formulation | Diffusion (noising) | Flow (interpolation) |
| Training | ELBO / Score matching | Velocity matching |
| Sampling steps | 250-1000 | 100-250 |
| Path geometry | Curved | Straight |
| Theory | Variational inference | Optimal transport |

## References

- **Albergo, M. & Vanden-Eijnden, E.** (2023). [Building Normalizing Flows with Stochastic Interpolants](https://arxiv.org/abs/2209.15571). *ICLR 2023*.
- **Lipman, Y., et al.** (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). *ICLR 2023*.
- **Liu, X., et al.** (2023). [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003). *ICLR 2023*.

## See Also

- [OT Flow](ot_flow.md) - Optimal transport variant
- [VDM](vdm.md) - Traditional diffusion approach
- [Consistency](consistency.md) - Few-step sampling
