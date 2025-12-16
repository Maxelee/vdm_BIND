# Consistency Models

Consistency Models enable single-step or few-step generation by learning to map any point on the diffusion trajectory directly to the data.

## Overview

Consistency Models learn a function that maps any noisy sample directly to the clean data, regardless of the noise level. This enables generation in just 1-4 steps instead of hundreds.

**When to use Consistency Models:**
- ✅ **Fastest inference** - single or few steps
- ✅ Real-time applications
- ✅ Large-scale inference (many halos)
- ⚠️ Slightly lower quality than full diffusion
- ⚠️ More complex training (distillation or CT)

## Mathematical Background

### Consistency Property

A consistency function $f_\theta$ satisfies:

$$f_\theta(z_t, t) = f_\theta(z_{t'}, t') = x \quad \forall t, t' \in [0, T]$$

meaning any point on the same trajectory maps to the same clean sample.

### Self-Consistency Loss

For Consistency Training (CT), we enforce self-consistency:

$$\mathcal{L}_{\text{CT}} = \mathbb{E}_{t, x} \left[ d(f_\theta(z_{t+\Delta t}, t+\Delta t), f_{\theta^-}(z_t, t)) \right]$$

where:
- $\theta^-$ is an EMA of the model weights
- $d(\cdot, \cdot)$ is a distance metric (LPIPS, L2, etc.)
- $\Delta t$ is the discretization step

### Boundary Condition

At $t=0$, the function must return the input:

$$f_\theta(x, 0) = x$$

This is enforced via skip connections.

### Consistency Distillation

Alternatively, distill from a pre-trained diffusion model:

$$\mathcal{L}_{\text{CD}} = \mathbb{E}_{t, x} \left[ d(f_\theta(z_{t+\Delta t}, t+\Delta t), f_{\theta^-}(\hat{z}_t, t)) \right]$$

where $\hat{z}_t$ is obtained by one DDIM step from $z_{t+\Delta t}$.

## Key Features

1. **Single-step generation** - Generate in one forward pass
2. **Multi-step refinement** - Improve quality with more steps
3. **Flexible training** - CT (from scratch) or CD (distillation)
4. **Denoising pretraining** - Warm start with score matching

## Usage Example

### Loading a Pre-trained Model

```python
from bind.workflow_utils import ConfigLoader, ModelManager, sample
import vdm

vdm.set_verbosity('summary')

# Load Consistency model
config = ConfigLoader('configs/consistency.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
model = model.cuda().eval()

# Single-step generation (fastest!)
with torch.no_grad():
    samples = sample(
        model, 
        condition,
        batch_size=4,
        n_sampling_steps=1,  # Just one step!
    )

# Multi-step for better quality
with torch.no_grad():
    samples = sample(
        model, 
        condition,
        batch_size=4,
        n_sampling_steps=4,  # Still very fast
    )
```

### Training a New Model

```bash
python train_unified.py --model consistency --config configs/consistency.ini
```

## Configuration Options

Key parameters in `configs/consistency.ini`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ct_n_steps` | 18 | CT discretization steps |
| `ema_decay` | 0.999 | EMA decay for target network |
| `denoising_warmup_epochs` | 50 | Epochs of denoising pretraining |
| `use_denoising_pretraining` | True | Start with score matching |
| `sigma_min` | 0.002 | Minimum noise level |
| `sigma_max` | 80.0 | Maximum noise level |
| `n_sampling_steps` | 1 | Default sampling steps |
| `x0_mode` | `x0` | Prediction mode |

## Training Strategy

Consistency Models benefit from a two-phase training:

### Phase 1: Denoising Pretraining (Optional)

Train as a standard score-matching model:

```python
# First 50 epochs: standard diffusion training
loss = ||score_model(z_t, t) - score_true||^2
```

### Phase 2: Consistency Training

Train with self-consistency loss:

```python
# After warmup: consistency training
loss = ||f(z_{t+dt}, t+dt) - f_ema(z_t, t)||^2
```

## Sampling Modes

| Steps | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| 1 | Good | **Instant** | Real-time, many samples |
| 2 | Better | Very fast | Balanced |
| 4 | High | Fast | Quality-sensitive |
| 8+ | Best | Medium | Maximum quality |

## Comparison with Other Models

| Aspect | DDPM | VDM | Consistency |
|--------|------|-----|-------------|
| Min steps | ~100 | ~100 | **1** |
| Quality (1 step) | N/A | N/A | Good |
| Quality (100 steps) | Excellent | Excellent | Excellent |
| Training | Simple | Simple | Complex |
| Theory | Score matching | ELBO | Self-consistency |

## References

- **Song, Y., et al.** (2023). [Consistency Models](https://arxiv.org/abs/2303.01469). *ICML 2023*.
- **Song, Y., et al.** (2023). [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189). *arXiv preprint*.

## See Also

- [VDM](vdm.md) - Full diffusion model
- [Interpolant](interpolant.md) - Another fast approach
- [Benchmark](../concepts/benchmark.md) - Quality comparison
