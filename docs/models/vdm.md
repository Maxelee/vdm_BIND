# Variational Diffusion Models (VDM)

Variational Diffusion Models provide a continuous-time formulation of diffusion with learnable noise schedules.

## Overview

VDM is based on the variational perspective of diffusion models, optimizing an evidence lower bound (ELBO) that can be decomposed into interpretable terms.

**When to use VDM:**
- ✅ High-quality generation is critical
- ✅ You want well-understood, theoretically grounded models
- ✅ Scientific applications requiring interpretable uncertainty
- ⚠️ Inference is slower than consistency models

## Mathematical Background

### Forward Process (Noising)

The forward diffusion process adds Gaussian noise according to a schedule:

$$q(z_t | x) = \mathcal{N}(z_t; \alpha_t x, \sigma_t^2 I)$$

where:
- $z_t$ is the noisy sample at time $t \in [0, 1]$
- $\alpha_t$ and $\sigma_t$ define the noise schedule
- Signal-to-noise ratio: $\text{SNR}(t) = \alpha_t^2 / \sigma_t^2$

VDM uses a log-SNR parameterization: $\gamma(t) = \log(\alpha_t^2 / \sigma_t^2)$

### Reverse Process (Denoising)

The model learns to reverse the forward process:

$$p_\theta(x | z_t) = \mathcal{N}(x; \mu_\theta(z_t, t), \Sigma_\theta(z_t, t))$$

### Training Objective (ELBO)

The variational bound decomposes into:

$$\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \mathcal{L}_{\text{latent}} + \mathcal{L}_{\text{reconstruction}}$$

where the diffusion loss is:

$$\mathcal{L}_{\text{diffusion}} = \frac{1}{2} \mathbb{E}_{t, \epsilon} \left[ \gamma'(t) \| x - \hat{x}_\theta(z_t, t) \|^2 \right]$$

The $\gamma'(t)$ weighting (SNR weighting) is key to VDM's performance.

### Noise Schedule

VDM supports multiple noise schedules:

| Schedule | Formula | Use Case |
|----------|---------|----------|
| `fixed_linear` | $\gamma(t) = \gamma_{\min} + t(\gamma_{\max} - \gamma_{\min})$ | Default |
| `learned_linear` | Learnable endpoints | Adaptive |
| `cosine` | Cosine interpolation | Smoother |

## Key Features

1. **Continuous-time formulation** - No discrete step approximation
2. **SNR weighting** - Automatically balances loss across noise levels
3. **Learnable schedule** - Can adapt noise schedule during training
4. **ELBO decomposition** - Interpretable loss components

## Usage Example

### Loading a Pre-trained Model

```python
from bind.workflow_utils import ConfigLoader, ModelManager, sample
import vdm

vdm.set_verbosity('summary')

# Load VDM model
config = ConfigLoader('configs/clean_vdm_aggressive_stellar.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
model = model.cuda().eval()

# Generate samples
with torch.no_grad():
    samples = sample(
        model, 
        condition,          # DM + large-scale context
        batch_size=4,
        n_sampling_steps=250,
        conditional_params=params  # Cosmological parameters
    )
```

### Training a New Model

```bash
python train_unified.py --model vdm --config configs/clean_vdm_aggressive_stellar.ini
```

## Configuration Options

Key parameters in `configs/clean_vdm_aggressive_stellar.ini`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma_min` | -13.3 | Minimum log-SNR (high noise) |
| `gamma_max` | 5.0 | Maximum log-SNR (low noise) |
| `noise_schedule` | `fixed_linear` | Schedule type |
| `n_sampling_steps` | 250 | Reverse process steps |
| `learning_rate` | 3e-4 | Optimizer learning rate |
| `antithetic_time_sampling` | True | Use antithetic time samples |
| `channel_weights` | `1.0,1.0,2.0` | Per-channel loss weights |

## Architecture Details

VDM uses the `UNetVDM` score network:

```
UNetVDM(
    input_channels=3,           # DM_hydro, Gas, Stars
    conditioning_channels=1,    # DM condition
    large_scale_channels=3,     # Multi-scale context
    embedding_dim=256,          # Time embedding dimension
    n_blocks=4,                 # ResNet blocks per stage
    n_attention_heads=8,        # Attention heads
    use_fourier_features=True,  # Multi-scale Fourier features
)
```

## References

- **Kingma, D., et al.** (2021). [Variational Diffusion Models](https://arxiv.org/abs/2107.00630). *NeurIPS 2021*.
- **Luo, C.** (2022). [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970). *arXiv preprint*.

## See Also

- [DDPM](ddpm.md) - Discrete-time alternative
- [Interpolant](interpolant.md) - Flow matching approach
- [DiT](dit.md) - Transformer backbone
