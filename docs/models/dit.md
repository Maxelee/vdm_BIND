# Diffusion Transformer (DiT)

DiT replaces the traditional UNet backbone with a transformer architecture, achieving state-of-the-art quality on image generation tasks.

## Overview

Diffusion Transformers use the transformer architecture's attention mechanism for the denoising network, providing better scalability and quality at the cost of increased compute.

**When to use DiT:**
- ✅ Best possible generation quality
- ✅ Large-scale training with sufficient compute
- ✅ When attention over the full image is important
- ⚠️ Slower inference than UNet-based models
- ⚠️ Higher memory requirements

## Mathematical Background

### Patch Embedding

Images are divided into non-overlapping patches and linearly embedded:

$$z_0 = \text{PatchEmbed}(x) \in \mathbb{R}^{N \times D}$$

where $N = (H/p) \times (W/p)$ is the number of patches and $p$ is the patch size.

### Positional Encoding

2D sinusoidal position embeddings encode spatial location:

$$\text{pos}_{i,j} = [\sin(\omega_k i), \cos(\omega_k i), \sin(\omega_k j), \cos(\omega_k j)]$$

### Adaptive Layer Normalization (adaLN-Zero)

DiT conditions on timestep $t$ and class/parameters $c$ via adaptive normalization:

$$\text{adaLN}(h, t, c) = \gamma_{t,c} \cdot \frac{h - \mu}{\sigma} + \beta_{t,c}$$

where $\gamma, \beta$ are predicted from the conditioning:

$$[\gamma, \beta, \alpha] = \text{MLP}(t + c)$$

The $\alpha$ parameter gates the residual connection (adaLN-Zero initialization).

### Transformer Blocks

Each DiT block contains:
1. **Self-Attention** with adaLN conditioning
2. **Cross-Attention** (optional) for spatial conditioning
3. **MLP** with adaLN conditioning
4. **Skip connections** with learned gating

### Unpatchify

Final output is unpatchified back to image space:

$$\hat{x} = \text{Unpatchify}(\text{Linear}(z_L))$$

## Key Features

1. **Global attention** - Each patch attends to all others
2. **Scalable** - Performance improves with model size
3. **adaLN-Zero** - Stable training with zero-initialized gates
4. **Cross-attention** - Rich conditioning on spatial inputs

## Model Variants

| Variant | Hidden Dim | Layers | Heads | Parameters |
|---------|------------|--------|-------|------------|
| DiT-S | 384 | 12 | 6 | ~33M |
| DiT-B | 768 | 12 | 12 | ~130M |
| DiT-L | 1024 | 24 | 16 | ~460M |
| DiT-XL | 1152 | 28 | 16 | ~675M |

## Usage Example

### Loading a Pre-trained Model

```python
from bind.workflow_utils import ConfigLoader, ModelManager, sample
import vdm

vdm.set_verbosity('summary')

# Load DiT model
config = ConfigLoader('configs/dit.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
model = model.cuda().eval()

# Generate samples
with torch.no_grad():
    samples = sample(
        model, 
        condition,
        batch_size=4,
        n_sampling_steps=250,
    )
```

### Training a New Model

```bash
# Use 32-bit precision for stability
python train_unified.py --model dit --config configs/dit.ini
```

**Important:** DiT training can be unstable with fp16. Use `precision=32` in the config.

## Configuration Options

Key parameters in `configs/dit.ini`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dit_variant` | `DiT-S` | Model size variant |
| `patch_size` | 4 | Patch size for embedding |
| `hidden_size` | 384 | Transformer hidden dimension |
| `depth` | 12 | Number of transformer blocks |
| `num_heads` | 6 | Attention heads |
| `mlp_ratio` | 4.0 | MLP expansion ratio |
| `precision` | 32 | **Use 32 for stability** |
| `n_sampling_steps` | 250 | Diffusion sampling steps |
| `learning_rate` | 1e-4 | Lower than UNet models |

## Architecture Details

```python
DiT(
    input_size=128,          # Image size
    patch_size=4,            # Patch size (128/4 = 32 patches per side)
    in_channels=3,           # DM_hydro, Gas, Stars
    hidden_size=384,         # DiT-S
    depth=12,                # Transformer layers
    num_heads=6,             # Attention heads
    mlp_ratio=4.0,           # MLP expansion
    cond_channels=4,         # DM + large-scale (1+3)
    use_cross_attention=True,  # Spatial conditioning
    n_params=6,              # Cosmological parameters
)
```

## Training Tips

1. **Use 32-bit precision** - fp16 can cause NaN with transformers
2. **Lower learning rate** - 1e-4 works better than 3e-4
3. **Warmup steps** - Use learning rate warmup (1000-5000 steps)
4. **Gradient clipping** - Clip gradients to 1.0
5. **Larger batch size** - Transformers benefit from larger batches

## Comparison with UNet

| Aspect | UNet | DiT |
|--------|------|-----|
| Attention | Local + Global | Global only |
| Inductive bias | Spatial hierarchy | None (learned) |
| Scaling | Diminishing returns | Linear improvement |
| Memory | Efficient | Higher |
| Speed | Faster | Slower |
| Quality | Excellent | Best |

## References

- **Peebles, W. & Xie, S.** (2023). [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). *ICCV 2023*.
- **Dosovitskiy, A., et al.** (2021). [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929). *ICLR 2021*.

## See Also

- [VDM](vdm.md) - UNet-based VDM
- [Interpolant](interpolant.md) - Faster alternative
- [Benchmark](../concepts/benchmark.md) - Compare model quality
