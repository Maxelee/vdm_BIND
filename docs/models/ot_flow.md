# Optimal Transport Flow Matching (OT Flow)

Flow matching with optimal transport couplings for more efficient paths.

## Overview

OT Flow extends flow matching by using optimal transport to find efficient couplings between noise and data distributions.

## Mathematical Background

### Standard Flow Matching

Pairs noise and data randomly: $(z_0, x) \sim \pi_{\text{independent}}$

### OT Flow Matching

Pairs using optimal transport: $(z_0, x) \sim \pi_{\text{OT}}$

The OT coupling minimizes transport cost:
$$\pi_{\text{OT}} = \arg\min_\pi \mathbb{E}_{(z_0, x) \sim \pi} \| z_0 - x \|^2$$

### Benefits

- Straighter paths → fewer sampling steps
- Better conditioning → faster convergence
- Lower variance gradients

## Usage

```python
config = ConfigLoader('configs/ot_flow.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
samples = sample(model, condition, n_sampling_steps=100)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ot_method` | `sinkhorn` | OT solver (sinkhorn, exact) |
| `ot_reg` | 0.01 | Entropic regularization |
| `n_sampling_steps` | 100 | Integration steps |

## References

- Lipman, Y., et al. (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- Tong, A., et al. (2023). [Improving and Generalizing Flow-Based Generative Models](https://arxiv.org/abs/2302.00482)
