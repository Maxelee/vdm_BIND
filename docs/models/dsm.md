# Denoising Score Matching (DSM)

Score matching with a custom UNet for fair comparison with VDM.

## Overview

DSM learns the score function $\nabla_x \log p(x)$ using denoising score matching, then generates samples via Langevin dynamics or reverse SDE.

## Mathematical Background

### Score Function

$$s_\theta(x, t) \approx \nabla_x \log p_t(x)$$

### Denoising Score Matching

$$\mathcal{L} = \mathbb{E}_{t, x, \epsilon} \left[ \| s_\theta(x + \sigma_t \epsilon, t) + \epsilon / \sigma_t \|^2 \right]$$

### Sampling (Langevin Dynamics)

$$x_{k+1} = x_k + \frac{\eta}{2} s_\theta(x_k) + \sqrt{\eta} z_k$$

## Usage

```python
config = ConfigLoader('configs/dsm.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
samples = sample(model, condition, n_sampling_steps=500)
```

## Key Difference from DDPM

DSM in VDM-BIND uses the **same UNet architecture** as VDM for fair comparison, while DDPM uses the NCSNpp architecture from score_models.

## References

- Vincent, P. (2011). [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vin101/publications/smdae_techreport.pdf)
- Song, Y. & Ermon, S. (2019). [Generative Modeling by Estimating Gradients](https://arxiv.org/abs/1907.05600)
