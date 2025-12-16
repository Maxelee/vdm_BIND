# DDPM (Denoising Diffusion Probabilistic Models)

Classic discrete-time diffusion models using the NCSNpp architecture.

## Overview

DDPM defines a discrete-time Markov chain that gradually adds noise, then learns to reverse this process.

## Mathematical Background

### Forward Process

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

### Reverse Process

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

### Training

$$\mathcal{L} = \mathbb{E}_{t, x, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

## Usage

```python
config = ConfigLoader('configs/ddpm.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
samples = sample(model, condition, n_sampling_steps=1000)
```

## References

- Ho, J., et al. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Song, Y., et al. (2021). [Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456)
