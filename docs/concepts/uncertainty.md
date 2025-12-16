# Uncertainty Quantification

Quantifying prediction uncertainty is critical for scientific applications of generative models.

## Overview

VDM-BIND provides multiple methods for uncertainty quantification:

1. **Multi-realization sampling** - Generate multiple samples and measure variance
2. **MC Dropout** - Enable dropout at inference for approximate Bayesian inference
3. **Ensemble uncertainty** - Combine predictions from multiple models

## Why Uncertainty Matters

In scientific applications like cosmological simulations:

- **Identify unreliable regions** - High uncertainty indicates model limitations
- **Error propagation** - Propagate uncertainty to downstream analyses
- **Model validation** - Compare uncertainty to actual errors
- **Active learning** - Focus data collection on high-uncertainty cases

## Multi-Realization Sampling

Generate multiple samples from the same condition to estimate variability:

```python
from vdm.uncertainty import UncertaintyEstimator

# Create estimator
estimator = UncertaintyEstimator(model, n_realizations=10)

# Get predictions with uncertainty
mean, std, samples = estimator.predict_with_uncertainty(
    condition, 
    return_samples=True
)

# mean: (B, C, H, W) - mean prediction
# std: (B, C, H, W) - standard deviation
# samples: (B, N, C, H, W) - all realizations
```

### Uncertainty Maps

Different methods for computing uncertainty maps:

```python
# Standard deviation
std_map = estimator.compute_uncertainty_map(samples, method='std')

# Interquartile range (robust to outliers)
iqr_map = estimator.compute_uncertainty_map(samples, method='iqr')

# Entropy (for discrete outputs)
entropy_map = estimator.compute_uncertainty_map(samples, method='entropy')
```

## MC Dropout

Enable dropout during inference for approximate Bayesian inference:

```python
from vdm.uncertainty import UncertaintyEstimator

# Enable MC Dropout mode
estimator = UncertaintyEstimator(
    model, 
    n_realizations=20,
    mc_dropout=True,
    dropout_rate=0.1  # Override model's dropout
)

# Predictions now have epistemic uncertainty
mean, std, _ = estimator.predict_with_uncertainty(condition)
```

### Epistemic vs Aleatoric Uncertainty

- **Aleatoric** (data uncertainty) - Captured by multi-realization sampling
- **Epistemic** (model uncertainty) - Captured by MC Dropout or ensembles

## Ensemble Uncertainty

Combine multiple trained models for robust uncertainty:

```python
from vdm.uncertainty import EnsembleUncertainty
from vdm.ensemble import create_ensemble_from_checkpoints

# Load ensemble
ensemble = create_ensemble_from_checkpoints([
    'checkpoints/model_seed42.ckpt',
    'checkpoints/model_seed123.ckpt',
    'checkpoints/model_seed456.ckpt',
])

# Compute ensemble uncertainty
estimator = EnsembleUncertainty(ensemble)
mean, std = estimator.predict_with_uncertainty(condition)
```

## Calibration Analysis

Check if predicted uncertainties match actual errors:

```python
from vdm.uncertainty import calibration_analysis

# Compute calibration metrics
results = calibration_analysis(
    predictions=mean,
    uncertainties=std,
    ground_truth=target,
)

print(f"ECE (Expected Calibration Error): {results['ece']:.4f}")
print(f"Coverage at 1σ: {results['coverage_1sigma']:.2%}")
print(f"Coverage at 2σ: {results['coverage_2sigma']:.2%}")
```

### Reliability Diagram

```python
from vdm.uncertainty import plot_reliability_diagram

fig = plot_reliability_diagram(
    predictions=mean,
    uncertainties=std,
    ground_truth=target,
    n_bins=10,
)
fig.savefig('reliability_diagram.png')
```

## Best Practices

1. **Use multiple realizations** - At least 10-20 for stable estimates
2. **Check calibration** - Ensure uncertainties are meaningful
3. **Combine methods** - MC Dropout + multi-realization captures both uncertainty types
4. **Per-channel analysis** - Uncertainty varies by output channel
5. **Spatial analysis** - Identify systematic patterns in uncertainty maps

## Example: Full Uncertainty Pipeline

```python
import vdm
from vdm.uncertainty import UncertaintyEstimator, calibration_analysis
from bind.workflow_utils import ConfigLoader, ModelManager

vdm.set_verbosity('silent')

# Load model
config = ConfigLoader('configs/interpolant.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
model = model.cuda().eval()

# Create uncertainty estimator
estimator = UncertaintyEstimator(
    model,
    n_realizations=20,
    mc_dropout=True,
)

# Generate predictions with uncertainty
mean, std, samples = estimator.predict_with_uncertainty(
    condition.cuda(),
    return_samples=True,
)

# Analyze calibration
calib = calibration_analysis(mean, std, target.cuda())
print(f"Model calibration: {calib['ece']:.4f} ECE")

# Identify high-uncertainty regions
high_uncertainty = std > std.mean() + 2 * std.std()
print(f"High uncertainty pixels: {high_uncertainty.sum().item()}")
```

## References

- Gal, Y. & Ghahramani, Z. (2016). [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142)
- Lakshminarayanan, B., et al. (2017). [Simple and Scalable Predictive Uncertainty Estimation](https://arxiv.org/abs/1612.01474)

## See Also

- [Ensemble](ensemble.md) - Model ensemble methods
- [Benchmark](benchmark.md) - Evaluation metrics
