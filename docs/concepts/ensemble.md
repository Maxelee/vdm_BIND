# Model Ensembles

Combine multiple models for improved predictions and uncertainty estimation.

## Overview

Model ensembles aggregate predictions from multiple models to:

1. **Improve accuracy** - Average out individual model errors
2. **Reduce variance** - More stable predictions
3. **Quantify uncertainty** - Disagreement indicates uncertainty
4. **Leverage diversity** - Different architectures capture different features

## Creating Ensembles

### From Checkpoints

```python
from vdm.ensemble import create_ensemble_from_checkpoints

# Load multiple checkpoints
ensemble = create_ensemble_from_checkpoints([
    'checkpoints/vdm_seed42.ckpt',
    'checkpoints/vdm_seed123.ckpt',
    'checkpoints/interpolant_best.ckpt',
], device='cuda')

# Generate prediction
prediction = ensemble.predict(condition)
```

### Manual Construction

```python
from vdm.ensemble import ModelEnsemble

# Create ensemble from loaded models
ensemble = ModelEnsemble([model1, model2, model3])

# With custom weights
weighted_ensemble = ModelEnsemble(
    models=[model1, model2, model3],
    weights=[0.5, 0.3, 0.2],
)
```

## Ensemble Types

### WeightedEnsemble

Simple weighted average of predictions:

```python
from vdm.ensemble import WeightedEnsemble

ensemble = WeightedEnsemble(
    models=models,
    weights='uniform',  # or 'validation' or custom list
)

# If weights='validation', weights are set based on validation loss
ensemble.set_weights_from_validation(val_loader)
```

### ChannelWiseEnsemble

Different model weights per output channel:

```python
from vdm.ensemble import ChannelWiseEnsemble

# Best model varies by channel
ensemble = ChannelWiseEnsemble(
    models=models,
    channel_weights={
        0: [0.6, 0.3, 0.1],  # DM hydro
        1: [0.3, 0.5, 0.2],  # Gas
        2: [0.2, 0.2, 0.6],  # Stars
    }
)
```

### DiversityEnsemble

Select diverse model subset to maximize ensemble effectiveness:

```python
from vdm.ensemble import DiversityEnsemble

# Automatically select diverse models
ensemble = DiversityEnsemble.from_checkpoints(
    checkpoint_dir='checkpoints/',
    n_models=5,  # Select 5 most diverse
    diversity_metric='prediction',  # Based on prediction disagreement
)
```

## Ensemble Methods

### Simple Average

```python
# Default: average all model predictions
prediction = ensemble.predict(condition, method='mean')
```

### Median

More robust to outliers:

```python
prediction = ensemble.predict(condition, method='median')
```

### Trimmed Mean

Remove extreme predictions before averaging:

```python
prediction = ensemble.predict(condition, method='trimmed', trim_fraction=0.1)
```

### With Uncertainty

```python
mean, std = ensemble.predict_with_uncertainty(condition)
# std captures model disagreement
```

## Multi-Model Architecture Ensembles

Combine different model types for maximum diversity:

```python
from vdm.ensemble import MultiArchitectureEnsemble

ensemble = MultiArchitectureEnsemble({
    'vdm': 'checkpoints/vdm_best.ckpt',
    'interpolant': 'checkpoints/interpolant_best.ckpt',
    'ddpm': 'checkpoints/ddpm_best.ckpt',
})

# Predictions benefit from architectural diversity
prediction = ensemble.predict(condition)
```

## Training Ensemble Members

### Different Seeds

```bash
# Train with different random seeds
for seed in 42 123 456 789; do
    python train_unified.py --model vdm --seed $seed \
        --config configs/clean_vdm_aggressive_stellar.ini \
        --name vdm_seed$seed
done
```

### Different Architectures

```bash
# Train different model types
python train_unified.py --model vdm --config configs/clean_vdm_aggressive_stellar.ini
python train_unified.py --model interpolant --config configs/interpolant.ini
python train_unified.py --model ddpm --config configs/ddpm.ini
```

### Different Data Subsets (Bagging)

```python
from vdm.ensemble import train_bagged_ensemble

# Train models on bootstrap samples
checkpoints = train_bagged_ensemble(
    config_path='configs/clean_vdm_aggressive_stellar.ini',
    n_models=5,
    bootstrap_fraction=0.8,
)
```

## Performance Comparison

| Method | SSIM | Power Spectrum Error | Uncertainty Quality |
|--------|------|---------------------|---------------------|
| Single VDM | 0.85 | 5.2% | - |
| Single Interpolant | 0.86 | 4.8% | - |
| 3-Model Uniform | 0.88 | 4.1% | Good |
| 3-Model Weighted | 0.89 | 3.8% | Better |
| 5-Model Diverse | 0.90 | 3.5% | Best |

## Best Practices

1. **Use diverse models** - Different seeds, architectures, or hyperparameters
2. **Validate ensemble weights** - Use held-out validation set
3. **Check agreement** - Large disagreement suggests challenging inputs
4. **Consider compute cost** - More models = better but slower
5. **Channel-specific weights** - Different models excel at different outputs

## Example: Full Ensemble Pipeline

```python
import vdm
from vdm.ensemble import create_ensemble_from_checkpoints

vdm.set_verbosity('silent')

# Find all checkpoints
import glob
checkpoints = glob.glob('checkpoints/*_best.ckpt')

# Create ensemble
ensemble = create_ensemble_from_checkpoints(
    checkpoints, 
    device='cuda',
    weights='validation',  # Auto-weight by validation performance
)

# Generate predictions with uncertainty
mean, std = ensemble.predict_with_uncertainty(condition)

# High confidence regions
confidence = 1 / (std + 1e-6)
high_conf_mask = confidence > confidence.quantile(0.9)
```

## See Also

- [Uncertainty](uncertainty.md) - Uncertainty quantification
- [Benchmark](benchmark.md) - Evaluation metrics
- [Model Comparison](../models/index.md) - Individual model details
