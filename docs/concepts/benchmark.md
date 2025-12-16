# Benchmark Suite

Standardized evaluation metrics for comparing model performance.

## Overview

The benchmark suite provides:

1. **Standardized metrics** - Consistent evaluation across models
2. **Domain-specific metrics** - Astrophysical quantities (power spectrum, mass)
3. **Quick benchmarking** - Easy model comparison
4. **Reproducible evaluation** - Fixed test sets and random seeds

## Quick Start

```python
from vdm.benchmark import quick_benchmark

# One-line evaluation
results = quick_benchmark(
    model,
    test_loader,
    device='cuda',
)

print(results)
# {'ssim': 0.87, 'psnr': 28.4, 'mse': 0.012, 'pk_error': 0.045, ...}
```

## Metrics

### Image Quality Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **SSIM** | [0, 1] | Structural similarity (higher = better) |
| **PSNR** | [0, ∞) | Peak signal-to-noise ratio in dB |
| **MSE** | [0, ∞) | Mean squared error (lower = better) |
| **LPIPS** | [0, 1] | Perceptual similarity (lower = better) |

```python
from vdm.benchmark import BenchmarkSuite

bench = BenchmarkSuite(metrics=['ssim', 'psnr', 'mse', 'lpips'])
results = bench.evaluate(predictions, ground_truth)
```

### Power Spectrum Metrics

Critical for cosmological applications:

```python
from vdm.benchmark import BenchmarkSuite

bench = BenchmarkSuite(
    metrics=['pk_error', 'pk_ratio', 'transfer_function'],
    pk_config={
        'box_size': 6.25,  # Mpc/h
        'mas': 'CIC',
    }
)

results = bench.evaluate(predictions, ground_truth)
# pk_error: Mean relative error in P(k)
# pk_ratio: Ratio P_pred(k) / P_true(k)
# transfer_function: Scale-dependent bias
```

### Mass-based Metrics

```python
from vdm.benchmark import BenchmarkSuite

bench = BenchmarkSuite(metrics=['mass_error', 'mass_fraction'])
results = bench.evaluate(predictions, ground_truth)
# mass_error: Relative error in total mass
# mass_fraction: Fraction of mass correctly placed
```

### Per-Channel Metrics

Evaluate each output channel separately:

```python
bench = BenchmarkSuite(
    metrics=['ssim', 'psnr'],
    per_channel=True,
)
results = bench.evaluate(predictions, ground_truth)
# results['ssim_ch0'], results['ssim_ch1'], results['ssim_ch2']
```

## BenchmarkSuite Class

Full-featured benchmark interface:

```python
from vdm.benchmark import BenchmarkSuite

# Create suite with all metrics
bench = BenchmarkSuite(
    metrics=['ssim', 'psnr', 'mse', 'pk_error', 'mass_error'],
    device='cuda',
    per_channel=True,
)

# Evaluate single batch
batch_results = bench.evaluate(pred_batch, true_batch)

# Evaluate full dataset
full_results = bench.evaluate_dataset(model, test_loader)

# Get summary statistics
summary = bench.summarize(full_results)
print(summary)
```

### Configuration Options

```python
bench = BenchmarkSuite(
    metrics=['ssim', 'psnr', 'pk_error'],
    
    # Power spectrum configuration
    pk_config={
        'box_size': 6.25,    # Physical box size
        'n_bins': 32,         # Number of k bins
        'mas': 'CIC',         # Mass assignment scheme
        'kmin': 0.1,          # Minimum k
        'kmax': 10.0,         # Maximum k
    },
    
    # General options
    device='cuda',
    per_channel=True,
    reduce='mean',           # 'mean', 'median', or 'none'
)
```

## Model Comparison

Compare multiple models on the same test set:

```python
from vdm.benchmark import compare_models

models = {
    'VDM': vdm_model,
    'Interpolant': interpolant_model,
    'DDPM': ddpm_model,
}

comparison = compare_models(
    models,
    test_loader,
    metrics=['ssim', 'psnr', 'pk_error'],
    n_realizations=5,  # Average over realizations
)

# Pretty print results
comparison.print_table()

# Export to CSV
comparison.to_csv('model_comparison.csv')
```

### Comparison Output

```
Model        | SSIM  | PSNR  | Pk Error
-------------|-------|-------|----------
VDM          | 0.871 | 28.4  | 4.5%
Interpolant  | 0.883 | 29.1  | 4.1%
DDPM         | 0.856 | 27.8  | 5.2%
```

## Channel-Specific Evaluation

Different channels may have different quality:

```python
from vdm.benchmark import evaluate_channels

channel_results = evaluate_channels(
    predictions,  # (B, 3, H, W)
    ground_truth,
    channel_names=['DM_hydro', 'Gas', 'Stars'],
    metrics=['ssim', 'pk_error'],
)

for ch_name, results in channel_results.items():
    print(f"{ch_name}: SSIM={results['ssim']:.3f}, Pk Error={results['pk_error']:.2%}")
```

## Statistical Testing

Compare models with statistical significance:

```python
from vdm.benchmark import statistical_comparison

stats = statistical_comparison(
    model_a=vdm_model,
    model_b=interpolant_model,
    test_loader=test_loader,
    metric='ssim',
    n_bootstrap=1000,
)

print(f"Model A SSIM: {stats['a_mean']:.3f} ± {stats['a_std']:.3f}")
print(f"Model B SSIM: {stats['b_mean']:.3f} ± {stats['b_std']:.3f}")
print(f"Difference: {stats['diff_mean']:.3f} (p={stats['p_value']:.4f})")
```

## Benchmark Report

Generate comprehensive HTML report:

```python
from vdm.benchmark import generate_report

generate_report(
    models={'VDM': vdm_model, 'Interpolant': interp_model},
    test_loader=test_loader,
    output_path='benchmark_report.html',
    include_plots=True,
    include_samples=5,  # Include 5 sample comparisons
)
```

## Example: Full Benchmarking Pipeline

```python
import vdm
from vdm.benchmark import BenchmarkSuite, compare_models
from bind.workflow_utils import ConfigLoader, ModelManager

vdm.set_verbosity('silent')

# Load models
models = {}
for model_type in ['vdm', 'interpolant', 'ddpm']:
    config = ConfigLoader(f'configs/{model_type}.ini')
    _, model = ModelManager.initialize(config, skip_data_loading=True)
    models[model_type] = model.cuda().eval()

# Load test data
from vdm.astro_dataset import get_astro_data
_, _, test_loader = get_astro_data(
    'data/test/', batch_size=16, num_workers=4
)

# Run comparison
comparison = compare_models(
    models,
    test_loader,
    metrics=['ssim', 'psnr', 'pk_error', 'mass_error'],
    n_realizations=10,
)

# Print results
comparison.print_table()

# Save detailed results
comparison.to_json('benchmark_results.json')
```

## Best Practices

1. **Use multiple metrics** - Different metrics capture different aspects
2. **Per-channel evaluation** - Model performance varies by channel
3. **Statistical significance** - Use bootstrap for confidence intervals
4. **Multiple realizations** - Average over stochastic sampling
5. **Fixed test set** - Same samples for all models

## See Also

- [Uncertainty](uncertainty.md) - Uncertainty quantification
- [Ensemble](ensemble.md) - Model ensembles
- [Model Comparison](../models/index.md) - Model architecture details
