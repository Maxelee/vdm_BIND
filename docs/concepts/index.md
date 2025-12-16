# Core Concepts

This section covers important concepts and utilities in VDM-BIND beyond the core generative models.

## Scientific Uncertainty

```{toctree}
:maxdepth: 2

uncertainty
```

In scientific applications, knowing *how confident* a model is matters as much as the prediction itself. The [Uncertainty](uncertainty.md) module provides:

- Multi-realization sampling for stochastic uncertainty
- MC Dropout for epistemic uncertainty
- Calibration analysis to validate uncertainty estimates

## Model Ensembles

```{toctree}
:maxdepth: 2

ensemble
```

Combining multiple models improves both accuracy and uncertainty estimation. The [Ensemble](ensemble.md) module provides:

- Weighted and uniform ensemble methods
- Channel-wise ensembles for per-output specialization
- Multi-architecture ensembles for maximum diversity

## Benchmarking

```{toctree}
:maxdepth: 2

benchmark
```

Standardized evaluation is critical for fair model comparison. The [Benchmark](benchmark.md) module provides:

- Image quality metrics (SSIM, PSNR, MSE)
- Domain-specific metrics (power spectrum, mass conservation)
- Statistical significance testing

## Quick Links

| Concept | Module | Key Functions |
|---------|--------|---------------|
| [Uncertainty](uncertainty.md) | `vdm.uncertainty` | `UncertaintyEstimator`, `calibration_analysis` |
| [Ensemble](ensemble.md) | `vdm.ensemble` | `ModelEnsemble`, `create_ensemble_from_checkpoints` |
| [Benchmark](benchmark.md) | `vdm.benchmark` | `BenchmarkSuite`, `quick_benchmark`, `compare_models` |
