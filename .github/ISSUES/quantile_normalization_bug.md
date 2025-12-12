# BUG: Training scripts don't use quantile_path from config

## Summary

Training scripts (`train_consistency.py`, `train_ddpm.py`, `train_dsm.py`, `train_interpolant.py`, `train_ot_flow.py`) do not read or use the `quantile_path` parameter from config files. This causes a **train-test normalization mismatch** when `quantile_path` is specified in config.

## Impact

- **All models are trained with Z-score normalization for the stellar channel**
- Config files specify `quantile_path` but it's ignored during training
- Inference code (`ConfigLoader`, `ModelManager`) DOES read `quantile_path` and applies quantile normalization to test data
- This mismatch causes stellar channel predictions to be **orders of magnitude wrong** when using quantile unnormalization

## Root Cause

In all training scripts, `get_astro_data()` is called without the `quantile_path` argument:

```python
# train_consistency.py line ~429
datamodule = get_astro_data(
    dataset=dataset,
    data_root=data_root,
    batch_size=batch_size,
    num_workers=num_workers,
    # MISSING: quantile_path=quantile_path
)
```

The default in `get_astro_data()` is `quantile_path=None`, which uses Z-score normalization.

## Affected Files

- `train_consistency.py`
- `train_ddpm.py`
- `train_dsm.py`
- `train_interpolant.py`
- `train_ot_flow.py`
- `train_model_clean.py` (needs verification)
- `train_triple_model.py` (needs verification)

## Temporary Workaround

The notebook `analysis/notebooks/training_validation.ipynb` has been updated to always use Z-score unnormalization for the stellar channel, regardless of `quantile_transformer` argument.

## Proper Fix

1. Update all training scripts to:
   - Read `quantile_path` from config file
   - Pass it to `get_astro_data()`

2. Re-train models with quantile normalization if desired

3. Update inference code to detect which normalization was used during training (possibly store in checkpoint hparams)

## Example Fix for train_consistency.py

```python
# Add to config parsing section (~line 280)
quantile_path = get_str('quantile_path', None)
if quantile_path and quantile_path.lower() == 'none':
    quantile_path = None

# Update get_astro_data call (~line 429)
datamodule = get_astro_data(
    dataset=dataset,
    data_root=data_root,
    batch_size=batch_size,
    num_workers=num_workers,
    quantile_path=quantile_path,  # ADD THIS
)
```

## Priority

Medium - Current workaround works, but proper fix needed before training new models with quantile normalization.

## Date Discovered

December 11, 2025
