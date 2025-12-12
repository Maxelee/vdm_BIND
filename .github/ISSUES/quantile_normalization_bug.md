# BUG: Some training scripts don't use quantile_path from config

## Summary

Some training scripts do not read or use the `quantile_path` parameter from config files, while others do. This causes a **train-test normalization mismatch** when using models trained without quantile support but with configs that specify `quantile_path`.

## Impact

- Models trained **without** quantile support output Z-score normalized stellar values (range ~[-7, +4])
- Inference dataloader reads `quantile_path` from config and normalizes targets to [0, 1]
- **Result**: Target uses quantile normalization, generated uses Z-score → need different unnormalization for each!

## Training Script Status

| Script | Uses `quantile_path`? | Status |
|--------|----------------------|--------|
| `train_model_clean.py` | ✅ **YES** (7 mentions) | Reads from config, passes to `get_astro_data()` |
| `train_triple_model.py` | ✅ **YES** (4 mentions) | Reads from config, passes to `get_astro_data()` |
| `train_ddpm.py` | ✅ **YES** (4 mentions) | Reads from config, passes to `get_astro_data()` |
| `train_dsm.py` | ✅ **YES** (4 mentions) | Reads from config, passes to `get_astro_data()` |
| `train_interpolant.py` | ❌ **NO** | Does NOT read quantile_path → always Z-score |
| `train_consistency.py` | ❌ **NO** | Does NOT read quantile_path → always Z-score |
| `train_ot_flow.py` | ❌ **NO** | Does NOT read quantile_path → always Z-score |

## Root Cause

In affected training scripts, `get_astro_data()` is called without the `quantile_path` argument:

```python
# train_interpolant.py, train_consistency.py, train_ot_flow.py
datamodule = get_astro_data(
    dataset=dataset,
    data_root=data_root,
    batch_size=batch_size,
    num_workers=num_workers,
    # MISSING: quantile_path=quantile_path
)
```

The default in `get_astro_data()` is `quantile_path=None`, which uses Z-score normalization.

## Temporary Workaround

The notebook `analysis/notebooks/training_validation.ipynb` has been updated with an `unnormalize_field` function that:
- Uses **quantile** inverse transform for **targets** (since dataloader uses quantile)
- Uses **Z-score** inverse transform for **generated** outputs (since affected models were trained with Z-score)

```python
# For targets (from dataloader with quantile):
target_unnorm = unnormalize_field(target, ch_idx, norm_stats, quantile_transformer)

# For generated (from model trained with Z-score):
gen_unnorm = unnormalize_field(gen, ch_idx, norm_stats, None)
```

## Proper Fix

1. Update affected training scripts (`train_interpolant.py`, `train_consistency.py`, `train_ot_flow.py`) to:
   - Read `quantile_path` from config file
   - Pass it to `get_astro_data()`

2. **Re-train** affected models (interpolant, consistency, ot_flow) with quantile normalization

3. Update inference code to detect which normalization was used during training (possibly store in checkpoint hparams)

## Example Fix for train_interpolant.py

```python
# Add to config parsing section
quantile_path = get_str('quantile_path', None)
if quantile_path and quantile_path.lower() == 'none':
    quantile_path = None

# Update get_astro_data call
datamodule = get_astro_data(
    dataset=dataset,
    data_root=data_root,
    batch_size=batch_size,
    num_workers=num_workers,
    quantile_path=quantile_path,  # ADD THIS
)
```

## Priority

**High** for new training runs - need to fix scripts before training new models with quantile normalization.

**Low** for current analysis - workaround in notebook handles existing checkpoints correctly.

## Date Discovered

December 11, 2025

## Files Affected

- `train_interpolant.py` - needs fix
- `train_consistency.py` - needs fix  
- `train_ot_flow.py` - needs fix
- `analysis/notebooks/training_validation.ipynb` - workaround implemented
