# Triple VDM

Three independent single-channel VDMs for specialized per-channel generation.

## Overview

Triple VDM trains three separate VDM models, one for each output channel (DM_hydro, Gas, Stars), allowing each model to specialize.

## When to Use

- ✅ Channels have very different characteristics
- ✅ Want per-channel hyperparameter tuning
- ✅ Easier debugging of individual channels
- ⚠️ No cross-channel consistency during generation
- ⚠️ 3x parameters and training time

## Architecture

```
TripleVDM:
├── hydro_dm_model: UNetVDM (1 output channel)
├── gas_model: UNetVDM (1 output channel)  
└── stars_model: UNetVDM (1 output channel)
```

## Usage

```python
config = ConfigLoader('configs/clean_vdm_triple.ini')
_, model = ModelManager.initialize(config, skip_data_loading=True)
samples = sample(model, condition, n_sampling_steps=250)
# Returns shape (B, 1, 3, H, W) - channels are concatenated
```

## Comparison with 3-Channel VDM

| Aspect | 3-Channel VDM | Triple VDM |
|--------|---------------|------------|
| Cross-channel | Learned | None |
| Parameters | 1x | 3x |
| Training | Single run | Single run (parallel) |
| Per-channel tuning | Shared | Independent |

## Configuration

Uses same config format as VDM, but model_name should contain "triple":

```ini
[TRAINING]
model_name = triple_vdm_separate_models
```

## References

Same as [VDM](vdm.md).
