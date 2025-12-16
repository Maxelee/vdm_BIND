# Model Guide

VDM-BIND supports multiple diffusion model architectures, each with different characteristics and trade-offs.

## Available Models

| Model | Type | Key Feature | Speed | Quality |
|-------|------|-------------|-------|---------|
| [VDM](vdm.md) | Variational | Continuous-time, learned schedule | Medium | High |
| [DDPM](ddpm.md) | Score-based | NCSNpp architecture | Medium | High |
| [Interpolant](interpolant.md) | Flow Matching | Stochastic interpolation | Fast | High |
| [DSM](dsm.md) | Score Matching | Denoising score matching | Medium | High |
| [Consistency](consistency.md) | Distillation | Single/few-step sampling | **Fast** | Good |
| [OT Flow](ot_flow.md) | Optimal Transport | Wasserstein-optimal paths | Medium | High |
| [DiT](dit.md) | Transformer | Attention-based | Slow | **Best** |

## Choosing a Model

### For Best Quality
- **DiT** (Diffusion Transformer) - State-of-the-art quality with transformer architecture
- **VDM** - Well-understood theory, excellent for scientific applications

### For Fast Inference
- **Consistency Model** - Single-step or few-step generation
- **Interpolant** - Efficient flow matching with fewer steps

### For Balanced Performance
- **DDPM** - Classic approach with good quality/speed balance
- **DSM** - Score matching with custom UNet for fair comparison

## Model Architecture

All models share a common UNet backbone (`UNetVDM`) with:

- **Fourier Features** - Multi-scale frequency encoding
- **Residual Blocks** - Deep feature extraction  
- **Attention Layers** - Global context modeling
- **Cross-Attention** - Conditioning on spatial inputs
- **Parameter Conditioning** - Scalar parameter injection

Exception: **DiT** uses a transformer backbone instead of UNet.

## Quick Comparison Code

```python
from bind.workflow_utils import ConfigLoader, ModelManager, sample
import time

models_to_test = {
    'Interpolant': 'configs/interpolant.ini',
    'Consistency': 'configs/consistency.ini',
    'DiT': 'configs/dit.ini',
}

for name, config_path in models_to_test.items():
    config = ConfigLoader(config_path, verbose=False)
    _, model = ModelManager.initialize(config, verbose=False, skip_data_loading=True)
    model = model.cuda().eval()
    
    # Time inference
    start = time.time()
    with torch.no_grad():
        samples = sample(model, condition, batch_size=4)
    elapsed = time.time() - start
    
    print(f"{name}: {elapsed:.2f}s for 4 samples")
```

## Configuration

Each model has a corresponding config file in `configs/`:

```
configs/
├── clean_vdm_aggressive_stellar.ini  # VDM
├── ddpm.ini                          # DDPM
├── interpolant.ini                   # Flow Matching
├── stochastic_interpolant.ini        # Stochastic Interpolant
├── dsm.ini                           # DSM
├── consistency.ini                   # Consistency Model
├── ot_flow.ini                       # OT Flow
└── dit.ini                           # DiT
```

## Training

Train any model using the unified training script:

```bash
python train_unified.py --model MODEL_TYPE --config configs/CONFIG.ini

# Examples:
python train_unified.py --model interpolant --config configs/interpolant.ini
python train_unified.py --model dit --config configs/dit.ini
```
