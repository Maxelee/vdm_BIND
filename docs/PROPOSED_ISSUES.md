# Proposed GitHub Issues for VDM-BIND

This document outlines proposed issues to improve VDM-BIND's usability, generalizability, and performance.

---

## ✅ COMPLETED ISSUES

### ~~Issue #1: Create requirements.txt and improve installation docs~~ ✅
**Status:** COMPLETED (merged to main)
**Branch:** `docs/installation-improvements`
- ✅ Added `requirements.txt` file
- ✅ Added `CONTRIBUTING.md`
- ✅ Enhanced installation documentation

---

### ~~Issue #2: Sync all documentation files~~ ✅
**Status:** COMPLETED (merged to main)
**Branch:** `docs/sync-readmes`
- ✅ Fixed 8 models inconsistency in MODEL_COMPARISON.md
- ✅ Updated notebook README structure
- ✅ Added parameter count clarification (6 base + up to 29 derived = 35 total)

---

### ~~Issue #7: Abstract simulation data interface~~ ✅
**Status:** COMPLETED (merged to main)
**Branch:** `feature/data-interface`
- ✅ Created `vdm/data_interface.py` with abstract base classes
- ✅ Implemented `SimulationLoader`, `HaloCatalogLoader` ABCs
- ✅ Added support for SubFind, Rockstar, CSV halo catalogs
- ✅ Created example implementations for CAMELS

---

### ~~Issue #8: Flexible parameter conditioning~~ ✅
**Status:** COMPLETED (merged to main)
**Branch:** `feature/flexible-params`
- ✅ Support 0 parameters (unconditional generation)
- ✅ Support arbitrary N parameters via config
- ✅ Added CSV/JSON param normalization loading
- ✅ Added example configs and documentation

---

### ~~Issue #9: On-the-fly normalization computation script~~ ✅
**Status:** COMPLETED (merged to main)
**Branch:** `feature/auto-normalization`
- ✅ Created `scripts/compute_normalization.py`
- ✅ Computes mean/std for each field type
- ✅ Supports HDF5 and NPZ input formats

---

### ~~Issue #14: Add CI/CD pipeline~~ ✅
**Status:** COMPLETED (merged to main)
**Branch:** `ci/github-actions`
- ✅ Added `.github/workflows/ci.yml` for PR testing
- ✅ Added `.github/workflows/release.yml` for releases
- ✅ Includes linting, testing, coverage reporting

---

### ~~Issue #15: Add integration tests with synthetic data~~ ✅
**Status:** COMPLETED (merged to main)
**Branch:** `test/synthetic-data`
- ✅ Created `scripts/generate_synthetic_data.py`
- ✅ Created `tests/test_synthetic.py` with pipeline tests
- ✅ Tests UNet, VDM, Interpolant forward passes

---

## 🏷️ REMAINING: Documentation & Usability

### Issue #3: Add comprehensive API documentation
**Labels:** `documentation`, `enhancement`
**Branch:** `docs/api-reference`
**Status:** NOT STARTED

**Description:**
- Add docstrings to all public functions
- Generate Sphinx/MkDocs API reference
- Add usage examples in docstrings
- Create "Quick Start" guide for common workflows

---

## 🔬 REMAINING: Architecture Improvements

### ~~Issue #4: Add DiT (Diffusion Transformer) backbone option~~ ✅
**Labels:** `enhancement`, `architecture`
**Branch:** `feature/dit-backbone`
**Status:** COMPLETED (merged to main)

**Description:**
Add support for Diffusion Transformer architecture as an alternative to UNet:
- ✅ Implemented DiT blocks with adaptive layer norm (adaLN-Zero)
- ✅ Added `vdm/dit.py` with full DiT architecture
- ✅ Added `vdm/dit_model.py` (LightDiTVDM Lightning wrapper)
- ✅ Created `configs/dit.ini` for DiT training
- ✅ Added `dit` to train_unified.py MODEL_TYPES
- ✅ 22 unit tests in `tests/test_dit.py`
- ✅ Updated MODEL_COMPARISON.md with DiT documentation

**Features:**
- Patch-based transformer with 2D sinusoidal position embeddings
- adaLN-Zero conditioning on timestep + parameter conditioning
- Cross-attention for spatial conditioning (DM fields)
- Pre-defined variants: DiT-S (384d/12L), DiT-B (768d/12L), DiT-L (1024d/24L), DiT-XL (1152d/28L)

**References:**
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

---

### Issue #5: Implement Fourier Neural Operator (FNO) option
**Labels:** `enhancement`, `architecture`, `physics-informed`
**Branch:** `feature/fno-backbone`
**Status:** NOT STARTED

**Description:**
FNO could be well-suited for cosmological data as it learns in frequency domain:
- Implement FNO layers
- Compare with UNet on power spectrum recovery
- May naturally handle multi-scale structure

---

### ~~Issue #6: Add model ensemble support~~ ✅
**Labels:** `enhancement`
**Branch:** `feature/uncertainty-benchmark-ensemble`
**Status:** COMPLETED (merged to main)

**Description:**
Allow combining predictions from multiple models:
- ✅ ModelEnsemble: Simple averaging of multiple models
- ✅ WeightedEnsemble: Learnable or fixed per-model weights
- ✅ ChannelWiseEnsemble: Per-channel weighting for specialized models
- ✅ DiversityEnsemble: Promotes diverse predictions
- ✅ create_ensemble_from_checkpoints(): Load from checkpoint files
- ✅ create_multi_seed_ensemble(): Combine models from different seeds

---

## 🌍 REMAINING: Generalization

### Issue #10: Data format converter scripts
**Labels:** `enhancement`, `usability`
**Branch:** `feature/data-converters`
**Status:** NOT STARTED

**Description:**
Scripts to convert various simulation formats to BIND format:
- `convert_illustris.py`
- `convert_simba.py`
- `convert_generic_hdf5.py`

---

## ⚡ REMAINING: Performance

### Issue #11: Add 3D support with memory optimization
**Labels:** `enhancement`, `performance`
**Branch:** `feature/3d-optimized`
**Status:** NOT STARTED

**Description:**
Current 3D support is memory-limited. Improvements:
- Implement patch-based 3D processing
- Add gradient checkpointing
- Support mixed precision (bfloat16)
- Benchmark memory usage

---

### Issue #12: Distributed inference for large volumes
**Labels:** `enhancement`, `performance`
**Branch:** `feature/distributed-inference`
**Status:** NOT STARTED

**Description:**
For applying BIND to large cosmological volumes (>500 Mpc):
- Implement MPI-based distributed inference
- Domain decomposition with ghost zones
- Aggregate results across ranks

---

### Issue #13: ONNX/TensorRT export for deployment
**Labels:** `enhancement`, `deployment`
**Branch:** `feature/model-export`
**Status:** NOT STARTED

**Description:**
Export trained models for fast inference:
- ONNX export
- TensorRT conversion
- Benchmark speedup
- Document deployment workflow

---

## 📊 REMAINING: Analysis & Evaluation

### ~~Issue #16: Standardized benchmark suite~~ ✅
**Labels:** `enhancement`, `analysis`
**Branch:** `feature/uncertainty-benchmark-ensemble`
**Status:** COMPLETED (merged to main)

**Description:**
Create standardized evaluation:
- ✅ BenchmarkSuite class for consistent model comparison
- ✅ Pixel metrics: MSE, RMSE, MAE, correlation, SSIM
- ✅ Power spectrum metrics: ratio, correlation at fixed k
- ✅ Mass metrics: bias, scatter
- ✅ Timing: inference time, throughput
- ✅ Results export to JSON
- ✅ quick_benchmark() for rapid iteration

---

### ~~Issue #17: Add uncertainty quantification~~ ✅
**Labels:** `enhancement`, `science`
**Branch:** `feature/uncertainty-benchmark-ensemble`
**Status:** COMPLETED (merged to main)

**Description:**
Quantify prediction uncertainty:
- ✅ UncertaintyEstimator: Multi-realization sampling
- ✅ MC Dropout support for approximate Bayesian inference
- ✅ EnsembleUncertainty: Uncertainty from model ensembles
- ✅ Calibration analysis: coverage, ECE, reliability diagrams
- ✅ Uncertainty maps: std, variance, IQR, entropy methods

---

## 📋 Summary Table

### ✅ Completed Issues (11/17)

| Issue | Description | Status |
|-------|-------------|--------|
| #1 | Installation docs & requirements.txt | ✅ DONE |
| #2 | Sync documentation files | ✅ DONE |
| #4 | DiT (Diffusion Transformer) backbone | ✅ DONE |
| #6 | Model ensemble support | ✅ DONE |
| #7 | Abstract data interface | ✅ DONE |
| #8 | Flexible parameter conditioning | ✅ DONE |
| #9 | Auto-normalization script | ✅ DONE |
| #14 | CI/CD pipeline | ✅ DONE |
| #15 | Synthetic data tests | ✅ DONE |
| #16 | Standardized benchmark suite | ✅ DONE |
| #17 | Uncertainty quantification | ✅ DONE |

### 🔄 Remaining Issues (6/17)

| Issue | Priority | Effort | Impact | Category |
|-------|----------|--------|--------|----------|
| #3 API documentation | 🟡 Medium | Medium | High | Docs |
| #5 FNO backbone | 🟢 Low | Large | Medium | Architecture |
| #10 Data converters | 🟡 Medium | Medium | High | Generalization |
| #11 3D optimization | 🔴 High | Large | High | Performance |
| #12 Distributed inference | 🟢 Low | Large | Medium | Performance |
| #13 Model export (ONNX) | 🟢 Low | Medium | Medium | Deployment |

---

## 🎯 Recommended Next Steps

Based on impact and effort, here are the recommended next issues to tackle:

1. **Issue #11: 3D Optimization** - High impact for large simulations
2. **Issue #3: API Documentation** - Improves usability significantly
3. **Issue #10: Data Converters** - Enables broader adoption
4. **Issue #5: FNO Backbone** - Physics-informed architecture

---

## Suggested Branch Strategy

1. **main** - Stable releases
2. **develop** - Integration branch
3. **feature/*** - New features
4. **fix/*** - Bug fixes
5. **docs/*** - Documentation updates
6. **experiment/*** - Research experiments
