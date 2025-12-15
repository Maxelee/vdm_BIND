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

### Issue #4: Add DiT (Diffusion Transformer) backbone option
**Labels:** `enhancement`, `architecture`
**Branch:** `feature/dit-backbone`
**Status:** NOT STARTED

**Description:**
Add support for Diffusion Transformer architecture as an alternative to UNet:
- Implement DiT blocks with adaptive layer norm
- Add config option `architecture = unet|dit`
- Benchmark against UNet on same training data
- Document memory/compute requirements

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

### Issue #6: Add model ensemble support
**Labels:** `enhancement`
**Branch:** `feature/ensemble`
**Status:** NOT STARTED

**Description:**
Allow combining predictions from multiple models:
- Ensemble of different model types (VDM + Interpolant)
- Ensemble of same model with different seeds
- Weighted averaging or learned combination

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

### Issue #16: Standardized benchmark suite
**Labels:** `enhancement`, `analysis`
**Branch:** `feature/benchmark-suite`
**Status:** NOT STARTED

**Description:**
Create standardized evaluation:
- Power spectrum ratio at fixed k values
- SSIM distribution statistics
- Integrated mass scatter
- Inference time benchmarks
- Compare all 8 models consistently

---

### Issue #17: Add uncertainty quantification
**Labels:** `enhancement`, `science`
**Branch:** `feature/uncertainty`
**Status:** NOT STARTED

**Description:**
Quantify prediction uncertainty:
- Multi-realization variance
- Per-pixel uncertainty maps
- Ensemble disagreement
- Calibration analysis

---

## 📋 Summary Table

### ✅ Completed Issues (7/17)

| Issue | Description | Status |
|-------|-------------|--------|
| #1 | Installation docs & requirements.txt | ✅ DONE |
| #2 | Sync documentation files | ✅ DONE |
| #7 | Abstract data interface | ✅ DONE |
| #8 | Flexible parameter conditioning | ✅ DONE |
| #9 | Auto-normalization script | ✅ DONE |
| #14 | CI/CD pipeline | ✅ DONE |
| #15 | Synthetic data tests | ✅ DONE |

### 🔄 Remaining Issues (10/17)

| Issue | Priority | Effort | Impact | Category |
|-------|----------|--------|--------|----------|
| #3 API documentation | 🟡 Medium | Medium | High | Docs |
| #4 DiT backbone | 🟡 Medium | Large | Medium | Architecture |
| #5 FNO backbone | 🟢 Low | Large | Medium | Architecture |
| #6 Model ensemble | 🟡 Medium | Medium | Medium | Architecture |
| #10 Data converters | 🟡 Medium | Medium | High | Generalization |
| #11 3D optimization | 🔴 High | Large | High | Performance |
| #12 Distributed inference | 🟢 Low | Large | Medium | Performance |
| #13 Model export (ONNX) | 🟢 Low | Medium | Medium | Deployment |
| #16 Benchmark suite | 🟡 Medium | Medium | High | Analysis |
| #17 Uncertainty quantification | 🔴 High | Medium | High | Science |

---

## 🎯 Recommended Next Steps

Based on impact and effort, here are the recommended next issues to tackle:

1. **Issue #17: Uncertainty Quantification** - High scientific value, medium effort
2. **Issue #16: Benchmark Suite** - Important for paper/validation
3. **Issue #3: API Documentation** - Improves usability significantly
4. **Issue #10: Data Converters** - Enables broader adoption

---

## Suggested Branch Strategy

1. **main** - Stable releases
2. **develop** - Integration branch
3. **feature/*** - New features
4. **fix/*** - Bug fixes
5. **docs/*** - Documentation updates
6. **experiment/*** - Research experiments
