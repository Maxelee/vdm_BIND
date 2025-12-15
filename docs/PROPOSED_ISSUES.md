# Proposed GitHub Issues for VDM-BIND

This document outlines proposed issues to improve VDM-BIND's usability, generalizability, and performance.

---

## 🏷️ Documentation & Usability

### Issue #1: Create requirements.txt and improve installation docs
**Labels:** `documentation`, `good first issue`
**Branch:** `docs/installation-improvements`

**Description:**
- ✅ Add `requirements.txt` file (DONE)
- Add installation instructions for pylians3 (MAS_library, Pk_library)
- Add Docker/Singularity container option for reproducibility
- Test installation on fresh environment

---

### Issue #2: Sync all documentation files
**Labels:** `documentation`
**Branch:** `docs/sync-readmes`

**Description:**
README files have inconsistencies:
- [x] Main README references 8 models, MODEL_COMPARISON.md said 6 (FIXED)
- [x] Notebook README had outdated structure (FIXED)
- [ ] Add parameter count clarification (15 vs 35 params)
- [ ] Update citation placeholders when paper is published
- [ ] Add CONTRIBUTING.md for collaborators

---

### Issue #3: Add comprehensive API documentation
**Labels:** `documentation`, `enhancement`
**Branch:** `docs/api-reference`

**Description:**
- Add docstrings to all public functions
- Generate Sphinx/MkDocs API reference
- Add usage examples in docstrings
- Create "Quick Start" guide for common workflows

---

## 🔬 Architecture Improvements

### Issue #4: Add DiT (Diffusion Transformer) backbone option
**Labels:** `enhancement`, `architecture`
**Branch:** `feature/dit-backbone`

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

**Description:**
FNO could be well-suited for cosmological data as it learns in frequency domain:
- Implement FNO layers
- Compare with UNet on power spectrum recovery
- May naturally handle multi-scale structure

---

### Issue #6: Add model ensemble support
**Labels:** `enhancement`
**Branch:** `feature/ensemble`

**Description:**
Allow combining predictions from multiple models:
- Ensemble of different model types (VDM + Interpolant)
- Ensemble of same model with different seeds
- Weighted averaging or learned combination

---

## 🌍 Generalization

### Issue #7: Abstract simulation data interface
**Labels:** `enhancement`, `generalization`
**Branch:** `feature/data-interface`
**Priority:** HIGH

**Description:**
Create abstract interfaces to support arbitrary simulations:
```python
class SimulationLoader(ABC):
    def load_particles(self, ptype) -> Tuple[positions, masses]
    def get_box_size(self) -> float
    def get_cosmology(self) -> Optional[Dict]

class HaloCatalogLoader(ABC):
    def load_halos(self, mass_threshold) -> Tuple[pos, mass, radii]
```

Implement for:
- [x] CAMELS (existing)
- [ ] Illustris/TNG direct
- [ ] SIMBA
- [ ] User-provided format

---

### Issue #8: Flexible parameter conditioning
**Labels:** `enhancement`, `generalization`
**Branch:** `feature/flexible-params`

**Description:**
Currently hardcoded to 35 CAMELS parameters. Make flexible:
- Support 0 parameters (unconditional generation)
- Support arbitrary number of user-defined parameters
- Config-driven parameter specification
- Document how to add custom parameters

---

### Issue #9: On-the-fly normalization computation script
**Labels:** `enhancement`, `usability`
**Branch:** `feature/auto-normalization`

**Description:**
Create script to compute normalization stats from user data:
```bash
python scripts/compute_normalization.py \
    --data_dir /path/to/training_data \
    --output data/custom_norm_stats.npz
```
- Compute mean/std for each field
- Optional quantile transformer fitting
- Validate stats are reasonable

---

### Issue #10: Data format converter scripts
**Labels:** `enhancement`, `usability`
**Branch:** `feature/data-converters`

**Description:**
Scripts to convert various simulation formats to BIND format:
- `convert_illustris.py`
- `convert_simba.py`
- `convert_generic_hdf5.py`

---

## ⚡ Performance

### Issue #11: Add 3D support with memory optimization
**Labels:** `enhancement`, `performance`
**Branch:** `feature/3d-optimized`

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

**Description:**
For applying BIND to large cosmological volumes (>500 Mpc):
- Implement MPI-based distributed inference
- Domain decomposition with ghost zones
- Aggregate results across ranks

---

### Issue #13: ONNX/TensorRT export for deployment
**Labels:** `enhancement`, `deployment`
**Branch:** `feature/model-export`

**Description:**
Export trained models for fast inference:
- ONNX export
- TensorRT conversion
- Benchmark speedup
- Document deployment workflow

---

## 🧪 Testing & CI

### Issue #14: Add CI/CD pipeline
**Labels:** `infrastructure`
**Branch:** `ci/github-actions`

**Description:**
- Add GitHub Actions for automated testing
- Run tests on PR
- Coverage reporting
- Linting (black, flake8)

---

### Issue #15: Add integration tests with synthetic data
**Labels:** `testing`
**Branch:** `test/synthetic-data`

**Description:**
Create small synthetic dataset for full pipeline testing:
- ~10 halos with known properties
- Include ground truth
- Fast to run (<1 min)
- Test full BIND pipeline end-to-end

---

## 📊 Analysis & Evaluation

### Issue #16: Standardized benchmark suite
**Labels:** `enhancement`, `analysis`
**Branch:** `feature/benchmark-suite`

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

**Description:**
Quantify prediction uncertainty:
- Multi-realization variance
- Per-pixel uncertainty maps
- Ensemble disagreement
- Calibration analysis

---

## 📋 Summary Table

| Issue | Priority | Effort | Impact |
|-------|----------|--------|--------|
| #7 Abstract data interface | 🔴 High | Large | High |
| #8 Flexible params | 🔴 High | Medium | High |
| #9 Auto-normalization | 🟡 Medium | Small | High |
| #4 DiT backbone | 🟡 Medium | Large | Medium |
| #14 CI/CD | 🟡 Medium | Medium | Medium |
| #2 Sync docs | 🟢 Low | Small | Medium |
| #13 Model export | 🟢 Low | Medium | Medium |

---

## Suggested Branch Strategy

1. **main** - Stable releases
2. **develop** - Integration branch
3. **feature/*** - New features
4. **fix/*** - Bug fixes
5. **docs/*** - Documentation updates
6. **experiment/*** - Research experiments
