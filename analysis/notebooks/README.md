# BIND Paper Analysis Notebooks

This directory contains publication-ready Jupyter notebooks for analyzing BIND (Baryonic Inference from N-body Dark matter) model performance.

## Overview

These notebooks are refactored from the original `Paper_plots.ipynb` to provide cleaner, modular, and reproducible analyses. Each notebook focuses on a specific aspect of the BIND evaluation.

## Directory Structure

```
paper_notebooks/
├── README.md                        # This file
├── paper_utils.py                   # Shared utility functions
├── 01_bind_overview.ipynb           # Data visualization and BIND process overview
├── 02_power_spectrum_analysis.ipynb # Power spectrum ratio S(k) analysis
├── 03_density_profile_analysis.ipynb# Radial density profile residuals
├── 04_integrated_mass_analysis.ipynb# Integrated mass within R_vir multiples
└── 05_ssim_analysis.ipynb           # Structural similarity (SSIM) metrics
```

## Notebook Descriptions

### 1. BIND Overview (`01_bind_overview.ipynb`)
- **Purpose**: Introduction to the BIND methodology and data visualization
- **Contents**:
  - Data path configuration
  - Loading simulation maps (DMO and Hydro)
  - Visualization of full simulation boxes
  - Halo cutout comparison (DMO vs Hydro vs BIND-generated)
  - Side-by-side comparisons of input/output/target

### 2. Power Spectrum Analysis (`02_power_spectrum_analysis.ipynb`)
- **Purpose**: Quantify BIND accuracy in Fourier space
- **Contents**:
  - Power spectrum computation using `Pk_library`
  - Ratio analysis: $S(k) = P_{\text{gen}}(k) / P_{\text{DMO}}(k)$
  - Analysis across CV, 1P, and SB35 datasets
  - Parameter correlation with power spectrum accuracy
  - Visualization of mean ± scatter across simulations

### 3. Density Profile Analysis (`03_density_profile_analysis.ipynb`)
- **Purpose**: Evaluate radial structure reproduction
- **Contents**:
  - Azimuthally-averaged density profiles
  - Residual analysis: $(\rho_{\text{gen}} - \rho_{\text{hydro}}) / \rho_{\text{hydro}}$
  - Profile comparison from 0.1× to 3× $R_{200c}$
  - Channel-wise analysis (DM, Gas, Stars)
  - Correlation with astrophysical/cosmological parameters

### 4. Integrated Mass Analysis (`04_integrated_mass_analysis.ipynb`)
- **Purpose**: Test mass conservation within apertures
- **Contents**:
  - Integrated mass within 1×, 2×, 3× $R_{\text{vir}}$
  - Scatter plots: predicted vs true mass
  - Residual distributions
  - Mass-dependent accuracy analysis

### 5. SSIM Analysis (`05_ssim_analysis.ipynb`)
- **Purpose**: Perceptual quality assessment
- **Contents**:
  - Structural Similarity Index Measure (SSIM) computation
  - Distribution analysis across datasets
  - Channel-wise SSIM comparison
  - Statistical summary tables

## Utility Module (`paper_utils.py`)

Shared functions used across all notebooks:

| Function | Description |
|----------|-------------|
| `setup_plotting_style()` | Configure matplotlib for publication quality |
| `compute_power(field, box_size)` | Compute 2D power spectrum |
| `compute_binded_power(field, box_size)` | Power spectrum with binning |
| `get_projected_surface_density(cutout, R200c)` | Radial density profiles |
| `compute_integrated_mass(cutout, apertures)` | Mass within apertures |
| `compute_ssim_for_dataset(hydro, generated)` | SSIM computation |

## Data Requirements

These notebooks expect data in the following structure:

```
/mnt/home/mlee1/ceph/BIND2d_new/
├── CV/
│   ├── maps/hydro/sim_{idx}/maps_099.npy
│   ├── maps/DMO/sim_{idx}/maps_099.npy
│   ├── cutouts/hydro/sim_{idx}/cutouts_099.npy
│   ├── cutouts/DMO/sim_{idx}/cutouts_099.npy
│   └── generated_halos/sim_{idx}/generated_halos.npy
├── 1P/
│   └── ... (same structure, 66 simulations)
└── SB35/
    └── ... (same structure, 1000 simulations)
```

## Datasets

| Dataset | Simulations | Description |
|---------|-------------|-------------|
| CV | 25 | Cosmic Variance - same cosmology, different ICs |
| 1P | 66 | Single Parameter variations |
| SB35 | 1000 | Latin Hypercube sampling of 35 parameters |

## Usage

1. Ensure all dependencies are installed:
   ```bash
   pip install numpy matplotlib scipy scikit-image Pk_library pandas
   ```

2. Run notebooks in order (1→5) for complete analysis

3. Each notebook saves figures as PDF files in the current directory

## Output Files

Each notebook generates publication-ready figures:

- `01_bind_overview.ipynb` → `simulation_maps.pdf`, `halo_comparison.pdf`
- `02_power_spectrum_analysis.ipynb` → `power_spectrum_*.pdf`, `parameter_correlation_*.pdf`
- `03_density_profile_analysis.ipynb` → `density_profiles_*.pdf`, `profile_residuals_*.pdf`
- `04_integrated_mass_analysis.ipynb` → `integrated_mass_*.pdf`, `mass_scatter_*.pdf`
- `05_ssim_analysis.ipynb` → `ssim_distributions.pdf`, `ssim_comparison_bar.pdf`

## Citation

If you use these analyses, please cite the BIND paper:
```
[Citation information to be added upon publication]
```

## Model Types

These notebooks support multiple model types for the DMO → Hydro mapping:

| Model Type | Description | Config Example |
|------------|-------------|----------------|
| VDM (clean) | Variational Diffusion Model (3-channel) | `clean_vdm_aggressive_stellar.ini` |
| VDM (triple) | Three independent 1-channel VDMs | `clean_vdm_triple.ini` |
| DDPM | Score-based diffusion (NCSNpp) | `ddpm.ini` |
| **Interpolant** | Flow matching / stochastic interpolants | `interpolant.ini` |

The interpolant models learn a velocity field to transport from x_0 to x_1, using flow matching loss instead of diffusion objectives. They typically require fewer sampling steps (20-50 vs 250-1000) and have simpler loss functions.

## Notes

- All plots use consistent styling defined in `paper_utils.py`
- LaTeX rendering is enabled for mathematical expressions in plots
- Figure sizes are optimized for typical journal column widths
- Color schemes are colorblind-friendly where possible

