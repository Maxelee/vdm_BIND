"""
BIND Paper Analysis Utilities
=============================

This module provides reusable functions for analyzing BIND simulation results.
Functions are organized into logical sections:

1. Configuration & Plotting Setup
2. Data Loading
3. Power Spectrum Computation
4. Density Profile Analysis
5. Integrated Mass Computation
6. Statistical Analysis (SSIM, Correlations)

Author: Paper Analysis Refactoring
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
import glob
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter

# ============================================================================
# SECTION 1: Configuration & Plotting Setup
# ============================================================================

def setup_plotting_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


# Simulation box parameters
BOX_SIZE = 50.0  # Mpc/h
GRID_SIZE = 1024  # pixels
PIXEL_TO_MPC = BOX_SIZE / GRID_SIZE

# Model configuration - change this to switch between different trained models
# MODEL_NAME = 'clean_vdm_aggressive_stellar'
MODEL_NAME = 'clean_vdm_aggressive_stellar_nofocus'

# Channel names for output fields
CHANNEL_NAMES = ['Hydro DM', 'Gas', 'Star']
CHANNEL_LABELS = ['Hydro DM', 'Gas', 'Star', 'Total']

# LaTeX-formatted parameter names
PARAM_LATEX_NAMES = {
    'Omega0': r'$\Omega_0$',
    'OmegaBaryon': r'$\Omega_b$',
    'OmegaLambda': r'$\Omega_\Lambda$',
    'HubbleParam': r'$h$',
    'Sigma8': r'$\sigma_8$',
    'sigma8': r'$\sigma_8$',
    'PrimordialIndex': r'$n_s$',
}


# ============================================================================
# SECTION 2: Data Loading
# ============================================================================

def load_cv_simulation(sim_num, base_path='/mnt/home/mlee1/ceph/BIND2d_new/CV', model_name=None):
    """
    Load data from a CV (cosmic variance) simulation.
    
    Parameters
    ----------
    sim_num : int
        Simulation number (0-26, excluding 17)
    base_path : str
        Base path to CV simulations
    model_name : str, optional
        Model name to use. If None, uses global MODEL_NAME.
        
    Returns
    -------
    dict : Dictionary containing loaded arrays
    """
    if model_name is None:
        model_name = MODEL_NAME
        
    snap_path = f'{base_path}/sim_{sim_num}/snap_90'
    halo_path = f'{snap_path}/mass_threshold_13'
    gen_path = f'{halo_path}/{model_name}'
    
    data = {
        'full_dm': np.load(f'{snap_path}/sim_grid.npy'),
        'full_hydro': np.load(f'{snap_path}/full_hydro.npy'),
        'hydro_cutouts': np.load(f'{halo_path}/hydro_cutouts.npy'),
        'dmo_cutouts': np.load(f'{halo_path}/dmo_cutouts.npy'),
        'halo_metadata': np.load(f'{halo_path}/halo_metadata.npz', allow_pickle=True),
        'generated_halos': np.load(f'{gen_path}/generated_halos.npz', allow_pickle=True),
    }
    
    # Load BIND final maps (10 realizations)
    data['binded_maps'] = [
        np.load(f'{gen_path}/ue_1/final_map_{i}.npy') 
        for i in range(10)
    ]
    
    return data


def load_1p_simulation(sim_name, base_path='/mnt/home/mlee1/ceph/BIND2d_new/1P', model_name=None):
    """
    Load data from a 1P (one-parameter variation) simulation.
    
    Parameters
    ----------
    sim_name : str
        Simulation name (e.g., '1P_p1_n2')
    base_path : str
        Base path to 1P simulations
    model_name : str, optional
        Model name to use. If None, uses global MODEL_NAME.
        
    Returns
    -------
    dict : Dictionary containing loaded arrays
    """
    if model_name is None:
        model_name = MODEL_NAME
        
    snap_path = f'{base_path}/{sim_name}/snap_90'
    halo_path = f'{snap_path}/mass_threshold_13'
    gen_path = f'{halo_path}/{model_name}'
    
    # Load projected images from different location
    proj_data = np.load(f'/mnt/home/mlee1/ceph/train_data_1024/projected_images_1P/projections_xy_{sim_name}.npz')
    
    data = {
        'full_dm': np.load(f'{snap_path}/sim_grid.npy'),
        'full_hydro': proj_data['hydro_dm'] + proj_data['gas'] + proj_data['star'],
        'hydro_cutouts': np.load(f'{halo_path}/hydro_cutouts.npy'),
        'halo_metadata': np.load(f'{halo_path}/halo_metadata.npz', allow_pickle=True),
        'generated_halos': np.load(f'{gen_path}/generated_halos.npz', allow_pickle=True),
    }
    
    data['binded_maps'] = [
        np.load(f'{gen_path}/ue_1/final_map_{i}.npy') 
        for i in range(10)
    ]
    
    return data


def load_sb35_simulation(sim_num, base_path='/mnt/home/mlee1/ceph/BIND2d_new/SB35', model_name=None):
    """
    Load data from an SB35 (Sobol sequence) simulation.
    
    Parameters
    ----------
    sim_num : int
        Simulation number
    base_path : str
        Base path to SB35 simulations
    model_name : str, optional
        Model name to use. If None, uses global MODEL_NAME.
        
    Returns
    -------
    dict : Dictionary containing loaded arrays
    """
    if model_name is None:
        model_name = MODEL_NAME
        
    snap_path = f'{base_path}/sim_{sim_num}/snap_90'
    halo_path = f'{snap_path}/mass_threshold_13'
    gen_path = f'{halo_path}/{model_name}'
    
    # Load projected images from different location
    proj_data = np.load(f'/mnt/home/mlee1/ceph/train_data_1024/projected_images/projections_xy_{sim_num}.npz')
    
    data = {
        'full_dm': np.load(f'{snap_path}/sim_grid.npy'),
        'full_hydro': proj_data['hydro_dm'] + proj_data['gas'] + proj_data['star'],
        'hydro_cutouts': np.load(f'{halo_path}/hydro_cutouts.npy'),
        'halo_metadata': np.load(f'{halo_path}/halo_metadata.npz', allow_pickle=True),
        'generated_halos': np.load(f'{gen_path}/generated_halos.npz', allow_pickle=True),
    }
    
    data['binded_maps'] = [
        np.load(f'{gen_path}/ue_1/final_map_{i}.npy') 
        for i in range(10)
    ]
    
    return data


def load_hydro_replace(dataset, sim_id):
    """
    Load hydro-replacement map for a simulation.
    
    Parameters
    ----------
    dataset : str
        Dataset name ('CV', '1P', or 'SB35')
    sim_id : int or str
        Simulation identifier
        
    Returns
    -------
    np.ndarray : Hydro-replaced map
    """
    if dataset == 'CV':
        path = f'/mnt/home/mlee1/ceph/BIND2d/hydro_replace/CV/sim_{sim_id}/hydro_replace/final_map_hydro_replace.npy'
    elif dataset == '1P':
        path = f'/mnt/home/mlee1/ceph/BIND2d/hydro_replace/1P/{sim_id}/hydro_replace/final_map_hydro_replace.npy'
    elif dataset == 'SB35':
        path = f'/mnt/home/mlee1/ceph/BIND2d/hydro_replace/SB35/sim_{sim_id}/hydro_replace/final_map_hydro_replace.npy'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return np.load(path)


def load_1p_params():
    """Load 1P parameter file and return organized parameter data."""
    param_file = '/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt'
    oneP_params = pd.read_csv(param_file, delim_whitespace=True)
    
    names = oneP_params['#Name'].to_list()[1:]
    param_array = oneP_params.iloc[1:, 1:-1].values
    fiducial_params = oneP_params.iloc[0, 1:-1].values
    
    return oneP_params, names, param_array, fiducial_params


def load_sb35_metadata():
    """Load SB35 parameter metadata and simulation list."""
    metadata = pd.read_csv('/mnt/home/mlee1/50Mpc_boxes/data/param_df.csv')
    min_max_metadata = pd.read_csv('/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv')
    
    # Use BIND2d_new which contains the full test set (~100 sims)
    sims_paths = sorted(glob.glob('/mnt/home/mlee1/ceph/BIND2d_new/SB35/*'))
    sim_nums = sorted([int(sp.split('sim_')[-1]) for sp in sims_paths])
    
    return metadata, min_max_metadata, sim_nums


# ============================================================================
# SECTION 3: Power Spectrum Computation
# ============================================================================

def compute_overdensity(field):
    """
    Compute overdensity field δ = ρ/ρ̄ - 1.
    
    Parameters
    ----------
    field : np.ndarray
        2D density field
        
    Returns
    -------
    np.ndarray : Overdensity field
    """
    field = field.astype(np.float64)
    delta = field / np.mean(field)
    delta -= 1.0
    return delta.astype(np.float32)


def compute_power_spectrum(field, box_size=BOX_SIZE, mas='CIC'):
    """
    Compute 2D power spectrum of a field.
    
    Parameters
    ----------
    field : np.ndarray
        2D density field
    box_size : float
        Box size in Mpc/h
    mas : str
        Mass assignment scheme
        
    Returns
    -------
    tuple : (k, Pk, Nmodes)
    """
    import Pk_library as PKL
    
    delta = compute_overdensity(field)
    Pk2D = PKL.Pk_plane(delta, box_size, mas, threads=0, verbose=False)
    
    return Pk2D.k, Pk2D.Pk, Pk2D.Nmodes


def compute_power_spectra_trio(full_hydro, full_dm, full_hydro_replace):
    """
    Compute power spectra for hydro, DMO, and hydro-replace fields.
    
    Returns
    -------
    tuple : (k, Pk_hydro, Pk_dmo, Pk_replace, Nmodes)
    """
    delta_hydro = compute_overdensity(full_hydro)
    delta_dmo = compute_overdensity(full_dm)
    delta_replace = compute_overdensity(full_hydro_replace)
    
    import Pk_library as PKL
    
    Pk2D_hydro = PKL.Pk_plane(delta_hydro, BOX_SIZE, 'CIC', threads=0, verbose=False)
    Pk2D_dmo = PKL.Pk_plane(delta_dmo, BOX_SIZE, 'CIC', threads=0, verbose=False)
    Pk2D_replace = PKL.Pk_plane(delta_replace, BOX_SIZE, 'CIC', threads=0, verbose=False)
    
    return (Pk2D_hydro.k, Pk2D_hydro.Pk, Pk2D_dmo.Pk, 
            Pk2D_replace.Pk, Pk2D_hydro.Nmodes)


def compute_binded_power_spectra(binded_maps):
    """
    Compute power spectra for multiple BIND realizations.
    
    Parameters
    ----------
    binded_maps : list of np.ndarray
        List of BIND output maps (10 realizations)
        
    Returns
    -------
    tuple : (k, Pk_array, Nmodes) where Pk_array has shape (n_realizations, n_k)
    """
    import Pk_library as PKL
    
    binded = np.array(binded_maps, dtype=np.float64)
    delta_binded = binded / np.mean(binded, axis=(1, 2), keepdims=True) - 1.0
    delta_binded = delta_binded.astype(np.float32)
    
    Pk2D_list = [PKL.Pk_plane(delta_binded[i], BOX_SIZE, 'CIC', threads=0, verbose=False)
                 for i in range(len(binded))]
    
    k = Pk2D_list[0].k
    Nmodes = Pk2D_list[0].Nmodes
    Pk_array = np.array([pk.Pk for pk in Pk2D_list])
    
    return k, Pk_array, Nmodes


# ============================================================================
# SECTION 4: Density Profile Analysis
# ============================================================================

def compute_surface_density_profile(halo_mass, radius_pix, size=128, nbins=15):
    """
    Calculate projected surface density profile in logarithmic radial bins out to 3×R200.
    
    Parameters
    ----------
    halo_mass : np.ndarray
        2D mass map of halo cutout
    radius_pix : float
        R200 radius in pixels
    size : int
        Cutout size in pixels
    nbins : int
        Number of radial bins
        
    Returns
    -------
    tuple : (surface_densities, bin_centers) where bin_centers are in units of R200
    """
    # Ensure halo_mass is 2D
    if halo_mass.ndim == 1:
        sqrt_size = int(np.sqrt(len(halo_mass)))
        if sqrt_size * sqrt_size == len(halo_mass):
            halo_mass = halo_mass.reshape(sqrt_size, sqrt_size)
            size = sqrt_size
        else:
            raise ValueError(f"Cannot reshape 1D array of length {len(halo_mass)} to 2D")
    
    if halo_mass.shape[0] != size or halo_mass.shape[1] != size:
        size = halo_mass.shape[0]
    
    # Radial bins from 0.1×R200 to 3×R200 in log space
    radial_bins = np.logspace(np.log10(0.1), np.log10(3), nbins)
    bin_centers = np.sqrt(radial_bins[:-1] * radial_bins[1:])
    
    # Calculate annular areas in physical units (Mpc²)
    annular_areas = np.pi * (radial_bins[1:]**2 - radial_bins[:-1]**2) * (radius_pix * PIXEL_TO_MPC)**2
    
    # Create distance map from center
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Calculate surface density in each annular bin
    surface_densities = np.zeros(len(radial_bins) - 1)
    for i in range(len(radial_bins) - 1):
        mask = (dist >= radial_bins[i] * radius_pix) & (dist < radial_bins[i+1] * radius_pix)
        mass = halo_mass[mask]
        if annular_areas[i] > 0 and len(mass) > 0:
            surface_densities[i] = mass.sum() / annular_areas[i]
        else:
            surface_densities[i] = np.nan
    
    return surface_densities, bin_centers


def compute_integrated_mass(halo_mass, radius_pix, size=128, r_multiples=[1, 2, 3]):
    """
    Calculate total integrated mass within circular apertures of radius r × R_vir.
    
    Parameters
    ----------
    halo_mass : np.ndarray
        2D mass map
    radius_pix : float
        R_vir in pixels
    size : int
        Size of the cutout
    r_multiples : list
        List of R_vir multiples (e.g., [1, 2, 3])
    
    Returns
    -------
    dict : Dictionary with keys as r_multiples and values as integrated masses
    """
    # Ensure halo_mass is 2D
    if halo_mass.ndim == 1:
        sqrt_size = int(np.sqrt(len(halo_mass)))
        if sqrt_size * sqrt_size == len(halo_mass):
            halo_mass = halo_mass.reshape(sqrt_size, sqrt_size)
            size = sqrt_size
        else:
            raise ValueError(f"Cannot reshape 1D array of length {len(halo_mass)} to 2D")
    
    if halo_mass.shape[0] != size or halo_mass.shape[1] != size:
        size = halo_mass.shape[0]
    
    # Create distance map from center
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Calculate integrated mass within each aperture
    masses = {}
    for r_mult in r_multiples:
        mask = dist < r_mult * radius_pix
        masses[r_mult] = np.sum(halo_mass[mask])
    
    return masses


def radius_kpc_to_pixels(radius_kpc):
    """Convert radius from kpc to pixels."""
    return radius_kpc / 1000 * GRID_SIZE / BOX_SIZE


# ============================================================================
# SECTION 5: Statistical Analysis
# ============================================================================

def compute_spearman_correlation(param_values, metric_values):
    """
    Compute Spearman rank correlation between parameter and metric.
    
    Parameters
    ----------
    param_values : np.ndarray
        Parameter values
    metric_values : np.ndarray
        Metric values to correlate with
        
    Returns
    -------
    float : Spearman correlation coefficient (or NaN if insufficient data)
    """
    valid_mask = np.isfinite(param_values) & np.isfinite(metric_values)
    
    if np.sum(valid_mask) > 3:
        corr, _ = spearmanr(param_values[valid_mask], metric_values[valid_mask])
        return corr
    else:
        return np.nan


def compute_ssim(true_image, gen_image, component='total'):
    """
    Compute Structural Similarity Index between true and generated images.
    
    Parameters
    ----------
    true_image : np.ndarray
        True (target) image
    gen_image : np.ndarray
        Generated image
    component : str
        Component type ('dm', 'gas', 'stars', 'total')
        
    Returns
    -------
    float : SSIM value
    """
    from skimage.metrics import structural_similarity as ssim
    
    epsilon = 1e-10
    
    if component.lower() == 'stars':
        # For sparse stellar fields, use masked approach
        threshold = np.percentile(true_image[true_image > 0], 10) if (true_image > 0).sum() > 10 else 1e8
        mask = (true_image > threshold) | (gen_image > threshold)
        
        if mask.sum() < 100:
            # Too few pixels, use asinh transformation
            true_asinh = np.arcsinh(true_image / 1e9)
            gen_asinh = np.arcsinh(gen_image / 1e9)
            data_range = max(true_asinh.max() - true_asinh.min(), 
                           gen_asinh.max() - gen_asinh.min())
            if data_range > 0:
                return ssim(true_asinh, gen_asinh, data_range=data_range)
            else:
                return 1.0  # Both essentially zero
        else:
            true_processed = np.where(mask, np.log10(true_image + epsilon), -10)
            gen_processed = np.where(mask, np.log10(gen_image + epsilon), -10)
            return ssim(true_processed, gen_processed, data_range=25.0)
    else:
        # For dense fields (DM, Gas, Total): use log transformation
        true_log = np.log10(true_image + epsilon)
        gen_log = np.log10(gen_image + epsilon)
        
        true_log = np.clip(true_log, -10, 15)
        gen_log = np.clip(gen_log, -10, 15)
        
        true_log = np.nan_to_num(true_log, nan=-10, posinf=15, neginf=-10)
        gen_log = np.nan_to_num(gen_log, nan=-10, posinf=15, neginf=-10)
        
        return ssim(true_log, gen_log, data_range=25.0)


# ============================================================================
# SECTION 6: Plotting Utilities
# ============================================================================

def plot_simulation_map(ax, field, title='', cmap='inferno', vmin=8, vmax=12, 
                        log_scale=True, show_colorbar=True, extent=None):
    """
    Plot a 2D simulation map with standard formatting.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    field : np.ndarray
        2D field to plot
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    log_scale : bool
        Whether to apply log10 transformation
    show_colorbar : bool
        Whether to add colorbar
    extent : list
        [xmin, xmax, ymin, ymax] for imshow
    """
    if extent is None:
        extent = [0, BOX_SIZE, 0, BOX_SIZE]
    
    data = np.log10(field) if log_scale else field
    im = ax.imshow(data.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                   extent=extent, origin='lower')
    
    ax.set_xlabel('X [Mpc/h]')
    ax.set_ylabel('Y [Mpc/h]')
    if title:
        ax.set_title(title)
    
    if show_colorbar:
        return im
    return im


def add_halo_circles(ax, centers_mpc, radii_mpc, edgecolor='black', **kwargs):
    """Add circular patches marking halo positions and radii."""
    for (cx, cy), r in zip(centers_mpc, radii_mpc):
        circ = patches.Circle((cx, cy), radius=r, edgecolor=edgecolor, 
                              facecolor='none', linewidth=1.5, alpha=0.9, **kwargs)
        ax.add_patch(circ)


def create_figure_with_colorbar(figsize=(7, 7), pad=0.01, width=0.03):
    """Create figure and axes with space for colorbar."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    return fig, ax, cax


def savefig_paper(fig, filename, output_dir='paper_plots'):
    """Save figure with consistent settings for paper."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/{filename}', bbox_inches='tight', 
                pad_inches=0.02, dpi=150)
