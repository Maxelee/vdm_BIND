"""Sampling utilities for BIND inference.

This module provides functions for generating samples from trained models
and analyzing density profiles.
"""

import numpy as np
import torch
from tqdm import tqdm


def sample(vdm, conditions, batch_size=1, conditional_params=None, n_sampling_steps=None):
    """
    Process multiple conditions and return stacked samples with optional conditioning.

    Parameters
    ----------
    vdm : model
        Your VDM model (or any model with a draw_samples method).
    conditions : torch.Tensor
        Shape (N, C, H, W) or (N, C, H, W, D) where C is the number of
        conditioning channels:
        - C=1: base DM condition only
        - C>1: base DM + (C-1) large-scale maps
    batch_size : int, optional
        Number of samples per condition. Default: 1.
    conditional_params : array-like, optional
        Optional array of conditional parameters, shape (N, n_params).
    n_sampling_steps : int, optional
        Number of diffusion steps for sampling. If None, uses model's
        default (from hparams or 1000 as fallback).

    Returns
    -------
    torch.Tensor
        Shape (N, batch_size, 3, H, W) or (N, batch_size, 3, H, W, D).
    """
    vdm = vdm.to('cuda')
    samples = []

    # Determine number of sampling steps
    if n_sampling_steps is None:
        n_sampling_steps = getattr(vdm.hparams, 'n_sampling_steps', 1000)

    # Determine dimensionality
    is_3d = len(conditions.shape) == 5  # (N, C, H, W, D)

    for i, cond in enumerate(tqdm(conditions, desc=f'Generating Samples ({n_sampling_steps} steps)')):
        # cond is now (C, H, W) or (C, H, W, D) where C = 1 + num_large_scales
        # Expand to batch_size: (batch_size, C, H, W) or (batch_size, C, H, W, D)
        if is_3d:
            cond_expanded = cond.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to('cuda')
        else:
            cond_expanded = cond.unsqueeze(0).expand(batch_size, -1, -1, -1).to('cuda')

        if conditional_params is not None:
            param_row = torch.tensor(conditional_params[i], dtype=torch.float32, device='cuda')
            param_expanded = param_row.unsqueeze(0).expand(batch_size, -1).to('cuda')
        else:
            param_expanded = None

        hydro_sample = vdm.draw_samples(
            conditioning=cond_expanded,
            batch_size=batch_size,
            n_sampling_steps=n_sampling_steps,
            param_conditioning=param_expanded,
        )

        samples.append(hydro_sample.unsqueeze(0))  # Add N dimension

    return torch.cat(samples, dim=0).to("cpu")


def get_density(mass_map, radius, nbins=20, nR=2, logbins=False, physical_bins=False):
    """
    Calculate surface density in radial shells from the center of the image.

    Parameters
    ----------
    mass_map : np.ndarray
        2D array of mass/density values.
    radius : float
        Physical radius scale (e.g., R200 in kpc).
    nbins : int, optional
        Number of radial bins. Default: 20.
    nR : float, optional
        Maximum radius in units of the scale radius (ignored if physical_bins=True).
    logbins : bool, optional
        Whether to use logarithmic binning. Default: False.
    physical_bins : bool, optional
        If True, use physical units (kpc) for binning instead of normalized.

    Returns
    -------
    bin_centers : np.ndarray
        Radial bin centers (units depend on physical_bins setting).
    surface_density : np.ndarray
        Surface density in each radial shell [mass/area].
    """
    # Map parameters
    map_size_mpc = 50 / 1024 * 128  # Total physical size of map in Mpc
    pixel_size_mpc = map_size_mpc / mass_map.shape[0]  # Size per pixel in Mpc
    pixel_size_kpc = pixel_size_mpc * 1000  # Convert to kpc

    # Calculate pixel distances from center
    center = (mass_map.shape[0] / 2 - 0.5, mass_map.shape[1] / 2 - 0.5)
    y, x = np.indices(mass_map.shape)
    radii_pixels = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    radii_physical = radii_pixels * pixel_size_kpc  # Physical distance in kpc
    radii_normalized = radii_physical / radius  # Normalized by scale radius

    # Radial binning - choose between physical and normalized
    if physical_bins:
        min_radius = pixel_size_kpc * 4  # Minimum radius (4 pixels)
        max_radius = 2500  # 2.5 Mpc in kpc

        if not logbins:
            bin_edges = np.linspace(min_radius, max_radius, nbins + 1, endpoint=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        else:
            bin_edges = np.logspace(np.log10(min_radius), np.log10(max_radius), nbins + 1, endpoint=True)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean

        radii_for_binning = radii_physical
    else:
        if not logbins:
            bin_edges = np.linspace(0.01, nR, nbins + 1, endpoint=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        else:
            bin_edges = np.logspace(np.log10(0.01), np.log10(nR), nbins + 1, endpoint=True)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

        radii_for_binning = radii_normalized

    # Calculate surface density in each annular shell
    surface_density = []
    for i in range(len(bin_edges) - 1):
        mask = (radii_for_binning >= bin_edges[i]) & (radii_for_binning < bin_edges[i + 1])
        total_mass = np.sum(mass_map[mask])

        if physical_bins:
            inner_radius_kpc = bin_edges[i]
            outer_radius_kpc = bin_edges[i + 1]
        else:
            inner_radius_kpc = bin_edges[i] * radius
            outer_radius_kpc = bin_edges[i + 1] * radius

        shell_area_kpc2 = np.pi * (outer_radius_kpc ** 2 - inner_radius_kpc ** 2)

        if shell_area_kpc2 > 0:
            surf_density = total_mass / shell_area_kpc2
        else:
            surf_density = 0.0

        surface_density.append(surf_density)

    return np.array(bin_centers), np.array(surface_density)


def generate_density_profiles(
    hydro_sample,
    target,
    test_data,
    density_types=('dm', 'gas', 'star', 'all'),
    nbins=20,
    nR=2,
    logbins=True,
    physical_bins=False,
):
    """
    Generate density profiles for multiple fields.

    Parameters
    ----------
    hydro_sample : np.ndarray
        Sampled data array, shape (N, batch_size, 3, H, W).
    target : np.ndarray
        True target data array, shape (N, 3, H, W).
    test_data : DataHandler
        DataHandler instance with metadata and unnormalization methods.
    density_types : tuple, optional
        Tuple of fields to process ('dm', 'gas', 'star', 'all').
    nbins : int, optional
        Number of radial bins.
    nR : float, optional
        Maximum radius in units of R200.
    logbins : bool, optional
        Whether to use logarithmic binning.
    physical_bins : bool, optional
        If True, use physical units (kpc) for binning.

    Returns
    -------
    true_profiles : dict
        Dictionary with density types as keys and lists of profiles.
    sampled_profiles : dict
        Dictionary with density types as keys and lists of sampled profiles.
    """
    channel_map = {
        'dm': 0,
        'gas': 1,
        'star': 2,
        'all': [0, 1, 2],
    }

    invalid_types = [dt for dt in density_types if dt not in channel_map]
    if invalid_types:
        raise ValueError(f"Invalid density types: {invalid_types}. Valid options: {list(channel_map.keys())}")

    true_profiles = {dt: [] for dt in density_types}
    sampled_profiles = {dt: [] for dt in density_types}

    def calculate_density(mass_map, radius):
        return get_density(
            mass_map=np.array(mass_map),
            radius=radius,
            nbins=nbins,
            nR=nR,
            logbins=logbins,
            physical_bins=physical_bins,
        )[1]

    for ii in range(len(hydro_sample)):
        radius = test_data.metadata.loc[:, 'R_hydro'].iloc[ii]

        for dt in density_types:
            channel = channel_map[dt]

            # Process true density
            true_map = test_data.unnormalize_target(target[ii])[channel]
            if dt == 'all':
                true_map = np.sum(true_map, axis=0)
            true_profiles[dt].append(calculate_density(true_map, radius))

            # Process sampled densities
            sample_maps = [
                test_data.unnormalize_target(hydro_sample[ii, n])[channel]
                for n in range(len(hydro_sample[ii]))
            ]
            if dt == 'all':
                sample_maps = [np.sum(s, axis=0) for s in sample_maps]
            sampled_profiles[dt].append([calculate_density(smap, radius) for smap in sample_maps])

    return true_profiles, sampled_profiles
