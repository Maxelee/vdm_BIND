"""
Analysis functions for BIND pipeline.
Comprehensive analysis suite matching BIND_usage_CV.ipynb.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import Pk_library as PKL
    from bind.power_spec import (compute_power_spectrum_simple, 
                                  compute_power_spectrum_batch,
                                  compute_cross_power_spectrum)
    HAS_PKL = True
except ImportError:
    HAS_PKL = False
    print("Warning: Pk_library not found. Power spectrum analysis will be disabled.")


# Set publication-ready style
plt.rcParams.update({
    'font.size': 14, 
    'font.family': 'serif',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def plot_full_box_comparison(full_dmo, full_hydro, hydro_replaced, final_maps, output_dir, model_name):
    """
    Create 2x4 comparison plot showing full box maps and residuals.
    Matches BIND_usage_CV.ipynb visualization.
    """
    print(f"Creating full box comparison plot...")
    
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))

    # Top row - log-scaled density maps
    im0 = ax[0, 0].imshow(np.log10(full_dmo[:, :]), vmin=8, vmax=12, cmap='inferno', origin='lower')
    ax[0, 0].set_title('DMO', fontsize=16, fontweight='bold')
    ax[0, 0].set_xlabel('X [pixels]', fontsize=14)
    ax[0, 0].set_ylabel('Y [pixels]', fontsize=14)
    cbar0 = plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)
    cbar0.set_label(r'$\log_{10}(M / M_\odot)$', fontsize=12)

    im1 = ax[0, 1].imshow(np.log10(full_hydro[:, :]), vmin=8, vmax=12, cmap='inferno', origin='lower')
    ax[0, 1].set_title('Hydro', fontsize=16, fontweight='bold')
    ax[0, 1].set_xlabel('X [pixels]', fontsize=14)
    ax[0, 1].set_ylabel('Y [pixels]', fontsize=14)
    cbar1 = plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label(r'$\log_{10}(M / M_\odot)$', fontsize=12)

    im2 = ax[0, 2].imshow(np.log10(hydro_replaced[:, :]), vmin=8, vmax=12, cmap='inferno', origin='lower')
    ax[0, 2].set_title('Hydro Replaced', fontsize=16, fontweight='bold')
    ax[0, 2].set_xlabel('X [pixels]', fontsize=14)
    ax[0, 2].set_ylabel('Y [pixels]', fontsize=14)
    cbar2 = plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)
    cbar2.set_label(r'$\log_{10}(M / M_\odot)$', fontsize=12)

    im3 = ax[0, 3].imshow(np.log10(np.array(final_maps).mean(axis=0)), vmin=8, vmax=12, cmap='inferno', origin='lower')
    ax[0, 3].set_title('BIND Generated', fontsize=16, fontweight='bold')
    ax[0, 3].set_xlabel('X [pixels]', fontsize=14)
    ax[0, 3].set_ylabel('Y [pixels]', fontsize=14)
    cbar3 = plt.colorbar(im3, ax=ax[0, 3], fraction=0.046, pad=0.04)
    cbar3.set_label(r'$\log_{10}(M / M_\odot)$', fontsize=12)

    # Bottom row - residual maps
    im4 = ax[1, 0].imshow(full_dmo[:, :]/full_hydro[:, :]-1, vmin=-.1, vmax=.1, cmap='bwr', origin='lower')
    ax[1, 0].set_title('DMO - Hydro', fontsize=16, fontweight='bold')
    ax[1, 0].set_xlabel('X [pixels]', fontsize=14)
    ax[1, 0].set_ylabel('Y [pixels]', fontsize=14)
    cbar4 = plt.colorbar(im4, ax=ax[1, 0], fraction=0.046, pad=0.04)
    cbar4.set_label(r'$\Delta M / M$', fontsize=12)

    im5 = ax[1, 1].imshow(full_dmo[:, :]/hydro_replaced[:, :]-1, vmin=-.1, vmax=.1, cmap='bwr', origin='lower')
    ax[1, 1].set_title('DMO - Hydro Replaced', fontsize=16, fontweight='bold')
    ax[1, 1].set_xlabel('X [pixels]', fontsize=14)
    ax[1, 1].set_ylabel('Y [pixels]', fontsize=14)
    cbar5 = plt.colorbar(im5, ax=ax[1, 1], fraction=0.046, pad=0.04)
    cbar5.set_label(r'$\Delta M / M$', fontsize=12)

    im6 = ax[1, 2].imshow(full_dmo[:, :]/np.array(final_maps).mean(axis=0)-1, vmin=-.1, vmax=.1, cmap='bwr', origin='lower')
    ax[1, 2].set_title('DMO - BIND', fontsize=16, fontweight='bold')
    ax[1, 2].set_xlabel('X [pixels]', fontsize=14)
    ax[1, 2].set_ylabel('Y [pixels]', fontsize=14)
    cbar6 = plt.colorbar(im6, ax=ax[1, 2], fraction=0.046, pad=0.04)
    cbar6.set_label(r'$\Delta M / M$', fontsize=12)

    im7 = ax[1, 3].imshow(hydro_replaced[:, :]/np.array(final_maps).mean(axis=0)-1, vmin=-.1, vmax=.1, cmap='bwr', origin='lower')
    ax[1, 3].set_title('Hydro Replaced - BIND', fontsize=16, fontweight='bold')
    ax[1, 3].set_xlabel('X [pixels]', fontsize=14)
    ax[1, 3].set_ylabel('Y [pixels]', fontsize=14)
    cbar7 = plt.colorbar(im7, ax=ax[1, 3], fraction=0.046, pad=0.04)
    cbar7.set_label(r'$\Delta M / M$', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_maps.pdf')
    plt.close()
    print(f"Saved to {output_dir}/comparison_maps.pdf")


def compute_power_spectrum_fullbox(full_hydro, full_dmo, full_hydro_replace, final_maps, 
                                    boxsize, output_dir, model_name):
    """
    Compute power spectra for full box maps and create comparison plots.
    """
    if not HAS_PKL:
        print("Skipping power spectrum analysis - Pk_library not available")
        return
    
    print(f"Computing full box power spectra...")
    
    # Compute overdensity fields
    delta_hydro = full_hydro / np.mean(full_hydro, dtype=np.float64); delta_hydro -= 1.0
    delta_dmo = full_dmo / np.mean(full_dmo, dtype=np.float64); delta_dmo -= 1.0
    delta_replace = full_hydro_replace / np.mean(full_hydro_replace, dtype=np.float64); delta_replace -= 1.0

    # Parameters
    grid = 1024
    BoxSize = boxsize  # Mpc/h
    MAS = 'NGP'
    threads = 0

    # Compute power spectra
    Pk2D_replace = PKL.Pk_plane(delta_replace, BoxSize, MAS, threads)
    Pk2D_hydro = PKL.Pk_plane(delta_hydro, BoxSize, MAS, threads)
    Pk2D_dmo = PKL.Pk_plane(delta_dmo, BoxSize, MAS, threads)
    
    # Compute BIND power spectra for all realizations
    Pk_binded = []
    for final_map in final_maps:
        delta_binded = final_map / np.mean(final_map, dtype=np.float64); delta_binded -= 1.0
        Pk2D_binded = PKL.Pk_plane(delta_binded.astype(np.float32), BoxSize, MAS, threads)
        Pk_binded.append(Pk2D_binded.Pk)
    Pk_binded = np.array(Pk_binded)
    
    k = Pk2D_hydro.k
    Nmodes = Pk2D_hydro.Nmodes

    # Compute cross-correlations
    XPk2D_hydro_dmo = PKL.XPk_plane(delta_hydro, delta_dmo, BoxSize, MAS, MAS, threads)
    XPk2D_replace_dmo = PKL.XPk_plane(delta_replace, delta_dmo, BoxSize, MAS, MAS, threads)
    
    r_hydro_dmo = XPk2D_hydro_dmo.r
    r_replaced_dmo = XPk2D_replace_dmo.r
    
    # Cross-correlation for BIND realizations
    r_hydro_binded = []
    for final_map in final_maps:
        delta_binded = final_map / np.mean(final_map, dtype=np.float64); delta_binded -= 1.0
        XPk2D = PKL.XPk_plane(delta_hydro, delta_binded.astype(np.float32), BoxSize, MAS, MAS, threads)
        r_hydro_binded.append(XPk2D.r)
    r_hydro_binded = np.array(r_hydro_binded)
    
    # Create plots
    fig, axs = plt.subplots(ncols=2, figsize=(14, 5), sharex=True)

    # Power spectrum ratio
    axs[0].semilogx(k, Pk2D_hydro.Pk / Pk2D_dmo.Pk, 'r-', linewidth=2.5, label='Hydro/DMO', alpha=0.8)
    axs[0].semilogx(k, Pk2D_replace.Pk / Pk2D_dmo.Pk, 'g-', linewidth=2.5, label='Hydro Replaced/DMO', alpha=0.8)
    mean = np.mean(Pk_binded / Pk2D_dmo.Pk[None, :], axis=0)
    std = np.std(Pk_binded / Pk2D_dmo.Pk[None, :], axis=0)
    axs[0].fill_between(k, mean - std, mean + std, color='blue', alpha=0.3, label=r'BIND 1$\sigma$')
    axs[0].semilogx(k, mean, 'b-', linewidth=2.5, label='BIND/DMO', alpha=0.8)
    axs[0].axhline(1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    axs[0].set_xlabel(r'$k$ [h/Mpc]', fontsize=16)
    axs[0].set_ylabel(r'$P(k) / P_{\rm DMO}(k)$', fontsize=16)
    axs[0].set_ylim(0.5, 1.5)
    axs[0].legend(loc='best', fontsize=12, frameon=True, framealpha=0.9)
    axs[0].grid(True, alpha=0.3, linestyle='--')
    axs[0].tick_params(axis='both', which='major', labelsize=12)

    # Cross-correlation coefficient
    axs[1].semilogx(k, r_hydro_dmo, 'r-', linewidth=2.5, label='Hydro-DMO', alpha=0.8)
    axs[1].semilogx(k, r_replaced_dmo, 'g-', linewidth=2.5, label='Hydro Replaced-DMO', alpha=0.8)
    mean_r = np.mean(r_hydro_binded, axis=0)
    std_r = np.std(r_hydro_binded, axis=0)
    axs[1].fill_between(k, mean_r - std_r, mean_r + std_r, color='blue', alpha=0.3, label=r'BIND 1$\sigma$')
    axs[1].semilogx(k, mean_r, 'b-', linewidth=2.5, label='BIND-Hydro', alpha=0.8)
    axs[1].axhline(1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    axs[1].set_xlabel(r'$k$ [h/Mpc]', fontsize=16)
    axs[1].set_ylabel(r'Cross-Correlation $r(k)$', fontsize=16)
    axs[1].set_ylim(0, 1.05)
    axs[1].legend(loc='best', fontsize=12, frameon=True, framealpha=0.9)
    axs[1].grid(True, alpha=0.3, linestyle='--')
    axs[1].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/power_suppression.pdf')
    plt.close()
    print(f"Saved to {output_dir}/power_suppression.pdf")
    
    # Save power spectrum data
    np.savez(f'{output_dir}/power_spectra.npz',
             k=k, Pk_hydro=Pk2D_hydro.Pk, Pk_dmo=Pk2D_dmo.Pk, 
             Pk_replace=Pk2D_replace.Pk, Pk_binded=Pk_binded,
             r_hydro_dmo=r_hydro_dmo, r_replaced_dmo=r_replaced_dmo,
             r_hydro_binded=r_hydro_binded)


def get_projected_surface_density(halo_mass, radius_pix, size=128, nbins=15):
    """
    Calculate projected surface density profile in logarithmic radial bins.
    """
    # Ensure halo_mass is 2D
    if halo_mass.ndim == 1:
        # If 1D, try to reshape to square
        sqrt_size = int(np.sqrt(len(halo_mass)))
        if sqrt_size * sqrt_size == len(halo_mass):
            halo_mass = halo_mass.reshape(sqrt_size, sqrt_size)
            size = sqrt_size
        else:
            raise ValueError(f"Cannot reshape 1D array of length {len(halo_mass)} to 2D")
    
    # Update size if halo_mass has different dimensions
    if halo_mass.shape[0] != size or halo_mass.shape[1] != size:
        size = halo_mass.shape[0]  # Assume square
    
    radial_bins = np.logspace(-1, np.log10(10), nbins)
    bin_centers = np.sqrt(radial_bins[:-1] * radial_bins[1:])
    
    # Calculate annular areas
    annular_areas = np.pi * (radial_bins[1:]**2 - radial_bins[:-1]**2) * (radius_pix*50/1024)**2
    
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


def plot_density_profiles_and_mass(true_halos_all, gen_halos_all, halo_radii, 
                                    output_dir, model_name, nbins=15):
    """
    Create density profile and mass comparison plots.
    """
    print(f"Computing density profiles and mass comparisons...")
    
    # Compute masses
    true_mass = true_halos_all.sum(axis=-1).sum(axis=-1)  # Shape: (n_halos, 3)
    gen_mass = gen_halos_all.sum(axis=-1).sum(axis=-1)  # Shape: (n_halos, n_realizations, 3)
    
    # Compute density profiles with error handling
    hydro_density_dm = []
    hydro_density_gas = []
    hydro_density_star = []
    for idx, (m, r) in enumerate(zip(true_halos_all, halo_radii)):
        try:
            hydro_density_dm.append(get_projected_surface_density(m[0], r, size=128, nbins=nbins)[0])
            hydro_density_gas.append(get_projected_surface_density(m[1], r, size=128, nbins=nbins)[0])
            hydro_density_star.append(get_projected_surface_density(m[2], r, size=128, nbins=nbins)[0])
        except Exception as e:
            print(f"Warning: Error processing halo {idx} - m.shape={m.shape}, r={r}: {e}")
            # Use NaN arrays as fallback
            hydro_density_dm.append(np.full(nbins-1, np.nan))
            hydro_density_gas.append(np.full(nbins-1, np.nan))
            hydro_density_star.append(np.full(nbins-1, np.nan))
    
    hydro_density_dm = np.array(hydro_density_dm)
    hydro_density_gas = np.array(hydro_density_gas)
    hydro_density_star = np.array(hydro_density_star)

    gen_density_dm = []
    gen_density_gas = []
    gen_density_star = []
    for idx, (m, r) in enumerate(zip(gen_halos_all, halo_radii)):
        try:
            gen_density_dm.append(get_projected_surface_density(m.mean(axis=0)[0], r, size=128, nbins=nbins)[0])
            gen_density_gas.append(get_projected_surface_density(m.mean(axis=0)[1], r, size=128, nbins=nbins)[0])
            gen_density_star.append(get_projected_surface_density(m.mean(axis=0)[2], r, size=128, nbins=nbins)[0])
        except Exception as e:
            print(f"Warning: Error processing generated halo {idx} - m.shape={m.shape}, r={r}: {e}")
            gen_density_dm.append(np.full(nbins-1, np.nan))
            gen_density_gas.append(np.full(nbins-1, np.nan))
            gen_density_star.append(np.full(nbins-1, np.nan))
    
    gen_density_dm = np.array(gen_density_dm)
    gen_density_gas = np.array(gen_density_gas)
    gen_density_star = np.array(gen_density_star)
    _, bin_centers = get_projected_surface_density(gen_halos_all[0].mean(axis=0)[0], halo_radii[0], size=128, nbins=nbins)

    # Create plot
    fig, axs = plt.subplots(ncols=2, figsize=(14, 6), sharey=True)

    # Left panel: Density profile residuals (all components)
    dm_profile_mean = np.mean(hydro_density_dm/gen_density_dm - 1, axis=0)
    dm_profile_std = np.std(hydro_density_dm/gen_density_dm - 1, axis=0)
    gas_profile_mean = np.mean(hydro_density_gas/gen_density_gas - 1, axis=0)
    gas_profile_std = np.std(hydro_density_gas/gen_density_gas - 1, axis=0)
    star_profile_mean = np.mean(hydro_density_star/gen_density_star - 1, axis=0)
    star_profile_std = np.std(hydro_density_star/gen_density_star - 1, axis=0)

    axs[0].semilogx(bin_centers, dm_profile_mean, 'o-', linewidth=2.5, markersize=6, label='DM', color='C0')
    axs[0].fill_between(bin_centers, dm_profile_mean - dm_profile_std, dm_profile_mean + dm_profile_std, 
                        alpha=0.25, color='C0')

    axs[0].semilogx(bin_centers, gas_profile_mean, 'o-', linewidth=2.5, markersize=6, label='Gas', color='C1')
    axs[0].fill_between(bin_centers, gas_profile_mean - gas_profile_std, gas_profile_mean + gas_profile_std, 
                        alpha=0.25, color='C1')

    axs[0].semilogx(bin_centers, star_profile_mean, 'o-', linewidth=2.5, markersize=6, label='Stars', color='C2')
    axs[0].fill_between(bin_centers, star_profile_mean - star_profile_std, star_profile_mean + star_profile_std, 
                        alpha=0.25, color='C2')

    # Right panel: Mass residuals (all components)
    axs[1].plot(np.log10(true_mass[:, 0]), true_mass[:, 0] / gen_mass[:, :, 0].mean(axis=1) - 1, 
                'o', markersize=5, alpha=0.6, label='DM', color='C0')
    axs[1].plot(np.log10(true_mass[:, 1]), true_mass[:, 1] / gen_mass[:, :, 1].mean(axis=1) - 1, 
                'o', markersize=5, alpha=0.6, label='Gas', color='C1')
    axs[1].plot(np.log10(true_mass[:, 2]), true_mass[:, 2] / gen_mass[:, :, 2].mean(axis=1) - 1, 
                'o', markersize=5, alpha=0.6, label='Stars', color='C2')

    # Add horizontal reference lines and styling
    for ax in axs:
        ax.axhline(0, color='k', ls='--', linewidth=1.5, alpha=0.7)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='best', fontsize=12, frameon=True, framealpha=0.9)

    # Set labels
    axs[0].set_xlabel(r'$r / R_{\rm vir}$', fontsize=16)
    axs[0].set_ylabel(r'$\Delta \rho / \rho_{\rm hydro}$', fontsize=16)
    axs[1].set_xlabel(r'$\log_{10}(M_{\rm hydro} / M_\odot)$', fontsize=16)
    axs[1].set_ylabel(r'$\Delta M / M_{\rm hydro}$', fontsize=16)

    # Set y-limits
    axs[0].set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/density_and_mass.pdf')
    plt.close()
    print(f"Saved to {output_dir}/density_and_mass.pdf")


def compute_halo_level_power_spectra(true_halos_all, gen_halos_all, boxsize, output_dir, model_name):
    """
    Compute halo-level power spectra for individual components.
    """
    if not HAS_PKL:
        print("Skipping halo-level power spectrum analysis - Pk_library not available")
        return
    
    print(f"Computing halo-level power spectra...")
    
    # Use consolidated power spectrum functions from bind.power_spec
    def compute_binded_halo_cross_correlation(full_hydro, binded, boxsize=6.25):
        """Compute cross-correlation for batched halo fields."""
        binded = np.array(binded, dtype=np.float64)
        delta_hydro = full_hydro / np.mean(full_hydro, dtype=np.float64, axis=(1,2), keepdims=True)
        delta_hydro -= 1.0
        delta_binded = binded / np.mean(binded, dtype=np.float64, axis=(1,2), keepdims=True)
        delta_binded -= 1.0
        delta_binded = delta_binded.astype(np.float32)
        delta_hydro = delta_hydro.astype(np.float32)

        BoxSize = boxsize
        MAS = 'NGP'
        threads = 0

        XPk2D = [PKL.XPk_plane(delta_hydro[i], delta_binded[i], BoxSize, MAS, MAS, threads) 
                 for i in range(binded.shape[0])]

        k = XPk2D[0].k
        r = np.array([XPk2D[i].r for i in range(binded.shape[0])])
        Nmodes = XPk2D[0].Nmodes
        return k, r, Nmodes

    # Compute power spectra for each component using consolidated function
    k, Pk_binded_dm, _ = compute_power_spectrum_batch(np.array(gen_halos_all)[:, :, 0].mean(axis=1), BoxSize=6.25)
    k, Pk_binded_gas, _ = compute_power_spectrum_batch(np.array(gen_halos_all)[:, :, 1].mean(axis=1), BoxSize=6.25)
    k, Pk_binded_star, _ = compute_power_spectrum_batch(np.array(gen_halos_all)[:, :, 2].mean(axis=1), BoxSize=6.25)
    k, Pk_binded_total, _ = compute_power_spectrum_batch(np.array(gen_halos_all).mean(axis=1).sum(axis=1), BoxSize=6.25)

    k, Pk_hydro_dm, _ = compute_power_spectrum_batch(true_halos_all[:, 0], BoxSize=6.25)
    k, Pk_hydro_gas, _ = compute_power_spectrum_batch(true_halos_all[:, 1], BoxSize=6.25)
    k, Pk_hydro_star, _ = compute_power_spectrum_batch(true_halos_all[:, 2], BoxSize=6.25)
    k, Pk_hydro_total, _ = compute_power_spectrum_batch(true_halos_all.sum(axis=1), BoxSize=6.25)

    k, r_hydro_binded_dm, Nmodes = compute_binded_halo_cross_correlation(
        true_halos_all[:, 0], np.array(gen_halos_all)[:, :, 0].mean(axis=1), boxsize=6.25)
    k, r_hydro_binded_gas, Nmodes = compute_binded_halo_cross_correlation(
        true_halos_all[:, 1], np.array(gen_halos_all)[:, :, 1].mean(axis=1), boxsize=6.25)
    k, r_hydro_binded_star, Nmodes = compute_binded_halo_cross_correlation(
        true_halos_all[:, 2], np.array(gen_halos_all)[:, :, 2].mean(axis=1), boxsize=6.25)
    k, r_hydro_binded_total, Nmodes = compute_binded_halo_cross_correlation(
        true_halos_all.sum(axis=1), np.array(gen_halos_all).mean(axis=1).sum(axis=1), boxsize=6.25)

    # Create plot
    fig, axs = plt.subplots(nrows=2, figsize=(10, 8), sharex=True)

    # Power spectrum ratio
    mean = np.mean(np.sqrt(Pk_binded_dm / Pk_hydro_dm), axis=0)
    std = np.std(np.sqrt(Pk_binded_dm / Pk_hydro_dm), axis=0)
    axs[0].semilogx(k, mean, '-', linewidth=2.5, markersize=5, label='DM', color='C0')
    axs[0].fill_between(k, mean - std, mean + std, color='C0', alpha=0.3)

    mean = np.mean(np.sqrt(Pk_binded_gas / Pk_hydro_gas), axis=0)
    std = np.std(np.sqrt(Pk_binded_gas / Pk_hydro_gas), axis=0)
    axs[0].semilogx(k, mean, '-', linewidth=2.5, markersize=5, label='Gas', color='C1')
    axs[0].fill_between(k, mean - std, mean + std, color='C1', alpha=0.3)

    mean = np.mean(np.sqrt(Pk_binded_star / Pk_hydro_star), axis=0)
    std = np.std(np.sqrt(Pk_binded_star / Pk_hydro_star), axis=0)
    axs[0].semilogx(k, mean, '-', linewidth=2.5, markersize=5, label='Stars', color='C2')
    axs[0].fill_between(k, mean - std, mean + std, color='C2', alpha=0.3)

    mean = np.mean(np.sqrt(Pk_binded_total / Pk_hydro_total), axis=0)
    std = np.std(np.sqrt(Pk_binded_total / Pk_hydro_total), axis=0)
    axs[0].semilogx(k, mean, '-', linewidth=2.5, markersize=5, label='Total', color='k', linestyle=':')
    axs[0].fill_between(k, mean - std, mean + std, color='k', alpha=0.3)
    axs[0].set_ylim(0, 2)
    axs[0].axhline(1, color='k', ls='--', linewidth=1.5, alpha=0.7)
    axs[0].set_ylabel(r'$\sqrt{P_{\rm gen}(k) / P_{\rm hydro}(k)}$', fontsize=16)
    axs[0].legend(loc='best', fontsize=12, frameon=True, framealpha=0.9)
    axs[0].grid(True, alpha=0.3, linestyle='--')
    axs[0].tick_params(axis='both', which='major', labelsize=12)

    # Cross-correlation
    mean = np.mean(r_hydro_binded_dm, axis=0)
    std = np.std(r_hydro_binded_dm, axis=0)
    axs[1].semilogx(k, mean, '-', linewidth=2.5, markersize=5, label='DM', color='C0')
    axs[1].fill_between(k, mean - std, mean + std, color='C0', alpha=0.3)
    
    mean = np.mean(r_hydro_binded_gas, axis=0)
    std = np.std(r_hydro_binded_gas, axis=0)
    axs[1].semilogx(k, mean, '-', linewidth=2.5, markersize=5, label='Gas', color='C1')
    axs[1].fill_between(k, mean - std, mean + std, color='C1', alpha=0.3)
    
    mean = np.mean(r_hydro_binded_star, axis=0)
    std = np.std(r_hydro_binded_star, axis=0)
    axs[1].semilogx(k, mean, '-', linewidth=2.5, markersize=5, label='Stars', color='C2')
    axs[1].fill_between(k, mean - std, mean + std, color='C2', alpha=0.3)
    
    mean = np.mean(r_hydro_binded_total, axis=0)
    std = np.std(r_hydro_binded_total, axis=0)
    axs[1].semilogx(k, mean, '-', linewidth=2.5, markersize=5, label='Total', color='k', linestyle=':')
    axs[1].fill_between(k, mean - std, mean + std, color='k', alpha=0.3)
    axs[1].set_ylim(0, 1.05)
    axs[1].axhline(1, color='k', ls='--', linewidth=1.5, alpha=0.7)
    axs[1].set_ylabel(r'Cross-Correlation $r(k)$', fontsize=16)
    axs[1].set_xlabel(r'$k$ [h/Mpc]', fontsize=16)
    axs[1].legend(loc='best', fontsize=12, frameon=True, framealpha=0.9)
    axs[1].grid(True, alpha=0.3, linestyle='--')
    axs[1].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/halo_level_correlation.pdf')
    plt.close()
    print(f"Saved to {output_dir}/halo_level_correlation.pdf")


def plot_residual_histograms(true_halos_all, gen_halos_all, dm_halos_all, output_dir, model_name):
    """
    Create residual histograms comparing BIND-Hydro and Hydro-DMO.
    """
    print(f"Computing residual histograms...")
    
    fig, axs = plt.subplots(ncols=2, figsize=(14, 5), sharey=False)

    # Compute residuals for all halos
    num_halos = len(true_halos_all)
    all_residual_bind = []
    all_residual_dm = []

    for i in range(num_halos):
        residual_bind_i = np.mean(true_halos_all[i].sum(axis=0)[None, :, :]/gen_halos_all[i].sum(axis=1)-1, axis=0)
        residual_dm_i = true_halos_all[i].sum(axis=0)/dm_halos_all[i]-1
        all_residual_bind.append(residual_bind_i.flatten())
        all_residual_dm.append(residual_dm_i.flatten())

    # Flatten all residuals
    all_residual_bind = np.concatenate(all_residual_bind)
    all_residual_dm = np.concatenate(all_residual_dm)

    # Remove NaNs
    all_residual_bind = all_residual_bind[~np.isnan(all_residual_bind)]
    all_residual_dm = all_residual_dm[~np.isnan(all_residual_dm)]

    # Create histograms with density normalization
    bins = np.linspace(-1, 1, 51)
    hist_bind_hydro = np.histogram(all_residual_bind, bins=bins)
    hist_dm_hydro = np.histogram(all_residual_dm, bins=bins)

    # Calculate bin width
    bin_width = bins[1] - bins[0]

    # Normalize by total number of pixels and bin width to get density
    density_bind = hist_bind_hydro[0] / (len(all_residual_bind) * bin_width)
    density_dm = hist_dm_hydro[0] / (len(all_residual_dm) * bin_width)

    # Left panel: BIND - Hydro with Hydro - DMO comparison
    axs[0].plot(hist_bind_hydro[1][1:], density_bind, linewidth=2.5, 
                label='BIND - Hydro', color='C0')
    axs[0].plot(hist_dm_hydro[1][1:], density_dm, linewidth=2.5, 
                label='Hydro - DMO', color='k', linestyle=':')
    axs[0].axvline(np.mean(all_residual_bind), color='C0', 
                   ls='--', linewidth=2, alpha=0.7)
    axs[0].axvline(np.mean(all_residual_dm), color='k', 
                   ls='--', linewidth=2, alpha=0.7)
    axs[0].set_xlabel(r'$\Delta M / M$', fontsize=16)
    axs[0].set_ylabel('Density', fontsize=16)
    axs[0].legend(fontsize=12, frameon=True, framealpha=0.9, loc='upper left')
    axs[0].grid(True, alpha=0.3, linestyle='--')
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].text(0.95, 0.95, f'BIND Mean = {np.mean(all_residual_bind):.3f}\nHydro-DMO Mean = {np.mean(all_residual_dm):.3f}', 
                transform=axs[0].transAxes, fontsize=11, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right panel: Statistics summary
    axs[1].axis('off')
    stats_text = f"""
Mean Residual Statistics (All Halos):

BIND - Hydro:
  Mean: {np.mean(all_residual_bind):.4f}
  Std: {np.std(all_residual_bind):.4f}
  Median: {np.median(all_residual_bind):.4f}
  
Hydro - DMO:
  Mean: {np.mean(all_residual_dm):.4f}
  Std: {np.std(all_residual_dm):.4f}
  Median: {np.median(all_residual_dm):.4f}

Number of Halos: {num_halos}
Total Pixels: {len(all_residual_bind):,}
"""
    axs[1].text(0.1, 0.5, stats_text, transform=axs[1].transAxes, fontsize=13,
                va='center', ha='left', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals_comparison.pdf')
    plt.close()
    print(f"Saved to {output_dir}/residuals_comparison.pdf")


def plot_parameter_extremes_comparison(extreme_type, param_name, param_value,
                                       dmo_cutouts, hydro_cutouts, gen_cutouts,
                                       output_dir):
    """
    Plot comparison between hydro and generated halos for parameter extremes.
    Placeholder function for backward compatibility.
    
    Parameters:
    -----------
    extreme_type : str
        Type of extreme ('min' or 'max')
    param_name : str
        Name of the parameter
    param_value : float
        Value of the parameter at this extreme
    dmo_cutouts : ndarray
        DMO cutouts for these halos
    hydro_cutouts : ndarray
        Hydro cutouts
    gen_cutouts : ndarray
        Generated cutouts
    output_dir : str
        Directory to save plots
    """
    print(f"Parameter extreme analysis for {extreme_type} {param_name} = {param_value}")
    print("This function is a placeholder - implement if needed for parameter studies")
    pass


def run_all_analyses(full_dmo, full_hydro, hydro_replaced, final_maps,
                     dmo_cutouts, hydro_cutouts, gen_cutouts,
                     halo_radii, boxsize, output_dir, model_name):
    """
    Run all comprehensive analyses matching BIND_usage_CV.ipynb.
    
    Parameters:
    -----------
    full_dmo : ndarray
        Full DMO box projection
    full_hydro : ndarray
        Full hydro box projection
    hydro_replaced : ndarray
        Hydro replacement map
    final_maps : list of ndarray
        List of BIND generated full box maps (multiple realizations)
    dmo_cutouts : ndarray
        DMO halo cutouts, shape (n_halos, H, W)
    hydro_cutouts : ndarray
        Hydro halo cutouts, shape (n_halos, 3, H, W) [DM, Gas, Stars]
    gen_cutouts : ndarray
        Generated halo cutouts, shape (n_halos, n_realizations, 3, H, W)
    halo_radii : list
        Virial radii in pixels
    boxsize : float
        Box size in Mpc/h
    output_dir : str
        Output directory for plots
    model_name : str
        Model name
    """
    print(f"\n{'='*60}")
    print(f"Running comprehensive analyses for {model_name}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Full box comparison maps
    print("1. Creating full box comparison maps...")
    plot_full_box_comparison(full_dmo, full_hydro, hydro_replaced, final_maps, 
                             output_dir, model_name)
    
    # 2. Full box power spectrum analysis
    print("\n2. Computing full box power spectra...")
    compute_power_spectrum_fullbox(full_hydro, full_dmo, hydro_replaced, final_maps,
                                   boxsize, output_dir, model_name)
    
    # 3. Density profiles and mass comparisons
    print("\n3. Computing density profiles and mass comparisons...")
    plot_density_profiles_and_mass(hydro_cutouts, gen_cutouts, halo_radii,
                                   output_dir, model_name)
    
    # 4. Halo-level power spectra
    print("\n4. Computing halo-level power spectra...")
    compute_halo_level_power_spectra(hydro_cutouts, gen_cutouts, boxsize,
                                     output_dir, model_name)
    
    # 5. Residual histograms
    print("\n5. Computing residual histograms...")
    plot_residual_histograms(hydro_cutouts, gen_cutouts, dmo_cutouts,
                            output_dir, model_name)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete for {model_name}")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}\n")
