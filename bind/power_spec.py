# Power Spectrum Analysis Functions
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift, fftfreq
import Pk_library as PKL

def compute_power_spectrum_simple(image, BoxSize=50.0, MAS='CIC', threads=0):
    """
    Compute 2D power spectrum using pylians Pk_library.
    
    Parameters:
    - image: 2D numpy array of the density field
    - BoxSize: Size of the box in Mpc/h
    - MAS: Mass assignment scheme ('NGP', 'CIC', 'TSC', 'PCS')
    - threads: Number of threads
    
    Returns:
    - Pk: 1D array of power spectrum values
    - k: 1D array of wavenumber values
    """
    # Compute overdensity
    delta = image / np.mean(image, dtype=np.float64)
    delta -= 1.0
    
    # Compute power spectrum
    Pk2D = PKL.Pk_plane(delta, BoxSize, MAS, threads)
    
    return Pk2D.Pk, Pk2D.k


def compute_power_spectrum_log(image):
    """
    Alternative power spectrum computation using log-transform.
    Use this for density field analysis where log-normal distribution is expected.
    
    Parameters:
    - image: 2D numpy array of the density field
    
    Returns:
    - power_1d: 1D array of radially averaged power spectrum in log space
    """
    # Ensure input is finite and positive
    image = np.nan_to_num(image, nan=1e-12, posinf=image[np.isfinite(image)].max(), neginf=1e-12)
    image = np.maximum(image, 1e-12)
    
    # Take log for density field analysis
    log_image = np.log(image)
    
    # Remove mean to avoid DC spike
    log_image = log_image - np.mean(log_image)
    
    # Compute 2D FFT
    fft_result = fft2(log_image)
    power_2d = np.abs(fft_result)**2
    
    # Create frequency coordinates
    ny, nx = image.shape
    kx = fftfreq(nx)
    ky = fftfreq(ny)
    kx_2d, ky_2d = np.meshgrid(kx, ky)
    k_2d = np.sqrt(kx_2d**2 + ky_2d**2)
    
    # Radial binning
    k_max = min(nx, ny) // 2
    k_bins = np.arange(0, k_max + 1)
    power_1d = np.zeros(len(k_bins) - 1)
    
    for i in range(len(k_bins) - 1):
        mask = (k_2d >= k_bins[i]) & (k_2d < k_bins[i + 1])
        if np.any(mask):
            power_1d[i] = np.mean(power_2d[mask])
        else:
            power_1d[i] = 0.0
    
    # Handle any remaining NaN or inf values
    power_1d = np.nan_to_num(power_1d, nan=0.0, posinf=0.0, neginf=0.0)
    
    return power_1d


def analyze_power_preservation(ps_original, ps_method, method_name="Method"):
    """
    Analyze power preservation performance of a halo blending method.
    
    Parameters:
    - ps_original: Power spectrum of original (DMO) map
    - ps_method: Power spectrum of method result
    - method_name: Name of the method for reporting
    
    Returns:
    - analysis_dict: Dictionary with performance metrics
    """
    # Compute power ratio
    ratio = np.divide(ps_method, ps_original, out=np.ones_like(ps_method), where=(ps_original > 0))
    
    # Find first valid k mode
    first_valid_k = None
    for k in range(len(ps_original)):
        if ps_original[k] > 0:
            first_valid_k = k
            break
    
    analysis = {
        'method_name': method_name,
        'first_valid_k': first_valid_k,
        'k1_ratio': ratio[first_valid_k] if first_valid_k is not None else 0.0,
        'k1_suppression_pct': (1 - ratio[first_valid_k]) * 100 if first_valid_k is not None else 100.0,
        'power_ratio': ratio,
        'valid_modes': np.sum(ps_original > 0)
    }
    
    # Calculate additional metrics if enough valid modes
    if first_valid_k is not None and len(ratio) > first_valid_k + 5:
        analysis['k2_5_avg'] = np.mean(ratio[first_valid_k+1:first_valid_k+5])
        analysis['k6_15_avg'] = np.mean(ratio[first_valid_k+5:min(first_valid_k+15, len(ratio))])
    
    return analysis



def plot_power_spectrum_ratios(original_map, target_map, pasted_map, box_size_kpc):
    """Plot power spectrum ratios: target/original and pasted/original."""
    print("🔍 Computing power spectra...")
    
    # Compute power spectra using pylians
    ps_orig, k = compute_power_spectrum_simple(original_map, BoxSize=box_size_kpc/1000.0)
    ps_target, _ = compute_power_spectrum_simple(target_map, BoxSize=box_size_kpc/1000.0)
    ps_pasted, _ = compute_power_spectrum_simple(pasted_map, BoxSize=box_size_kpc/1000.0)
    
    # Compute ratios
    ratio_target_orig = np.divide(ps_target, ps_orig, out=np.ones_like(ps_target), where=(ps_orig > 0))
    ratio_pasted_orig = np.divide(ps_pasted, ps_orig, out=np.ones_like(ps_pasted), where=(ps_orig > 0))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Power spectra
    ax1.loglog(k[1:], ps_orig[1:], 'k-', label='Original (DMO)', linewidth=2, alpha=0.8)
    ax1.loglog(k[1:], ps_target[1:], 'r-', label='Target (Hydro)', linewidth=2, alpha=0.8)
    ax1.loglog(k[1:], ps_pasted[1:], 'b-', label='Pasted (Priority: Largest Halos)', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('k [h/Mpc]', fontsize=12)
    ax1.set_ylabel('Power Spectrum', fontsize=12)
    ax1.set_title('Power Spectra Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratios
    ax2.semilogx(k[1:], ratio_target_orig[1:], 'r-', label='Target/Original', linewidth=2.5, alpha=0.8)
    ax2.semilogx(k[1:], ratio_pasted_orig[1:], 'b-', label='Pasted/Original', linewidth=2.5, alpha=0.8)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Perfect Recovery')
    
    ax2.set_xlabel('k [h/Mpc]', fontsize=12)
    ax2.set_ylabel('Power Spectrum Ratio', fontsize=12)
    ax2.set_title('Power Spectrum Ratios (Large Halo Priority)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.1, 1.5)  # Reasonable range for ratios
    
    plt.tight_layout()
    plt.show()
    
    # Analyze power preservation using the new analysis function
    target_analysis = analyze_power_preservation(ps_orig, ps_target, "Target (Hydro)")
    pasted_analysis = analyze_power_preservation(ps_orig, ps_pasted, "Pasted (Large Halo Priority)")
    
    # Print statistics
    print(f"\n📊 Power Spectrum Analysis:")
    print(f"   Target k=1 ratio: {target_analysis['k1_ratio']:.3f} ({target_analysis['k1_suppression_pct']:.1f}% suppression)")
    print(f"   Pasted k=1 ratio: {pasted_analysis['k1_ratio']:.3f} ({pasted_analysis['k1_suppression_pct']:.1f}% suppression)")
    print(f"   Recovery efficiency: {pasted_analysis['k1_ratio']/target_analysis['k1_ratio']*100:.1f}%")
    
    # Print additional metrics if available
    if 'k2_5_avg' in target_analysis:
        print(f"\n🔵 k=2-5 average ratios:")
        print(f"   Target: {target_analysis['k2_5_avg']:.3f}")
        print(f"   Pasted: {pasted_analysis['k2_5_avg']:.3f}")
    
    if 'k6_15_avg' in target_analysis:
        print(f"\n🔴 k=6-15 average ratios:")
        print(f"   Target: {target_analysis['k6_15_avg']:.3f}")
        print(f"   Pasted: {pasted_analysis['k6_15_avg']:.3f}")
    
    return k, ratio_target_orig, ratio_pasted_orig


def plot_map_comparison(original_map, target_map, pasted_map, box_size_kpc):
    """Plot visual comparison of original, target, and pasted maps."""
    print("🖼️  Creating map comparison...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define common parameters for all plots
    extent = [0, box_size_kpc/1000, 0, box_size_kpc/1000]  # Convert to Mpc for display
    
    # Top row: Original scale images
    vmin_orig = np.percentile([original_map, target_map, pasted_map], 1)
    vmax_orig = np.percentile([original_map, target_map, pasted_map], 99)
    
    im1 = axes[0,0].imshow(np.log10(np.maximum(original_map, 1e-12)), 
                          extent=extent, origin='lower', cmap='viridis', 
                          vmin=np.log10(vmin_orig), vmax=np.log10(vmax_orig))
    axes[0,0].set_title('Original (DMO)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Mpc')
    axes[0,0].set_ylabel('Mpc')
    plt.colorbar(im1, ax=axes[0,0], label='log₁₀(density)')
    
    im2 = axes[0,1].imshow(np.log10(np.maximum(target_map, 1e-12)), 
                          extent=extent, origin='lower', cmap='viridis',
                          vmin=np.log10(vmin_orig), vmax=np.log10(vmax_orig))
    axes[0,1].set_title('Target (Hydro)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Mpc')
    axes[0,1].set_ylabel('Mpc')
    plt.colorbar(im2, ax=axes[0,1], label='log₁₀(density)')
    
    im3 = axes[0,2].imshow(np.log10(np.maximum(pasted_map, 1e-12)), 
                          extent=extent, origin='lower', cmap='viridis',
                          vmin=np.log10(vmin_orig), vmax=np.log10(vmax_orig))
    axes[0,2].set_title('Pasted (Large Halo Priority)', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('Mpc')
    axes[0,2].set_ylabel('Mpc')
    plt.colorbar(im3, ax=axes[0,2], label='log₁₀(density)')
    
    # Bottom row: Difference maps
    diff_target = np.log10(np.maximum(target_map, 1e-12)) - np.log10(np.maximum(original_map, 1e-12))
    diff_pasted = np.log10(np.maximum(pasted_map, 1e-12)) - np.log10(np.maximum(original_map, 1e-12))
    diff_residual = np.log10(np.maximum(target_map, 1e-12)) - np.log10(np.maximum(pasted_map, 1e-12))
    
    # Use symmetric color scale for differences
    diff_max = max(np.abs(np.percentile(diff_target, [5, 95])).max(),
                   np.abs(np.percentile(diff_pasted, [5, 95])).max(),
                   np.abs(np.percentile(diff_residual, [5, 95])).max())
    
    im4 = axes[1,0].imshow(diff_target, extent=extent, origin='lower', 
                          cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[1,0].set_title('Target - Original', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Mpc')
    axes[1,0].set_ylabel('Mpc')
    plt.colorbar(im4, ax=axes[1,0], label='Δlog₁₀(density)')
    
    im5 = axes[1,1].imshow(diff_pasted, extent=extent, origin='lower', 
                          cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[1,1].set_title('Pasted - Original', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Mpc')
    axes[1,1].set_ylabel('Mpc')
    plt.colorbar(im5, ax=axes[1,1], label='Δlog₁₀(density)')
    
    im6 = axes[1,2].imshow(diff_residual, extent=extent, origin='lower', 
                          cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[1,2].set_title('Target - Pasted (Residual)', fontsize=14, fontweight='bold')
    axes[1,2].set_xlabel('Mpc')
    axes[1,2].set_ylabel('Mpc')
    plt.colorbar(im6, ax=axes[1,2], label='Δlog₁₀(density)')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\n📈 Map Statistics:")
    print(f"   Original map range: [{original_map.min():.2e}, {original_map.max():.2e}]")
    print(f"   Target map range: [{target_map.min():.2e}, {target_map.max():.2e}]")
    print(f"   Pasted map range: [{pasted_map.min():.2e}, {pasted_map.max():.2e}]")
    
    # Calculate correlation coefficients
    corr_target_orig = np.corrcoef(original_map.flatten(), target_map.flatten())[0,1]
    corr_pasted_orig = np.corrcoef(original_map.flatten(), pasted_map.flatten())[0,1]
    corr_pasted_target = np.corrcoef(target_map.flatten(), pasted_map.flatten())[0,1]
    
    print(f"\n🔗 Correlation Coefficients:")
    print(f"   Target ↔ Original: {corr_target_orig:.4f}")
    print(f"   Pasted ↔ Original: {corr_pasted_orig:.4f}")
    print(f"   Pasted ↔ Target: {corr_pasted_target:.4f}")
    
    return diff_target, diff_pasted, diff_residual