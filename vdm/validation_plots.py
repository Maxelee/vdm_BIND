"""
Validation plotting for progressive field weighting training.
Generates comprehensive comparison plots between generated and true fields.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Optional
import Pk_library as PKL



def unnormalize_field(field, mean, std):
    """Unnormalize a field from standardized space."""
    return (field * std) + mean


def compute_binded_power(binded, boxsize=6.25):
    """
    Compute power spectrum from 2D fields.
    
    Parameters:
    -----------
    binded : array
        Array of 2D fields, shape (N, H, W)
    boxsize : float
        Physical box size in Mpc/h
    
    Returns:
    --------
    k : array
        Wavenumbers
    Pk_binded : array
        Power spectra for each field
    Nmodes : array
        Number of modes in each bin
    """
    binded = np.array(binded, dtype=np.float64)
    delta_binded = binded / np.mean(binded, dtype=np.float64, axis=(1,2), keepdims=True)
    delta_binded -= 1.0
    delta_binded = delta_binded.astype(np.float32)
    
    # parameters
    grid = 128  # the map will have grid^2 pixels
    BoxSize = boxsize  # Mpc/h
    MAS = 'NGP'  # MAS used to create the image
    threads = 0  # number of openmp threads
    
    Pk2D_binded = [PKL.Pk_plane(delta_binded[i], BoxSize, MAS, threads) for i in range(binded.shape[0])]
    k = Pk2D_binded[0].k
    Nmodes = Pk2D_binded[0].Nmodes
    Pk_binded = np.array([Pk2D_binded[i].Pk for i in range(binded.shape[0])])
    
    return k, Pk_binded, Nmodes


def get_projected_surface_density(halo_mass, radius_pix, size=128, nbins=15):
    """
    Calculate projected surface density profile in logarithmic radial bins.
    
    Parameters:
    -----------
    halo_mass : array
        2D mass distribution (128x128)
    radius_pix : float
        Virial radius in pixels
    size : int
        Size of the patch (default 128)
    nbins : int
        Number of radial bins
    
    Returns:
    --------
    surface_densities : array
        Surface density in each annular bin
    bin_centers : array
        Radial bin centers in units of virial radius
    """
    # Define radial bins in units of virial radius
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


class ValidationPlotter:
    """Generate validation plots for multi-field predictions."""
    
    def __init__(self, dataset='IllustrisTNG', boxsize=6.25, stellar_stats_path=None, 
                 quantile_transformer_path=None):
        """
        Initialize validation plotter.
        
        Parameters:
        -----------
        dataset : str
            Dataset name for normalization constants
        boxsize : float
            Physical box size in Mpc/h
        stellar_stats_path : str, optional
            Path to stellar normalization stats file (.npz)
            If provided, will use 4-channel mode with mask + magnitude
        quantile_transformer_path : str, optional
            Path to quantile transformer pickle file (.pkl)
            If provided, will use quantile normalization for stellar channel instead of Z-score
        """
        self.dataset = dataset
        self.boxsize = boxsize
        
        # Import normalization constants from vdm.constants
        from vdm.constants import norms_256
        self.norms = norms_256[dataset]
        
        # Load quantile transformer if provided
        self.quantile_transformer = None
        if quantile_transformer_path is not None and os.path.exists(quantile_transformer_path):
            import joblib
            self.quantile_transformer = joblib.load(quantile_transformer_path)
            print(f"ValidationPlotter: Loaded quantile transformer from {quantile_transformer_path}")
        
        # Check if we should use 4-channel mode
        self.use_4_channel = stellar_stats_path is not None and os.path.exists(stellar_stats_path)
        
        if self.use_4_channel:
            # Load stellar statistics for 4-channel reconstruction
            stellar_stats = np.load(stellar_stats_path)
            self.star_mag_mean = float(stellar_stats['star_mag_mean'])
            self.star_mag_std = float(stellar_stats['star_mag_std'])
            self.star_epsilon = float(stellar_stats['epsilon'])
            
            # Normalization order in norms_256: [mean_star, std_star, mean_gas, std_gas, mean_dmh, std_dmh, mean_dm, std_dm]
            # 4-channel order: [dmh, gas, star_mask, star_magnitude]
            # We still need gas and dmh norms, but not star norms (using new approach)
            self.mean_target = np.array([self.norms[4], self.norms[2]])  # [mean_dmh, mean_gas]
            self.std_target = np.array([self.norms[5], self.norms[3]])   # [std_dmh, std_gas]
            
            # Field names for plotting - in 4-channel mode
            self.field_names = ['DMO', 'Gas', 'Stars (Reconstructed)', 'Star Mask', 'Star Magnitude']
            self.field_colors = ['blue', 'green', 'red', 'purple', 'orange']
            
            print(f"ValidationPlotter: Using 4-channel mode with stellar stats from {stellar_stats_path}")
            print(f"  star_mag_mean={self.star_mag_mean:.4f}, star_mag_std={self.star_mag_std:.4f}, epsilon={self.star_epsilon}")
        else:
            # 3-channel mode (legacy)
            # Normalization order in norms_256: [mean_star, std_star, mean_gas, std_gas, mean_dmh, std_dmh, mean_dm, std_dm]
            # Target order in model: [dmh, gas, star] (channels 0, 1, 2)
            # So we need to map: channel 0 -> dmh (index 4,5), channel 1 -> gas (index 2,3), channel 2 -> star (index 0,1)
            
            self.mean_target = np.array([self.norms[4], self.norms[2], self.norms[0]])  # [mean_dmh, mean_gas, mean_star]
            self.std_target = np.array([self.norms[5], self.norms[3], self.norms[1]])   # [std_dmh, std_gas, std_star]
            
            # Field names for plotting - in the order they appear in the tensor
            self.field_names = ['DMO', 'Gas', 'Stars']
            self.field_colors = ['blue', 'green', 'red']
            
            print(f"ValidationPlotter: Using 3-channel mode (legacy)")
    
    def unnormalize_targets(self, x_tensor, param_conditioning=None):
        """
        Unnormalize target fields from log-normalized space.
        
        Parameters:
        -----------
        x_tensor : torch.Tensor
            Normalized targets
            - 3-channel mode: shape (B, 3, H, W), order [dmh, gas, star]
            - 4-channel mode: shape (B, 4, H, W), order [dmh, gas, star_mask, star_magnitude]
        param_conditioning : torch.Tensor, optional
            Parameter conditioning tensor, shape (B, N_params)
            We need param_conditioning[:, 0] (Omega_m) and param_conditioning[:, 6] (Omega_b)
        
        Returns:
        --------
        unnorm : np.array
            Unnormalized fields in linear space (physical units)
            - 3-channel mode: shape (B, 3, H, W), order [dmh, gas, star]
            - 4-channel mode: shape (B, 3, H, W), order [dmh, gas, star_reconstructed]
        """
        x_np = x_tensor.cpu().numpy()
        
        # Auto-detect mode based on number of channels
        n_channels = x_np.shape[1]
        is_4channel_input = (n_channels == 4)
        is_3channel_input = (n_channels == 3)
        
        if is_4channel_input:
            # Input has 4 channels: [dmh, gas, star_mask, star_magnitude]
            # Reconstruct stars from mask and magnitude
            
            # Unnormalize DMO and Gas (channels 0, 1)
            # x_np shape: (B, 4, H, W)
            # mean/std shape: (2,) in 4-channel mode, (3,) in 3-channel mode
            if self.use_4_channel:
                mean = self.mean_target.reshape(1, -1, 1, 1)  # Shape: (1, 2, 1, 1)
                std = self.std_target.reshape(1, -1, 1, 1)    # Shape: (1, 2, 1, 1)
            else:
                # Fallback: use first 2 elements of 3-channel norms
                mean = self.mean_target[:2].reshape(1, -1, 1, 1)
                std = self.std_target[:2].reshape(1, -1, 1, 1)
            
            unnorm_dm_gas = (x_np[:, :2] * std) + mean
            
            # Convert from log10 space to linear space for DMO and Gas
            unnorm_dm_gas = 10**unnorm_dm_gas - 1.0
            
            # Reconstruct stars from mask (channel 2) and magnitude (channel 3)
            star_mask = x_np[:, 2:3]  # Shape: (B, 1, H, W), binary mask
            star_magnitude = x_np[:, 3:4]  # Shape: (B, 1, H, W), normalized magnitude
            
            # Use stellar stats if available, otherwise use defaults
            if self.use_4_channel:
                star_mag_mean = self.star_mag_mean
                star_mag_std = self.star_mag_std
                star_epsilon = self.star_epsilon
            else:
                # Fallback defaults (shouldn't happen, but just in case)
                star_mag_mean = 6.995923
                star_mag_std = 1.100363
                star_epsilon = 1e-8
            
            # Denormalize magnitude: magnitude_log = (magnitude_normalized * std) + mean
            star_magnitude_log = (star_magnitude * star_mag_std) + star_mag_mean
            
            # Convert from log space to linear: stars_linear = 10^magnitude_log - epsilon
            star_magnitude_linear = 10**star_magnitude_log - star_epsilon
            
            # Apply mask: stars = mask * magnitude_linear
            stars_reconstructed = star_mask * star_magnitude_linear
            
            # Combine: [dmh, gas, stars_reconstructed]
            unnorm = np.concatenate([unnorm_dm_gas, stars_reconstructed], axis=1)
            
            # Multiply back by cosmological parameters
            if param_conditioning is not None:
                param_np = param_conditioning.cpu().numpy()
                omega_m = param_np[:, 0].reshape(-1, 1, 1, 1)
                omega_b = param_np[:, 6].reshape(-1, 1, 1, 1)
                
                unnorm[:, 0:1] *= omega_m  # DMO channel
                unnorm[:, 1:3] *= omega_b  # Gas and Stars channels
            
            return unnorm
            
        elif is_3channel_input:
            # Input has 3 channels: [dmh, gas, star]
            # This can happen in two cases:
            # 1. Legacy 3-channel mode
            # 2. 4-channel mode where model already reconstructed stars
            
            if self.use_4_channel:
                # Case 2: 4-channel mode, but input already reconstructed to 3 channels
                # We need to unnormalize using 3-channel norms from constants
                # Load 3-channel norms as fallback
                from vdm.constants import norms_256
                norms_3ch = norms_256[self.dataset]
                # Order: [mean_star, std_star, mean_gas, std_gas, mean_dmh, std_dmh, mean_dm, std_dm]
                # We need: [mean_dmh, mean_gas, mean_star], [std_dmh, std_gas, std_star]
                mean_3ch = np.array([norms_3ch[4], norms_3ch[2], norms_3ch[0]])  # [dmh, gas, star]
                std_3ch = np.array([norms_3ch[5], norms_3ch[3], norms_3ch[1]])
                
                mean = mean_3ch.reshape(1, -1, 1, 1)
                std = std_3ch.reshape(1, -1, 1, 1)
            else:
                # Case 1: Legacy 3-channel mode
                # x_np shape: (B, 3, H, W)
                # mean/std shape: (3,) -> reshape to (1, 3, 1, 1)
                mean = self.mean_target.reshape(1, -1, 1, 1)
                std = self.std_target.reshape(1, -1, 1, 1)
            
            unnorm = (x_np * std) + mean
            
            # Convert from log10 space to linear space
            # Special handling for stellar channel if using quantile normalization
            if self.quantile_transformer is not None:
                # Stellar channel (index 2) needs quantile inverse transform
                # DM and Gas (indices 0, 1) use standard Z-score denormalization above
                stellar_norm = x_np[:, 2:3]  # Shape: (B, 1, H, W)
                
                # Apply inverse quantile transformation
                original_shape = stellar_norm.shape
                stellar_flat = stellar_norm.reshape(-1, 1)
                stellar_unnorm_flat = self.quantile_transformer.inverse_transform(stellar_flat)
                stellar_unnorm = stellar_unnorm_flat.reshape(original_shape)
                
                # Replace stellar channel with quantile-denormalized version
                # Note: quantile transformer output is already in log10 space
                unnorm[:, 2:3] = stellar_unnorm
            
            # Convert from log10 space to linear space
            unnorm = 10**unnorm - 1.0
            
            # Multiply back by cosmological parameters (reverse the division in astro_dataset.py)
            if param_conditioning is not None:
                param_np = param_conditioning.cpu().numpy()
                # Channel 0 (DMO) was divided by Omega_m (param[0])
                # Channels 1,2 (Gas, Stars) were divided by Omega_b (param[6])
                # Reshape to (B, 1, 1, 1) for broadcasting
                omega_m = param_np[:, 0].reshape(-1, 1, 1, 1)
                omega_b = param_np[:, 6].reshape(-1, 1, 1, 1)
                
                unnorm[:, 0:1] *= omega_m  # DMO channel
                unnorm[:, 1:3] *= omega_b  # Gas and Stars channels
            
            return unnorm
        
        else:
            # Unexpected number of channels
            raise ValueError(f"Unexpected number of channels: {n_channels}. Expected 3 or 4, got {n_channels}")
    
    def generate_validation_plot(
        self,
        model,
        val_batch,
        global_step: int,
        n_samples: int = 4,
        n_power_samples: int = 16,
        save_path: Optional[str] = None
    ):
        """
        Generate comprehensive validation plot.
        
        Parameters:
        -----------
        model : LightVDM
            The model to validate
        val_batch : tuple
            Validation batch (m_dm, large_scale, x, param_conditioning)
        global_step : int
            Current training step
        n_samples : int
            Number of image samples to show
        n_power_samples : int
            Number of samples for power spectrum
        save_path : str
            Path to save figure (if None, returns figure)
        
        Returns:
        --------
        fig : matplotlib.Figure
            The generated figure
        """
        model.eval()
        
        with torch.no_grad():
            m_dm, large_scale, x_true, param_conditioning = val_batch
            
            # Ensure we don't exceed batch size
            n_samples = min(n_samples, x_true.shape[0])
            n_power_samples = min(n_power_samples, x_true.shape[0])
            
            # Generate samples
            conditioning = torch.cat([m_dm, large_scale], dim=1)
            x_gen = model.draw_samples(
                conditioning=conditioning[:n_power_samples],
                batch_size=n_power_samples,
                n_sampling_steps=model.hparams.n_sampling_steps,
                param_conditioning=param_conditioning[:n_power_samples] if param_conditioning is not None else None,
                verbose=False
            )
            
            # Unnormalize with parameter conditioning
            x_true_unnorm = self.unnormalize_targets(
                x_true[:n_power_samples],
                param_conditioning=param_conditioning[:n_power_samples] if param_conditioning is not None else None
            )
            x_gen_unnorm = self.unnormalize_targets(
                x_gen,
                param_conditioning=param_conditioning[:n_power_samples] if param_conditioning is not None else None
            )
            
            # Both modes produce 3-channel output: [dmh, gas, stars_reconstructed]
            # For 4-channel mode, stars are reconstructed from mask + magnitude
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.4)
            
            # Add title with current field weights and mode info
            title = f'Validation at Step {global_step}'
            if self.use_4_channel:
                title += ' (4-Channel Mode: Mask + Magnitude → Reconstructed Stars)'
            if hasattr(model.model, 'use_progressive_field_weighting') and model.model.use_progressive_field_weighting:
                field_weights = model.model.get_progressive_field_weights(device=x_true.device)
                title += f'\nField Weights - DMO: {field_weights[0]:.3f}, Gas: {field_weights[1]:.3f}, Stars: {field_weights[2]:.3f}'
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # 1. IMAGE COMPARISONS (Top 2 rows)
            # Always show the 3 main channels: [DMO, Gas, Stars]
            display_field_names = ['DMO', 'Gas', 'Stars']
            for i in range(min(n_samples, 4)):
                for ch_idx, ch_name in enumerate(display_field_names):
                    ax = fig.add_subplot(gs[i // 2, (i % 2) * 3 + ch_idx])
                    
                    true = x_true_unnorm[i, ch_idx]
                    gen = x_gen_unnorm[i, ch_idx]
                    
                    # Compute ratio (gen/true - 1)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = gen / (true + 1e-10) - 1.0
                        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Plot ratio with symmetric colormap
                    vmax = np.percentile(np.abs(ratio), 95)
                    im = ax.imshow(ratio, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
                    ax.set_title(f'{ch_name} - Sample {i+1}', fontsize=10)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Gen/True - 1')
            
            # 2. POWER SPECTRA (Row 3)
            try:
                display_field_names = ['DMO', 'Gas', 'Stars']
                display_field_colors = ['blue', 'green', 'red']
                
                for ch_idx, ch_name in enumerate(display_field_names):
                    ax = fig.add_subplot(gs[2, ch_idx * 2:(ch_idx * 2) + 2])
                    
                    # Compute power spectra
                    true_fields = x_true_unnorm[:n_power_samples, ch_idx]
                    gen_fields = x_gen_unnorm[:n_power_samples, ch_idx]
                    
                    k_true, Pk_true, _ = compute_binded_power(true_fields, self.boxsize)
                    k_gen, Pk_gen, _ = compute_binded_power(gen_fields, self.boxsize)
                    
                    # Plot mean and std
                    Pk_true_mean = np.mean(Pk_true, axis=0)
                    Pk_true_std = np.std(Pk_true, axis=0)
                    Pk_gen_mean = np.mean(Pk_gen, axis=0)
                    Pk_gen_std = np.std(Pk_gen, axis=0)
                    
                    # Power spectrum ratio
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = np.sqrt(Pk_gen_mean / (Pk_true_mean + 1e-20))
                        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
                    
                    ax.plot(k_true, ratio, 'o-', color=display_field_colors[ch_idx], 
                           linewidth=2, markersize=4, label=f'{ch_name}')
                    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
                    ax.fill_between(k_true, 0.9, 1.1, color='gray', alpha=0.2)
                    
                    ax.set_xscale('log')
                    ax.set_xlabel('k [h/Mpc]', fontsize=10)
                    ax.set_ylabel(r'$\sqrt{P_{gen}/P_{true}}$', fontsize=10)
                    ax.set_title(f'{ch_name} Power Spectrum Ratio', fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0.5, 1.5)
                    ax.legend()
            
            except Exception as e:
                print(f"Warning: Could not compute power spectra: {e}")
            
            # 3. DENSITY PROFILES (Row 4)
            # Assume virial radius ~ 10-20 pixels for typical halos
            try:
                display_field_names = ['DMO', 'Gas', 'Stars']
                display_field_colors = ['blue', 'green', 'red']
                
                radius_pix = 15.0  # Adjust based on your halos
                
                for ch_idx, ch_name in enumerate(display_field_names):
                    ax = fig.add_subplot(gs[3, ch_idx * 2:(ch_idx * 2) + 2])
                    
                    # Compute profiles for first few samples
                    n_profile_samples = min(4, n_power_samples)
                    
                    for i in range(n_profile_samples):
                        true_field = x_true_unnorm[i, ch_idx]
                        gen_field = x_gen_unnorm[i, ch_idx]
                        
                        surf_dens_true, bin_centers = get_projected_surface_density(
                            true_field, radius_pix, size=x_true.shape[-1]
                        )
                        surf_dens_gen, _ = get_projected_surface_density(
                            gen_field, radius_pix, size=x_gen.shape[-1]
                        )
                        
                        # Compute ratio
                        with np.errstate(divide='ignore', invalid='ignore'):
                            ratio = surf_dens_gen / (surf_dens_true + 1e-20) - 1.0
                            ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        alpha = 0.3 + 0.15 * i  # Vary transparency
                        ax.plot(bin_centers, ratio, 'o-', color=display_field_colors[ch_idx], 
                               alpha=alpha, linewidth=1.5, markersize=3)
                    
                    ax.axhline(0.0, color='black', linestyle='--', alpha=0.5)
                    ax.fill_between(bin_centers, -0.2, 0.2, color='gray', alpha=0.2)
                    
                    ax.set_xscale('log')
                    ax.set_xlabel('R/Rvir', fontsize=10)
                    ax.set_ylabel('Gen/True - 1', fontsize=10)
                    ax.set_title(f'{ch_name} Surface Density Profile', fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(-1.0, 1.0)
            
            except Exception as e:
                print(f"Warning: Could not compute density profiles: {e}")
            
            plt.tight_layout()
            
            if save_path is not None:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                model.train()
                return None
            
            model.train()
            return fig


def create_validation_callback(
    plotter: ValidationPlotter,
    val_dataloader,
    log_every_n_steps: int = 500,
    save_dir: str = 'validation_plots'
):
    """
    Create a callback function for validation plotting during training.
    
    Parameters:
    -----------
    plotter : ValidationPlotter
        The plotter instance
    val_dataloader : DataLoader
        Validation data loader
    log_every_n_steps : int
        How often to generate plots
    save_dir : str
        Directory to save plots
    
    Returns:
    --------
    callback : function
        Callback function to be called during training
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Get a fixed validation batch
    val_batch = next(iter(val_dataloader))
    
    def callback(trainer, model, global_step):
        if global_step % log_every_n_steps == 0:
            print(f"\n{'='*60}")
            print(f"Generating validation plot at step {global_step}...")
            print(f"{'='*60}")
            
            # Move batch to correct device
            device = next(model.parameters()).device
            val_batch_device = tuple(
                x.to(device) if x is not None and isinstance(x, torch.Tensor) else x 
                for x in val_batch
            )
            
            save_path = os.path.join(save_dir, f'validation_step_{global_step:06d}.png')
            
            try:
                plotter.generate_validation_plot(
                    model=model,
                    val_batch=val_batch_device,
                    global_step=global_step,
                    n_samples=4,
                    n_power_samples=16,
                    save_path=save_path
                )
                print(f"✓ Validation plot saved to: {save_path}")
            except Exception as e:
                print(f"✗ Error generating validation plot: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"{'='*60}\n")
    
    return callback, val_batch
