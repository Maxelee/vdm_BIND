#!/usr/bin/env python
"""
Generate synthetic test data for VDM-BIND testing.

This script creates small synthetic datasets that mimic the structure of real
training data but are much smaller and faster to process. Useful for:
- CI/CD pipeline testing
- Quick sanity checks
- Development without access to full dataset

Usage:
    python scripts/generate_synthetic_data.py --output tests/fixtures/synthetic_data --n_samples 10

Author: VDM-BIND Team
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple


def generate_halo_field(
    size: int = 128,
    center: Tuple[float, float] = (0.5, 0.5),
    mass: float = 1e14,
    r200c: float = 0.2,
    noise_level: float = 0.1,
    seed: int = None
) -> np.ndarray:
    """
    Generate a synthetic 2D halo density field.
    
    Creates a NFW-like profile with noise.
    
    Args:
        size: Image size in pixels
        center: Halo center in normalized coordinates [0, 1]
        mass: Halo mass (affects amplitude)
        r200c: Virial radius in normalized coordinates
        noise_level: Noise amplitude relative to signal
        seed: Random seed for reproducibility
    
    Returns:
        2D density field of shape (size, size)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create coordinate grid
    y, x = np.mgrid[0:size, 0:size] / size
    
    # Distance from center
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # NFW-like profile: rho ~ 1 / (r * (1 + r)^2)
    r_scaled = r / r200c + 0.1  # Avoid singularity at center
    profile = 1.0 / (r_scaled * (1 + r_scaled)**2)
    
    # Scale by mass
    amplitude = np.log10(mass) - 10  # Normalize to reasonable values
    field = profile * amplitude
    
    # Add some substructure (smaller halos)
    n_subhalos = np.random.randint(2, 6)
    for _ in range(n_subhalos):
        sub_center = (
            center[0] + np.random.uniform(-r200c, r200c),
            center[1] + np.random.uniform(-r200c, r200c)
        )
        sub_r = np.sqrt((x - sub_center[0])**2 + (y - sub_center[1])**2)
        sub_r200 = r200c * np.random.uniform(0.1, 0.3)
        sub_r_scaled = sub_r / sub_r200 + 0.1
        sub_profile = 1.0 / (sub_r_scaled * (1 + sub_r_scaled)**2)
        field += sub_profile * amplitude * np.random.uniform(0.1, 0.3)
    
    # Add noise
    noise = np.random.normal(0, noise_level * np.abs(field).mean(), field.shape)
    field = field + noise
    
    # Ensure non-negative
    field = np.maximum(field, 0)
    
    return field.astype(np.float32)


def generate_gas_field(dm_field: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Generate gas field correlated with DM but smoother.
    
    Args:
        dm_field: Dark matter density field
        seed: Random seed
    
    Returns:
        Gas density field
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Gas follows DM but is smoother (convolved)
    from scipy.ndimage import gaussian_filter
    gas = gaussian_filter(dm_field, sigma=3)
    
    # Add some offset and noise
    gas = gas * np.random.uniform(0.8, 1.2)
    gas += np.random.normal(0, 0.05 * gas.std(), gas.shape)
    gas = np.maximum(gas, 0)
    
    return gas.astype(np.float32)


def generate_stellar_field(dm_field: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Generate stellar field - concentrated near center, sparse.
    
    Args:
        dm_field: Dark matter density field
        seed: Random seed
    
    Returns:
        Stellar density field (sparse)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Stars are more concentrated than DM
    size = dm_field.shape[0]
    y, x = np.mgrid[0:size, 0:size] / size
    center = (0.5, 0.5)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Steep profile for stars
    stellar = dm_field * np.exp(-r / 0.1)
    
    # Make it sparse - threshold low values
    threshold = np.percentile(stellar, 80)
    stellar = np.where(stellar > threshold, stellar, 0)
    
    # Add Poisson-like noise to non-zero values
    mask = stellar > 0
    stellar[mask] *= np.random.uniform(0.5, 1.5, mask.sum())
    
    return stellar.astype(np.float32)


def generate_large_scale_maps(
    size: int = 128,
    n_scales: int = 3,
    seed: int = None
) -> np.ndarray:
    """
    Generate large-scale conditioning maps.
    
    Args:
        size: Image size
        n_scales: Number of large-scale maps (typically 3)
        seed: Random seed
    
    Returns:
        Large-scale maps of shape (n_scales, size, size)
    """
    if seed is not None:
        np.random.seed(seed)
    
    maps = []
    for i in range(n_scales):
        # Different smoothing scales
        sigma = 5 * (i + 1)  # 5, 10, 15 pixel smoothing
        
        # Generate random field and smooth
        field = np.random.randn(size, size)
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(field, sigma=sigma)
        
        # Normalize
        field = (field - field.mean()) / (field.std() + 1e-8)
        maps.append(field)
    
    return np.stack(maps, axis=0).astype(np.float32)


def generate_synthetic_sample(
    size: int = 128,
    n_params: int = 35,
    seed: int = None
) -> dict:
    """
    Generate a complete synthetic training sample.
    
    Args:
        size: Image size in pixels
        n_params: Number of conditional parameters
        seed: Random seed
    
    Returns:
        Dictionary with all training data fields
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random halo properties
    halo_mass = 10**np.random.uniform(13, 15)  # 10^13 to 10^15 M_sun
    center = (np.random.uniform(0.4, 0.6), np.random.uniform(0.4, 0.6))
    r200c = np.random.uniform(0.1, 0.3)
    
    # Generate fields
    dm = generate_halo_field(size, center, halo_mass, r200c, seed=seed)
    dm_hydro = generate_halo_field(size, center, halo_mass * 0.95, r200c * 1.1, seed=seed+1 if seed else None)
    gas = generate_gas_field(dm, seed=seed+2 if seed else None)
    star = generate_stellar_field(dm, seed=seed+3 if seed else None)
    
    # Large-scale conditioning
    large_scale = generate_large_scale_maps(size, n_scales=3, seed=seed+4 if seed else None)
    
    # Conditional parameters (cosmological + astrophysical)
    params = np.random.uniform(0, 1, n_params).astype(np.float32)
    
    return {
        'dm': dm,                    # DM condition (input)
        'dm_hydro': dm_hydro,        # DM from hydro sim (target 0)
        'gas': gas,                  # Gas density (target 1)
        'star': star,                # Stellar density (target 2)
        'large_scale_dm_12': large_scale[0],
        'large_scale_dm_25': large_scale[1],
        'large_scale_dm_50': large_scale[2],
        'conditional_params': params,
        'halo_mass': np.float32(halo_mass),
        'halo_center': np.array(center, dtype=np.float32),
    }


def generate_synthetic_dataset(
    output_dir: str,
    n_samples: int = 10,
    size: int = 128,
    n_params: int = 35,
    seed: int = 42
) -> None:
    """
    Generate a synthetic dataset for testing.
    
    Args:
        output_dir: Output directory
        n_samples: Number of samples to generate
        size: Image size
        n_params: Number of conditional parameters
        seed: Base random seed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {n_samples} synthetic samples...")
    
    for i in range(n_samples):
        sample = generate_synthetic_sample(size, n_params, seed=seed + i)
        
        # Save as npz
        filename = output_dir / f"synthetic_halo_{i:03d}.npz"
        np.savez(filename, **sample)
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples")
    
    print(f"✓ Saved {n_samples} samples to {output_dir}")
    
    # Generate normalization stats for this synthetic data
    print("\nComputing normalization statistics...")
    
    dm_values, gas_values, star_values = [], [], []
    for i in range(n_samples):
        sample = np.load(output_dir / f"synthetic_halo_{i:03d}.npz")
        dm_values.append(np.log10(sample['dm'] + 1).flatten())
        gas_values.append(np.log10(sample['gas'] + 1).flatten())
        star_values.append(np.log10(sample['star'] + 1).flatten())
    
    dm_all = np.concatenate(dm_values)
    gas_all = np.concatenate(gas_values)
    star_all = np.concatenate(star_values)
    
    # Save stats
    np.savez(
        output_dir / 'dark_matter_normalization_stats.npz',
        dm_mag_mean=dm_all.mean(),
        dm_mag_std=dm_all.std()
    )
    np.savez(
        output_dir / 'gas_normalization_stats.npz',
        gas_mag_mean=gas_all.mean(),
        gas_mag_std=gas_all.std()
    )
    np.savez(
        output_dir / 'stellar_normalization_stats.npz',
        star_mag_mean=star_all.mean(),
        star_mag_std=star_all.std()
    )
    
    print(f"  DM:     mean={dm_all.mean():.4f}, std={dm_all.std():.4f}")
    print(f"  Gas:    mean={gas_all.mean():.4f}, std={gas_all.std():.4f}")
    print(f"  Stellar: mean={star_all.mean():.4f}, std={star_all.std():.4f}")
    print(f"✓ Saved normalization stats to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for VDM-BIND",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='tests/fixtures/synthetic_data',
        help='Output directory (default: tests/fixtures/synthetic_data)'
    )
    
    parser.add_argument(
        '--n_samples', '-n',
        type=int,
        default=10,
        help='Number of samples to generate (default: 10)'
    )
    
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=128,
        help='Image size in pixels (default: 128)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        args.output,
        n_samples=args.n_samples,
        size=args.size,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
