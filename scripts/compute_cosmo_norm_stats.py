#!/usr/bin/env python
"""
Compute normalization statistics for cosmologically-normalized fields.

When cosmo_norm=True, fields are divided by cosmological parameters BEFORE log transform:
- DM fields (condition, large_scale, target[0]): divided by Omega_m (param index 0)
- Baryonic fields (target[1], target[2]): divided by Omega_b (param index 6)

This script computes:
1. Mean and std of log10(field/param + 1) for each field type (for Z-score normalization)
2. Quantile transformer for stellar channel (since stellar uses quantile normalization)

Uses multiprocessing for fast parallel loading of training data.

Usage:
    python scripts/compute_cosmo_norm_stats.py [--n_samples 10000] [--output_dir data/]
    python scripts/compute_cosmo_norm_stats.py --fit_quantile --quantile_samples 100000
    python scripts/compute_cosmo_norm_stats.py --n_workers 32  # Use 32 parallel workers
"""

import argparse
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import joblib
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def process_single_file(fpath):
    """
    Process a single file and return statistics.
    Returns dict with sums, sum_of_squares, and counts for online mean/std computation.
    Also returns ALL stellar pixels for quantile fitting (not subsampled).
    """
    try:
        with np.load(fpath) as data:
            m_dm = data['condition']
            m_target = data['target']
            params = data['params']
            large_scale = data['large_scale']
            if large_scale.shape[0] == 4:
                large_scale = large_scale[1:]
            
            omega_m = params[0]  # Omega_m at index 0
            omega_b = params[6]  # Omega_b at index 6
            
            # Cosmo-normalized log transform: log10(field/param + 1)
            dm_cond_log = np.log10(m_dm / omega_m + 1)
            ls_log = np.log10(large_scale / omega_m + 1)
            dm_tgt_log = np.log10(m_target[0] / omega_m + 1)
            gas_log = np.log10(m_target[1] / omega_b + 1)
            star_log = np.log10(m_target[2] / omega_b + 1)
            
            # Return statistics for Welford's online algorithm
            return {
                'dm_cond': {'sum': dm_cond_log.sum(), 'sum_sq': (dm_cond_log**2).sum(), 'n': dm_cond_log.size},
                'large_scale': {'sum': ls_log.sum(), 'sum_sq': (ls_log**2).sum(), 'n': ls_log.size},
                'dm_target': {'sum': dm_tgt_log.sum(), 'sum_sq': (dm_tgt_log**2).sum(), 'n': dm_tgt_log.size},
                'gas': {'sum': gas_log.sum(), 'sum_sq': (gas_log**2).sum(), 'n': gas_log.size},
                'stellar': {'sum': star_log.sum(), 'sum_sq': (star_log**2).sum(), 'n': star_log.size},
                'stellar_samples': star_log.flatten(),  # ALL pixels for quantile fitting
            }
    except Exception as e:
        return None


def compute_cosmo_norm_stats(
    data_root: str,
    n_samples: int = 10000,
    output_dir: str = 'data/',
    seed: int = 42,
    fit_quantile: bool = True,
    n_workers: int = None,
):
    """
    Compute normalization statistics for cosmologically-normalized fields.
    
    Uses parallel processing for fast loading of training data.
    
    Args:
        data_root: Root directory containing training data
        n_samples: Number of random file samples to use for computing stats
        output_dir: Directory to save normalization stats
        seed: Random seed for reproducibility
        fit_quantile: Whether to fit quantile transformer for stellar channel
        n_workers: Number of parallel workers (default: cpu_count)
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 64)  # Cap at 64 to avoid overwhelming filesystem
    print("=" * 70)
    print("Computing Cosmological Normalization Statistics")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print(f"N file samples: {n_samples}")
    print(f"N workers: {n_workers}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Get file list
    print("Scanning data directory...")
    all_files = []
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if 'halo' in f and f.endswith('.npz'):
                all_files.append(os.path.join(root, f))
    all_files = sorted(all_files)
    print(f"Found {len(all_files)} files")
    
    # Randomly sample files
    np.random.seed(seed)
    if len(all_files) > n_samples:
        indices = np.random.choice(len(all_files), size=n_samples, replace=False)
        sample_files = [all_files[i] for i in indices]
    else:
        sample_files = all_files
    print(f"Using {len(sample_files)} random file samples")
    print()
    
    # Process files in parallel
    print(f"Processing files with {n_workers} parallel workers...")
    
    # Initialize aggregators for online mean/std computation
    aggregated = {
        'dm_cond': {'sum': 0.0, 'sum_sq': 0.0, 'n': 0},
        'large_scale': {'sum': 0.0, 'sum_sq': 0.0, 'n': 0},
        'dm_target': {'sum': 0.0, 'sum_sq': 0.0, 'n': 0},
        'gas': {'sum': 0.0, 'sum_sq': 0.0, 'n': 0},
        'stellar': {'sum': 0.0, 'sum_sq': 0.0, 'n': 0},
    }
    stellar_samples_list = []
    
    # Use multiprocessing pool
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, sample_files, chunksize=100),
            total=len(sample_files),
            desc="Loading files"
        ))
    
    # Aggregate results
    print("\nAggregating statistics...")
    n_successful = 0
    for result in results:
        if result is None:
            continue
        n_successful += 1
        for key in aggregated:
            aggregated[key]['sum'] += result[key]['sum']
            aggregated[key]['sum_sq'] += result[key]['sum_sq']
            aggregated[key]['n'] += result[key]['n']
        stellar_samples_list.append(result['stellar_samples'])
    
    print(f"Successfully processed {n_successful}/{len(sample_files)} files")
    
    # Compute mean and std from aggregated sums
    def compute_mean_std(agg):
        mean = agg['sum'] / agg['n']
        var = (agg['sum_sq'] / agg['n']) - mean**2
        std = np.sqrt(max(var, 0))  # Protect against numerical issues
        return mean, std
    
    # Compute statistics
    print("\nComputing final statistics...")
    dm_input_mean, dm_input_std = compute_mean_std(aggregated['dm_cond'])
    ls_mean, ls_std = compute_mean_std(aggregated['large_scale'])
    dm_target_mean, dm_target_std = compute_mean_std(aggregated['dm_target'])
    gas_mean, gas_std = compute_mean_std(aggregated['gas'])
    star_mean, star_std = compute_mean_std(aggregated['stellar'])
    
    print(f"DM input (condition): mean={dm_input_mean:.6f}, std={dm_input_std:.6f}")
    print(f"Large-scale:          mean={ls_mean:.6f}, std={ls_std:.6f}")
    print(f"DM target:            mean={dm_target_mean:.6f}, std={dm_target_std:.6f}")
    print(f"Gas target:           mean={gas_mean:.6f}, std={gas_std:.6f}")
    print(f"Stellar target:       mean={star_mean:.6f}, std={star_std:.6f}")
    
    # Save statistics
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DM (input) stats
    dm_input_path = os.path.join(output_dir, 'cosmo_norm_dm_input_stats.npz')
    np.savez(
        dm_input_path,
        dm_input_mean=dm_input_mean,
        dm_input_std=dm_input_std,
        n_samples=len(sample_files),
    )
    print(f"\n✓ Saved DM input stats to: {dm_input_path}")
    
    # Save large-scale stats
    ls_path = os.path.join(output_dir, 'cosmo_norm_large_scale_stats.npz')
    np.savez(
        ls_path,
        large_scale_mean=ls_mean,
        large_scale_std=ls_std,
        n_samples=len(sample_files),
    )
    print(f"✓ Saved large-scale stats to: {ls_path}")
    
    # Save DM target stats
    dm_target_path = os.path.join(output_dir, 'cosmo_norm_dark_matter_stats.npz')
    np.savez(
        dm_target_path,
        dm_mag_mean=dm_target_mean,
        dm_mag_std=dm_target_std,
        n_samples=len(sample_files),
    )
    print(f"✓ Saved DM target stats to: {dm_target_path}")
    
    # Save Gas stats
    gas_path = os.path.join(output_dir, 'cosmo_norm_gas_stats.npz')
    np.savez(
        gas_path,
        gas_mag_mean=gas_mean,
        gas_mag_std=gas_std,
        n_samples=len(sample_files),
    )
    print(f"✓ Saved Gas stats to: {gas_path}")
    
    # Save Stellar stats
    star_path = os.path.join(output_dir, 'cosmo_norm_stellar_stats.npz')
    np.savez(
        star_path,
        star_mag_mean=star_mean,
        star_mag_std=star_std,
        n_samples=len(sample_files),
    )
    print(f"✓ Saved Stellar stats to: {star_path}")
    
    # Also save all stats in one file for convenience
    all_stats_path = os.path.join(output_dir, 'cosmo_norm_all_stats.npz')
    np.savez(
        all_stats_path,
        # DM input (condition)
        dm_input_mean=dm_input_mean,
        dm_input_std=dm_input_std,
        # Large-scale
        large_scale_mean=ls_mean,
        large_scale_std=ls_std,
        # DM target
        dm_target_mean=dm_target_mean,
        dm_target_std=dm_target_std,
        # Gas
        gas_mean=gas_mean,
        gas_std=gas_std,
        # Stellar
        star_mean=star_mean,
        star_std=star_std,
        # Metadata
        n_samples=len(sample_files),
        description="Stats for log10(field/cosmo_param + 1) where cosmo_param is Omega_m for DM, Omega_b for baryons",
    )
    print(f"✓ Saved all stats to: {all_stats_path}")
    
    # Fit quantile transformer for stellar channel
    if fit_quantile:
        print(f"\n{'='*70}")
        print("Fitting Quantile Transformer for Stellar Channel (cosmo_norm)")
        print(f"{'='*70}")
        
        try:
            from sklearn.preprocessing import QuantileTransformer
            
            # Use ALL samples collected during stats computation (not subsampled!)
            # This ensures we capture the full distribution including bright central pixels
            stellar_data = np.concatenate(stellar_samples_list)
            print(f"Using ALL {len(stellar_data):,} stellar pixel samples for quantile fit")
            print(f"  ({n_successful} files × {128*128} pixels/file)")
            
            # Add small noise to spread out sparse zeros (matches inference-time noise in Normalize)
            print("Adding small noise to spread out sparse zeros...")
            noise = np.random.randn(len(stellar_data)) * 1e-4
            stellar_data = stellar_data + noise
            
            # Fit quantile transformer
            qt = QuantileTransformer(
                n_quantiles=1000,
                output_distribution='uniform',  # Uniform [0, 1] distribution
                random_state=seed
            )
            qt.fit(stellar_data.reshape(-1, 1))
            
            # Save quantile transformer
            qt_file = os.path.join(output_dir, 'cosmo_norm_quantile_normalizer_stellar.pkl')
            joblib.dump(qt, qt_file)
            print(f"✓ Saved quantile transformer to: {qt_file}")
            
            # Print quantile info
            print(f"  n_quantiles: {len(qt.quantiles_)}")
            print(f"  output_distribution: {qt.output_distribution}")
            print(f"  Input range: [{stellar_data.min():.4f}, {stellar_data.max():.4f}]")
            
        except ImportError:
            print("⚠️  sklearn not installed, skipping quantile transformer")
        except Exception as e:
            print(f"⚠️  Error fitting quantile transformer: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("SUMMARY: Cosmological Normalization Statistics")
    print("=" * 70)
    print(f"{'Field':<20} {'Mean':<15} {'Std':<15}")
    print("-" * 50)
    print(f"{'DM input':<20} {dm_input_mean:<15.6f} {dm_input_std:<15.6f}")
    print(f"{'Large-scale':<20} {ls_mean:<15.6f} {ls_std:<15.6f}")
    print(f"{'DM target':<20} {dm_target_mean:<15.6f} {dm_target_std:<15.6f}")
    print(f"{'Gas':<20} {gas_mean:<15.6f} {gas_std:<15.6f}")
    print(f"{'Stellar':<20} {star_mean:<15.6f} {star_std:<15.6f}")
    print()
    
    return {
        'dm_input': (dm_input_mean, dm_input_std),
        'large_scale': (ls_mean, ls_std),
        'dm_target': (dm_target_mean, dm_target_std),
        'gas': (gas_mean, gas_std),
        'stellar': (star_mean, star_std),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute normalization stats for cosmologically-normalized fields'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/',
        help='Root directory containing training data'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=10000,
        help='Number of random samples to use (default: 10000)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/home/mlee1/vdm_BIND/data/',
        help='Directory to save normalization stats'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--fit_quantile',
        action='store_true',
        default=True,
        help='Fit quantile transformer for stellar channel (default: True)'
    )
    parser.add_argument(
        '--no_quantile',
        action='store_true',
        help='Skip fitting quantile transformer'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect CPU count, max 64)'
    )
    
    args = parser.parse_args()
    
    compute_cosmo_norm_stats(
        data_root=args.data_root,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        fit_quantile=not args.no_quantile,
        n_workers=args.n_workers,
    )
