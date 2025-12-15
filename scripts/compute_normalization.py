#!/usr/bin/env python
"""
Compute Normalization Statistics for Training Data.

This script computes the mean and standard deviation for each field type
(dark matter, gas, stellar) from your training data. These statistics are
required for consistent normalization during training and inference.

Usage:
    # Compute stats from training data directory
    python scripts/compute_normalization.py \
        --data_dir /path/to/training_data \
        --output data/my_norm_stats

    # Also fit quantile transformer for stellar channel
    python scripts/compute_normalization.py \
        --data_dir /path/to/training_data \
        --output data/my_norm_stats \
        --fit_quantile

    # Process only a subset of files (for testing)
    python scripts/compute_normalization.py \
        --data_dir /path/to/training_data \
        --output data/my_norm_stats \
        --max_files 100

Output files:
    - {output}_dark_matter_normalization_stats.npz
    - {output}_gas_normalization_stats.npz
    - {output}_stellar_normalization_stats.npz
    - {output}_quantile_normalizer_stellar.pkl (if --fit_quantile)

Author: VDM-BIND Team
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
import joblib
from typing import Dict, Tuple, Optional, List
import warnings


def welford_update(
    existing_aggregate: Tuple[int, float, float],
    new_value: np.ndarray
) -> Tuple[int, float, float]:
    """
    Welford's online algorithm for computing mean and variance.
    
    More numerically stable than naive (sum/n) approach for large datasets.
    
    Args:
        existing_aggregate: Tuple of (count, mean, M2) where M2 is sum of squared differences
        new_value: New array of values to incorporate
    
    Returns:
        Updated (count, mean, M2) tuple
    """
    count, mean, M2 = existing_aggregate
    
    # Flatten and handle the entire array
    flat_values = new_value.flatten()
    
    for x in flat_values:
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2
    
    return (count, mean, M2)


def finalize_welford(existing_aggregate: Tuple[int, float, float]) -> Tuple[float, float]:
    """Compute final mean and std from Welford aggregate."""
    count, mean, M2 = existing_aggregate
    if count < 2:
        return mean, 0.0
    variance = M2 / count
    return mean, np.sqrt(variance)


def compute_stats_streaming(
    file_list: List[str],
    field_key: str,
    transform_fn=None,
    desc: str = "Computing stats"
) -> Tuple[float, float]:
    """
    Compute mean and std using streaming (memory-efficient) approach.
    
    Args:
        file_list: List of .npz file paths
        field_key: Key to extract from each file
        transform_fn: Optional transform to apply (e.g., log10(x+1))
        desc: Description for progress bar
    
    Returns:
        (mean, std) tuple
    """
    # Initialize Welford aggregate: (count, mean, M2)
    aggregate = (0, 0.0, 0.0)
    
    for fpath in tqdm(file_list, desc=desc):
        try:
            data = np.load(fpath)
            if field_key not in data:
                continue
            
            field = data[field_key].astype(np.float64)
            
            if transform_fn is not None:
                field = transform_fn(field)
            
            # Update running statistics
            aggregate = welford_update(aggregate, field)
            
        except Exception as e:
            warnings.warn(f"Error processing {fpath}: {e}")
            continue
    
    return finalize_welford(aggregate)


def compute_normalization_stats(
    data_dir: str,
    output_prefix: str,
    max_files: Optional[int] = None,
    fit_quantile: bool = False,
    quantile_subsample: int = 100000,
) -> Dict[str, Dict[str, float]]:
    """
    Compute normalization statistics for all field types.
    
    Args:
        data_dir: Directory containing .npz training files
        output_prefix: Prefix for output files
        max_files: Maximum number of files to process (None = all)
        fit_quantile: Whether to fit quantile transformer for stellar channel
        quantile_subsample: Number of samples for quantile fitting
    
    Returns:
        Dictionary with statistics for each field
    """
    data_dir = Path(data_dir)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all training files
    file_patterns = ['*.npz']
    file_list = []
    for pattern in file_patterns:
        file_list.extend(glob.glob(str(data_dir / '**' / pattern), recursive=True))
    
    if not file_list:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    if max_files:
        file_list = file_list[:max_files]
    
    print(f"Found {len(file_list)} files to process")
    
    # Define field mappings (key in file -> output name)
    # Support multiple naming conventions
    field_mappings = {
        'dark_matter': {
            'keys': ['dm', 'dm_hydro', 'dark_matter', 'condition'],
            'output_key': 'dm_mag',
        },
        'gas': {
            'keys': ['gas', 'gas_density'],
            'output_key': 'gas_mag',
        },
        'stellar': {
            'keys': ['star', 'stars', 'stellar', 'stellar_density'],
            'output_key': 'star_mag',
        },
    }
    
    # Log transform function
    def log_transform(x):
        return np.log10(np.clip(x, 1e-10, None) + 1)
    
    results = {}
    
    # Process each field type
    for field_name, config in field_mappings.items():
        print(f"\n{'='*60}")
        print(f"Processing {field_name}...")
        print(f"{'='*60}")
        
        # Try each possible key
        found_key = None
        for key in config['keys']:
            # Check first file for this key
            try:
                test_data = np.load(file_list[0])
                if key in test_data:
                    found_key = key
                    break
            except:
                continue
        
        if found_key is None:
            print(f"  ⚠️  No data found for {field_name} (tried keys: {config['keys']})")
            continue
        
        print(f"  Using key: '{found_key}'")
        
        # Compute statistics with log transform
        mean, std = compute_stats_streaming(
            file_list,
            found_key,
            transform_fn=log_transform,
            desc=f"  {field_name}"
        )
        
        results[field_name] = {
            f"{config['output_key']}_mean": mean,
            f"{config['output_key']}_std": std,
        }
        
        print(f"  Mean: {mean:.6f}")
        print(f"  Std:  {std:.6f}")
        
        # Save individual stats file
        output_file = f"{output_prefix}_{field_name}_normalization_stats.npz"
        np.savez(
            output_file,
            **{f"{config['output_key']}_mean": mean, f"{config['output_key']}_std": std}
        )
        print(f"  Saved: {output_file}")
    
    # Fit quantile transformer for stellar if requested
    if fit_quantile and 'stellar' in results:
        print(f"\n{'='*60}")
        print("Fitting quantile transformer for stellar channel...")
        print(f"{'='*60}")
        
        try:
            from sklearn.preprocessing import QuantileTransformer
            
            # Collect subsample of stellar data
            stellar_samples = []
            samples_per_file = max(1, quantile_subsample // len(file_list))
            
            for fpath in tqdm(file_list, desc="  Collecting samples"):
                try:
                    data = np.load(fpath)
                    for key in field_mappings['stellar']['keys']:
                        if key in data:
                            field = data[key].flatten()
                            # Log transform
                            field = log_transform(field)
                            # Random subsample
                            if len(field) > samples_per_file:
                                idx = np.random.choice(len(field), samples_per_file, replace=False)
                                field = field[idx]
                            stellar_samples.append(field)
                            break
                except:
                    continue
                
                if len(stellar_samples) * samples_per_file >= quantile_subsample:
                    break
            
            stellar_data = np.concatenate(stellar_samples)[:quantile_subsample]
            print(f"  Collected {len(stellar_data)} samples")
            
            # Fit quantile transformer
            qt = QuantileTransformer(
                n_quantiles=1000,
                output_distribution='normal',
                random_state=42
            )
            qt.fit(stellar_data.reshape(-1, 1))
            
            # Save
            qt_file = f"{output_prefix}_quantile_normalizer_stellar.pkl"
            joblib.dump(qt, qt_file)
            print(f"  Saved: {qt_file}")
            
        except ImportError:
            print("  ⚠️  sklearn not installed, skipping quantile transformer")
        except Exception as e:
            print(f"  ⚠️  Error fitting quantile transformer: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for field_name, stats in results.items():
        print(f"\n{field_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")
    
    return results


def validate_stats(stats_dir: str) -> bool:
    """
    Validate that normalization stats are reasonable.
    
    Args:
        stats_dir: Directory containing normalization files
    
    Returns:
        True if stats are valid, False otherwise
    """
    stats_dir = Path(stats_dir)
    
    print("\nValidating normalization statistics...")
    
    valid = True
    
    for field in ['dark_matter', 'gas', 'stellar']:
        stat_file = stats_dir / f"{field}_normalization_stats.npz"
        
        if not stat_file.exists():
            # Try without prefix
            stat_file = list(stats_dir.glob(f"*{field}_normalization_stats.npz"))
            if stat_file:
                stat_file = stat_file[0]
            else:
                print(f"  ⚠️  {field}: file not found")
                continue
        
        data = np.load(stat_file)
        
        # Get mean and std (handle different key naming)
        mean_key = [k for k in data.keys() if 'mean' in k][0]
        std_key = [k for k in data.keys() if 'std' in k][0]
        
        mean = float(data[mean_key])
        std = float(data[std_key])
        
        # Validation checks
        issues = []
        
        if not np.isfinite(mean):
            issues.append("mean is not finite")
            valid = False
        
        if not np.isfinite(std) or std <= 0:
            issues.append("std is not positive finite")
            valid = False
        
        # Log-transformed cosmological fields typically have mean in [0, 5] and std in [0.5, 3]
        if mean < -1 or mean > 10:
            issues.append(f"mean={mean:.2f} seems unusual for log-transformed data")
        
        if std < 0.1 or std > 5:
            issues.append(f"std={std:.2f} seems unusual")
        
        if issues:
            print(f"  ⚠️  {field}: {', '.join(issues)}")
        else:
            print(f"  ✓  {field}: mean={mean:.4f}, std={std:.4f}")
    
    return valid


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for VDM-BIND training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        required=True,
        help='Directory containing .npz training files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/normalization',
        help='Output prefix for normalization files (default: data/normalization)'
    )
    
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files to process (for testing)'
    )
    
    parser.add_argument(
        '--fit_quantile',
        action='store_true',
        help='Fit quantile transformer for stellar channel'
    )
    
    parser.add_argument(
        '--quantile_subsample',
        type=int,
        default=100000,
        help='Number of samples for quantile fitting (default: 100000)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing normalization files instead of computing new ones'
    )
    
    args = parser.parse_args()
    
    if args.validate:
        validate_stats(args.data_dir)
    else:
        compute_normalization_stats(
            args.data_dir,
            args.output,
            max_files=args.max_files,
            fit_quantile=args.fit_quantile,
            quantile_subsample=args.quantile_subsample,
        )
        
        # Validate the computed stats
        output_dir = Path(args.output).parent
        validate_stats(output_dir)


if __name__ == '__main__':
    main()
