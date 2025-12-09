#!/usr/bin/env python
"""
BIND Predict: User-friendly CLI for applying trained BIND models to DMO simulations.

This script allows users to generate baryonic fields (Dark Matter from hydro, Gas, Stars)
from any Dark Matter Only (DMO) simulation.

SUPPORTED FORMATS:
    Snapshots: HDF5 (AREPO/Gadget format), single or multi-file
    Halo catalogs: SubFind HDF5, FOF HDF5, Rockstar, CSV

Example usage:
    # Basic usage with required arguments
    python bind_predict.py --dmo_path /path/to/dmo/snap_090.hdf5 \\
                          --halo_catalog /path/to/fof_subhalo_tab_090.hdf5 \\
                          --output_dir ./bind_output

    # With custom parameters
    python bind_predict.py --dmo_path /path/to/dmo/snap_090.hdf5 \\
                          --halo_catalog /path/to/fof_subhalo_tab_090.hdf5 \\
                          --output_dir ./bind_output \\
                          --box_size 50.0 \\
                          --mass_threshold 1e13 \\
                          --n_realizations 5

    # Using Rockstar halo catalog
    python bind_predict.py --dmo_path /path/to/dmo/snap_090.hdf5 \\
                          --halo_catalog /path/to/halos_0.0.ascii \\
                          --halo_format rockstar \\
                          --output_dir ./bind_output

    # Using CSV halo catalog
    python bind_predict.py --dmo_path /path/to/dmo/snap_090.hdf5 \\
                          --halo_catalog /path/to/halos.csv \\
                          --halo_format csv \\
                          --output_dir ./bind_output

Author: VDM-BIND Team
Repository: https://github.com/your-username/vdm_BIND
"""
import argparse
import numpy as np
import torch
import h5py
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import MAS_library as MASL


def load_dmo_snapshot(snapshot_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load DMO snapshot from HDF5 file.
    
    Supports both single-file and multi-file snapshots.
    
    Args:
        snapshot_path: Path to snapshot file (e.g., snap_090.hdf5 or snapdir_090/)
    
    Returns:
        positions: Particle positions in Mpc/h (N, 3)
        masses: Particle masses in M_sun/h (N,)
        box_size: Box size in Mpc/h
    """
    snapshot_path = Path(snapshot_path)
    
    positions = []
    masses = []
    box_size = None
    
    # Check if it's a directory (multi-file) or single file
    if snapshot_path.is_dir():
        # Multi-file snapshot
        files = sorted(snapshot_path.glob("snap_*.hdf5"))
        if not files:
            files = sorted(snapshot_path.glob("snapshot_*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No snapshot files found in {snapshot_path}")
    else:
        # Single file
        files = [snapshot_path]
    
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            # Get box size from header
            if box_size is None:
                header = dict(f['Header'].attrs.items())
                box_size = header.get('BoxSize', None)
                if box_size is None:
                    raise ValueError("Could not determine box size from snapshot")
                # Convert from code units (kpc/h) to Mpc/h if needed
                if box_size > 1000:  # Assume kpc/h
                    box_size /= 1000.0
            
            # Load DM particles (PartType1)
            if 'PartType1/Coordinates' in f:
                pos = f['PartType1/Coordinates'][:]
                positions.append(pos)
                
                # Get masses
                if 'PartType1/Masses' in f:
                    mass = f['PartType1/Masses'][:]
                else:
                    # Use mass table
                    mass_table = header.get('MassTable', f['Header'].attrs.get('MassTable'))
                    particle_mass = mass_table[1]
                    mass = np.full(len(pos), particle_mass)
                masses.append(mass)
    
    # Concatenate all particles
    positions = np.concatenate(positions)
    masses = np.concatenate(masses)
    
    # Convert units
    # Positions: kpc/h -> Mpc/h
    if positions.max() > 1000:
        positions /= 1000.0
    
    # Masses: 10^10 M_sun/h -> M_sun/h
    if masses.max() < 1e6:  # Likely in 10^10 M_sun/h units
        masses *= 1e10
    
    print(f"Loaded {len(positions)} DM particles")
    print(f"Box size: {box_size} Mpc/h")
    print(f"Mass range: {masses.min():.2e} - {masses.max():.2e} M_sun/h")
    
    return positions.astype(np.float32), masses.astype(np.float32), float(box_size)


def load_halo_catalog(catalog_path: str, mass_threshold: float = 1e13,
                      halo_format: str = 'auto') -> Dict:
    """
    Load halo catalog from various formats.
    
    Supported formats:
        - SubFind/FOF HDF5 (AREPO/Gadget)
        - Rockstar ASCII
        - CSV with columns: x, y, z, mass, radius (optional)
    
    Args:
        catalog_path: Path to halo catalog file
        mass_threshold: Minimum halo mass in M_sun/h
        halo_format: 'auto', 'subfind', 'rockstar', or 'csv'
    
    Returns:
        Dictionary with halo positions, masses, radii
    """
    catalog_path = Path(catalog_path)
    
    # Auto-detect format
    if halo_format == 'auto':
        suffix = catalog_path.suffix.lower()
        if suffix in ['.hdf5', '.h5']:
            halo_format = 'subfind'
        elif suffix in ['.ascii', '.list', '.txt']:
            halo_format = 'rockstar'
        elif suffix == '.csv':
            halo_format = 'csv'
        else:
            print(f"Warning: Unknown format for {catalog_path}, trying SubFind HDF5")
            halo_format = 'subfind'
    
    if halo_format == 'subfind':
        return _load_subfind_catalog(catalog_path, mass_threshold)
    elif halo_format == 'rockstar':
        return _load_rockstar_catalog(catalog_path, mass_threshold)
    elif halo_format == 'csv':
        return _load_csv_catalog(catalog_path, mass_threshold)
    else:
        raise ValueError(f"Unknown halo format: {halo_format}")


def _load_subfind_catalog(catalog_path: Path, mass_threshold: float) -> Dict:
    """Load SubFind/FOF HDF5 halo catalog."""
    with h5py.File(catalog_path, 'r') as f:
        # Try different common group names
        if 'Group/GroupPos' in f:
            positions = f['Group/GroupPos'][:]
            masses = f['Group/Group_M_Crit200'][:]
            radii = f['Group/Group_R_Crit200'][:]
        elif 'Subhalo/SubhaloPos' in f:
            # Subfind format
            positions = f['Subhalo/SubhaloPos'][:]
            masses = f['Subhalo/SubhaloMass'][:]
            radii = np.zeros(len(masses))  # May not have radii
        else:
            raise ValueError(f"Could not find halo data in {catalog_path}")
    
    # Convert units
    # Positions: kpc/h -> Mpc/h
    if positions.max() > 1000:
        positions /= 1000.0
    
    # Masses: 10^10 M_sun/h -> M_sun/h  
    if masses.max() < 1e6:
        masses *= 1e10
    
    # Radii: kpc/h -> Mpc/h
    if radii.max() > 1000:
        radii /= 1000.0
    
    # Apply mass threshold
    mask = masses >= mass_threshold
    
    catalog = {
        'positions': positions[mask],
        'masses': masses[mask],
        'radii': radii[mask],
        'indices': np.where(mask)[0]
    }
    
    print(f"Loaded {mask.sum()} halos above {mass_threshold:.1e} M_sun/h (SubFind format)")
    return catalog


def _load_rockstar_catalog(catalog_path: Path, mass_threshold: float) -> Dict:
    """Load Rockstar ASCII halo catalog."""
    import pandas as pd
    
    # Read Rockstar format (space-separated with # comments)
    # Standard columns: id, descid, mvir, vmax, vrms, rvir, rs, np, x, y, z, ...
    
    # Try to read the file and detect columns
    with open(catalog_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if 'mvir' in line.lower():
                    # This is a header line with column names
                    columns = line.strip('#').split()
                    break
        else:
            # Default Rockstar columns
            columns = ['id', 'descid', 'mvir', 'vmax', 'vrms', 'rvir', 'rs', 
                      'np', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Jx', 'Jy', 'Jz', 
                      'spin', 'rs_klypin', 'mvir_all', 'm200b', 'm200c', 'm500c',
                      'm2500c', 'Xoff', 'Voff', 'spin_bullock', 'b_to_a', 'c_to_a']
    
    # Read data, skipping comments
    data = pd.read_csv(catalog_path, comment='#', sep=r'\s+', 
                       names=columns[:], header=None, on_bad_lines='skip')
    
    # Extract position and mass columns
    if 'x' in data.columns and 'mvir' in data.columns:
        positions = data[['x', 'y', 'z']].values
        masses = data['mvir'].values
        radii = data['rvir'].values if 'rvir' in data.columns else np.zeros(len(masses))
    else:
        raise ValueError(f"Could not find x, y, z, mvir columns in Rockstar file")
    
    # Rockstar outputs in Mpc/h and M_sun/h by default (check header)
    # Apply mass threshold
    mask = masses >= mass_threshold
    
    catalog = {
        'positions': positions[mask].astype(np.float32),
        'masses': masses[mask].astype(np.float32),
        'radii': radii[mask].astype(np.float32),
        'indices': np.where(mask)[0]
    }
    
    print(f"Loaded {mask.sum()} halos above {mass_threshold:.1e} M_sun/h (Rockstar format)")
    return catalog


def _load_csv_catalog(catalog_path: Path, mass_threshold: float) -> Dict:
    """
    Load CSV halo catalog.
    
    Expected columns: x, y, z, mass (and optionally: radius)
    Units: Mpc/h for positions, M_sun/h for mass
    """
    import pandas as pd
    
    data = pd.read_csv(catalog_path)
    
    # Normalize column names to lowercase
    data.columns = data.columns.str.lower().str.strip()
    
    # Find position columns
    pos_cols = None
    for variants in [['x', 'y', 'z'], ['pos_x', 'pos_y', 'pos_z'], 
                     ['position_x', 'position_y', 'position_z']]:
        if all(c in data.columns for c in variants):
            pos_cols = variants
            break
    
    if pos_cols is None:
        raise ValueError(f"CSV must have position columns (x, y, z). Found: {list(data.columns)}")
    
    positions = data[pos_cols].values
    
    # Find mass column
    mass_col = None
    for variant in ['mass', 'm200', 'mvir', 'halo_mass', 'm_halo']:
        if variant in data.columns:
            mass_col = variant
            break
    
    if mass_col is None:
        raise ValueError(f"CSV must have mass column. Found: {list(data.columns)}")
    
    masses = data[mass_col].values
    
    # Find radius column (optional)
    radius_col = None
    for variant in ['radius', 'r200', 'rvir', 'halo_radius', 'r_halo']:
        if variant in data.columns:
            radius_col = variant
            break
    
    radii = data[radius_col].values if radius_col else np.zeros(len(masses))
    
    # Apply mass threshold
    mask = masses >= mass_threshold
    
    catalog = {
        'positions': positions[mask].astype(np.float32),
        'masses': masses[mask].astype(np.float32),
        'radii': radii[mask].astype(np.float32),
        'indices': np.where(mask)[0]
    }
    
    print(f"Loaded {mask.sum()} halos above {mass_threshold:.1e} M_sun/h (CSV format)")
    return catalog


def voxelize_particles(positions: np.ndarray, masses: np.ndarray, 
                       box_size: float, grid_size: int = 1024) -> np.ndarray:
    """
    Voxelize particles onto a 3D grid using CIC interpolation.
    
    Args:
        positions: Particle positions (N, 3) in Mpc/h
        masses: Particle masses (N,) in M_sun/h
        box_size: Box size in Mpc/h
        grid_size: Grid resolution
    
    Returns:
        3D mass field (grid_size, grid_size, grid_size)
    """
    pos = np.ascontiguousarray(positions.astype(np.float32))
    mass = np.ascontiguousarray(masses.astype(np.float32))
    
    field = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    MASL.MA(pos, field, float(box_size), MAS='CIC', W=mass, verbose=False)
    
    return field


def extract_halo_cutout(field: np.ndarray, center: np.ndarray, 
                        scale_mpc: float, box_size: float, 
                        target_res: int = 128) -> np.ndarray:
    """
    Extract cutout around halo center with periodic boundaries.
    
    Args:
        field: 3D or 2D density field
        center: Halo center position in Mpc/h
        scale_mpc: Physical scale to extract in Mpc/h
        box_size: Box size in Mpc/h
        target_res: Target resolution for output
    
    Returns:
        Cutout array at target resolution
    """
    grid_size = field.shape[0]
    pix_size = box_size / grid_size
    half_size_pix = int(scale_mpc / (2 * pix_size))
    
    # Convert center to pixel coordinates
    center_pix = (center / pix_size).astype(int) % grid_size
    
    ndim = len(field.shape)
    
    if ndim == 3:
        # 3D extraction with periodic boundaries
        ix = (np.arange(-half_size_pix, half_size_pix) + center_pix[0]) % grid_size
        iy = (np.arange(-half_size_pix, half_size_pix) + center_pix[1]) % grid_size
        iz = (np.arange(-half_size_pix, half_size_pix) + center_pix[2]) % grid_size
        
        cutout = field[np.ix_(ix, iy, iz)]
    else:
        # 2D extraction
        ix = (np.arange(-half_size_pix, half_size_pix) + center_pix[0]) % grid_size
        iy = (np.arange(-half_size_pix, half_size_pix) + center_pix[1]) % grid_size
        
        cutout = field[np.ix_(ix, iy)]
    
    # Downsample if needed
    current_res = cutout.shape[0]
    if current_res > target_res:
        factor = current_res // target_res
        if ndim == 3:
            cutout = cutout.reshape(target_res, factor, target_res, factor, target_res, factor).mean(axis=(1, 3, 5))
        else:
            cutout = cutout.reshape(target_res, factor, target_res, factor).mean(axis=(1, 3))
    
    return cutout


def normalize_field(field: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply log transform and Z-score normalization."""
    log_field = np.log10(field + 1)
    return (log_field - mean) / std


def denormalize_field(normalized: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Reverse normalization to get physical units."""
    log_field = normalized * std + mean
    return 10**log_field - 1


def main():
    parser = argparse.ArgumentParser(
        description='BIND: Generate baryonic fields from DMO simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python bind_predict.py --dmo_path /path/to/snap_090.hdf5 \\
                          --halo_catalog /path/to/fof_subhalo_tab_090.hdf5 \\
                          --output_dir ./output

    # Custom mass threshold and realizations  
    python bind_predict.py --dmo_path /path/to/snap_090.hdf5 \\
                          --halo_catalog /path/to/fof_subhalo_tab_090.hdf5 \\
                          --output_dir ./output \\
                          --mass_threshold 5e12 \\
                          --n_realizations 10
                          
    # Using Rockstar catalog
    python bind_predict.py --dmo_path /path/to/snap_090.hdf5 \\
                          --halo_catalog /path/to/halos_0.0.ascii \\
                          --halo_format rockstar \\
                          --output_dir ./output
                          
    # Using CSV catalog (columns: x, y, z, mass)
    python bind_predict.py --dmo_path /path/to/snap_090.hdf5 \\
                          --halo_catalog /path/to/halos.csv \\
                          --halo_format csv \\
                          --output_dir ./output
"""
    )
    
    # Required arguments
    parser.add_argument('--dmo_path', type=str, required=True,
                       help='Path to DMO snapshot file or directory')
    parser.add_argument('--halo_catalog', type=str, required=True,
                       help='Path to halo catalog (HDF5, Rockstar ASCII, or CSV)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for generated fields')
    
    # Model configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model config file (uses default if not specified)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (uses best from config if not specified)')
    
    # Simulation parameters
    parser.add_argument('--box_size', type=float, default=None,
                       help='Box size in Mpc/h (auto-detected from snapshot if not specified)')
    parser.add_argument('--mass_threshold', type=float, default=1e13,
                       help='Minimum halo mass in M_sun/h (default: 1e13)')
    parser.add_argument('--grid_size', type=int, default=1024,
                       help='Grid resolution for voxelization (default: 1024)')
    
    # Halo catalog options
    parser.add_argument('--halo_format', type=str, default='auto',
                       choices=['auto', 'subfind', 'rockstar', 'csv'],
                       help='Halo catalog format (default: auto-detect)')
    
    # Generation parameters
    parser.add_argument('--n_realizations', type=int, default=1,
                       help='Number of stochastic realizations per halo (default: 1)')
    parser.add_argument('--n_sampling_steps', type=int, default=250,
                       help='Number of DDIM sampling steps (default: 250)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for generation (default: 8)')
    
    # Output options
    parser.add_argument('--save_full_box', action='store_true',
                       help='Also save full-box pasted outputs')
    parser.add_argument('--conserve_mass', action='store_true', default=True,
                       help='Normalize output to conserve total mass (default: True)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("BIND: Baryonic Inference from N-body Data")
    print("="*60)
    
    # Step 1: Load DMO snapshot
    print("\n[1/5] Loading DMO snapshot...")
    positions, masses, detected_box_size = load_dmo_snapshot(args.dmo_path)
    box_size = args.box_size if args.box_size is not None else detected_box_size
    print(f"Using box size: {box_size} Mpc/h")
    
    # Step 2: Load halo catalog
    print("\n[2/5] Loading halo catalog...")
    halo_catalog = load_halo_catalog(args.halo_catalog, args.mass_threshold, args.halo_format)
    
    if len(halo_catalog['positions']) == 0:
        print("ERROR: No halos found above mass threshold!")
        return
    
    # Step 3: Voxelize
    print("\n[3/5] Voxelizing DMO field...")
    dm_field = voxelize_particles(positions, masses, box_size, args.grid_size)
    print(f"Voxelized to {args.grid_size}^3 grid")
    
    # Step 4: Load model
    print("\n[4/5] Loading diffusion model...")
    
    # Find default config if not specified
    if args.config is None:
        from config import PROJECT_ROOT
        default_config = PROJECT_ROOT / 'configs' / 'clean_vdm_aggressive_stellar.ini'
        if default_config.exists():
            args.config = str(default_config)
            print(f"Using default config: {args.config}")
        else:
            raise FileNotFoundError("No config specified and default config not found")
    
    # Import BIND components
    from bind.workflow_utils import ConfigLoader, ModelManager, load_normalization_stats, sample
    
    config = ConfigLoader(args.config, verbose=args.verbose)
    model, _ = ModelManager.initialize(config, verbose=args.verbose, skip_data_loading=True)
    model = model.to(args.device)
    model.eval()
    
    # Load normalization stats
    norm_stats = load_normalization_stats()
    
    # Step 5: Generate for each halo
    print(f"\n[5/5] Generating baryonic fields for {len(halo_catalog['positions'])} halos...")
    
    # Multi-scale extraction parameters
    scales_mpc = [6.25, 12.5, 25.0, 50.0]
    target_res = 128
    
    # Project to 2D for efficiency (or use 3D if needed)
    print("Projecting to 2D...")
    dm_field_2d = dm_field.sum(axis=2)  # Project along z
    
    results = []
    
    for halo_idx, (halo_pos, halo_mass) in enumerate(zip(
        halo_catalog['positions'], halo_catalog['masses']
    )):
        print(f"\rProcessing halo {halo_idx+1}/{len(halo_catalog['positions'])} "
              f"(M = {halo_mass:.2e} M_sun/h)", end='')
        
        # Extract multi-scale cutouts
        condition_maps = []
        for scale in scales_mpc:
            cutout = extract_halo_cutout(dm_field_2d, halo_pos[:2], scale, box_size, target_res)
            # Normalize
            normalized = normalize_field(cutout, norm_stats['dm_mag_mean'], norm_stats['dm_mag_std'])
            condition_maps.append(normalized)
        
        # Prepare condition tensor: [condition (6.25 Mpc), large_scale (12.5, 25, 50 Mpc)]
        condition = torch.from_numpy(condition_maps[0]).unsqueeze(0).unsqueeze(0).float()
        large_scale = torch.from_numpy(np.stack(condition_maps[1:], axis=0)).unsqueeze(0).float()
        
        # Concatenate conditioning
        full_condition = torch.cat([condition, large_scale], dim=1).to(args.device)
        
        # Generate realizations
        for real_idx in range(args.n_realizations):
            with torch.no_grad():
                generated = sample(
                    model=model.model,
                    conditioning=full_condition,
                    n_sampling_steps=args.n_sampling_steps,
                    batch_size=1,
                    device=args.device,
                )
            
            # Denormalize outputs
            generated_np = generated.cpu().numpy()[0]  # (3, H, W)
            
            dm_hydro = denormalize_field(generated_np[0], norm_stats['dm_mag_mean'], norm_stats['dm_mag_std'])
            gas = denormalize_field(generated_np[1], norm_stats['gas_mag_mean'], norm_stats['gas_mag_std'])
            stars = denormalize_field(generated_np[2], norm_stats['star_mag_mean'], norm_stats['star_mag_std'])
            
            # Save individual halo result
            output_file = output_dir / f"halo_{halo_idx}_real_{real_idx}.npz"
            np.savez_compressed(
                output_file,
                dm_hydro=dm_hydro,
                gas=gas,
                stars=stars,
                condition=condition_maps[0],
                halo_position=halo_pos,
                halo_mass=halo_mass,
                box_size=box_size,
            )
            
            results.append({
                'halo_idx': halo_idx,
                'realization': real_idx,
                'file': str(output_file),
                'halo_mass': halo_mass,
            })
    
    print("\n")  # Newline after progress
    
    # Save summary
    import pandas as pd
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / 'generation_summary.csv', index=False)
    
    print("="*60)
    print(f"Generation complete!")
    print(f"Generated {len(results)} samples")
    print(f"Output saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
