"""
I/O utilities for loading simulation data.

This module provides centralized functions for loading particle data from
CAMELS simulations (both N-body DMO and hydrodynamic runs) and halo catalogs.

These functions are used by:
- run_bind_unified.py (BIND inference)
- data_generation/process_simulations.py (training data generation)
- bind_predict.py (user CLI)
"""

import os
import numpy as np
import h5py
from typing import Tuple, Optional, List, Union
from pathlib import Path


def load_simulation(nbody_path: Union[str, Path], hydro_snapdir: Union[str, Path], 
                    snapnum: int = 90) -> Tuple[np.ndarray, ...]:
    """
    Load particle data from N-body (DMO) and hydrodynamic simulations.
    
    Parameters
    ----------
    nbody_path : str or Path
        Path to N-body simulation directory
    hydro_snapdir : str or Path
        Path to hydro simulation snapshot directory
    snapnum : int
        Snapshot number (default: 90 for z=0 in CAMELS)
        
    Returns
    -------
    tuple of np.ndarray
        (dm_pos, dm_mass, hydro_dm_pos, hydro_dm_mass, 
         gas_pos, gas_mass, star_pos, star_mass)
        Positions in Mpc/h, masses in M_sun/h
    """
    nbody_path = str(nbody_path)
    hydro_snapdir = str(hydro_snapdir)
    snap_str = f'{snapnum:03d}'
    
    # Load N-body DM particles
    dm_pos = []
    dm_mass = []
    nbody_snap = os.path.join(nbody_path, f'snap_{snap_str}.hdf5')
    
    with h5py.File(nbody_snap, 'r') as f:
        dm_pos.append(f['PartType1/Coordinates'][:])
        mass_table = f['Header'].attrs['MassTable']
        dm_particle_mass = mass_table[1]
        num_dm = len(f['PartType1/Coordinates'][:])
        dm_mass.append(np.full(num_dm, dm_particle_mass))
    
    dm_pos = np.concatenate(dm_pos)
    dm_mass = np.concatenate(dm_mass)
    dm_pos /= 1000.0  # kpc -> Mpc
    dm_mass *= 1e10   # 10^10 M_sun -> M_sun
    
    # Load hydro particles (chunked files)
    hydro_dm_pos = []
    hydro_dm_mass = []
    gas_pos = []
    gas_mass = []
    star_pos = []
    star_mass = []
    
    for i in range(16):  # CAMELS uses up to 16 chunks
        fname = os.path.join(hydro_snapdir, f'snap_{snap_str}.{i}.hdf5')
        if not os.path.exists(fname):
            continue
            
        with h5py.File(fname, 'r') as f:
            # DM particles
            if 'PartType1/Coordinates' in f:
                hydro_dm_pos.append(f['PartType1/Coordinates'][:])
                if 'PartType1/Masses' in f:
                    hydro_dm_mass.append(f['PartType1/Masses'][:])
                else:
                    mass_table = f['Header'].attrs['MassTable']
                    dm_particle_mass = mass_table[1]
                    num_dm = len(f['PartType1/Coordinates'][:])
                    hydro_dm_mass.append(np.full(num_dm, dm_particle_mass))
            
            # Gas particles
            if 'PartType0/Coordinates' in f:
                gas_pos.append(f['PartType0/Coordinates'][:])
                gas_mass.append(f['PartType0/Masses'][:])
            
            # Star particles
            if 'PartType4/Coordinates' in f:
                star_pos.append(f['PartType4/Coordinates'][:])
                star_mass.append(f['PartType4/Masses'][:])
    
    # Concatenate and handle empty arrays
    hydro_dm_pos = np.concatenate(hydro_dm_pos) if hydro_dm_pos else np.array([])
    hydro_dm_mass = np.concatenate(hydro_dm_mass) if hydro_dm_mass else np.array([])
    gas_pos = np.concatenate(gas_pos) if gas_pos else np.array([])
    gas_mass = np.concatenate(gas_mass) if gas_mass else np.array([])
    star_pos = np.concatenate(star_pos) if star_pos else np.array([])
    star_mass = np.concatenate(star_mass) if star_mass else np.array([])
    
    # Convert units (kpc -> Mpc, 10^10 M_sun -> M_sun)
    if len(hydro_dm_pos) > 0:
        hydro_dm_pos /= 1000.0
        hydro_dm_mass *= 1e10
    if len(gas_pos) > 0:
        gas_pos /= 1000.0
        gas_mass *= 1e10
    if len(star_pos) > 0:
        star_pos /= 1000.0
        star_mass *= 1e10
    
    return (dm_pos, dm_mass, hydro_dm_pos, hydro_dm_mass, 
            gas_pos, gas_mass, star_pos, star_mass)


def load_halo_catalog(fof_path: Union[str, Path], 
                      mass_threshold: float = 1e13,
                      snapnum: int = 90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load halo catalog from FOF/Subfind files.
    
    Parameters
    ----------
    fof_path : str or Path
        Path to FOF directory or specific FOF file
    mass_threshold : float
        Minimum halo mass in M_sun/h (default: 1e13)
    snapnum : int
        Snapshot number (default: 90)
        
    Returns
    -------
    tuple of np.ndarray
        (positions, masses, radii) - filtered by mass threshold
        Positions in Mpc/h, masses in M_sun/h, radii in Mpc/h
    """
    fof_path = Path(fof_path)
    
    # Handle different path formats
    if fof_path.is_dir():
        # Try different naming conventions
        snap_str = f'{snapnum:03d}'
        possible_files = [
            fof_path / f'fof_subhalo_tab_{snap_str}.hdf5',
            fof_path / f'groups_{snap_str}.hdf5',
            fof_path / 'fof_subhalo_tab_090.hdf5',
        ]
        fof_file = None
        for f in possible_files:
            if f.exists():
                fof_file = f
                break
        if fof_file is None:
            raise FileNotFoundError(f"No FOF file found in {fof_path}")
    else:
        fof_file = fof_path
    
    halo_pos = np.array([])
    halo_mass = np.array([])
    halo_radii = np.array([])
    
    if fof_file.exists():
        with h5py.File(fof_file, 'r') as f:
            if 'Group/GroupPos' in f:
                halo_pos = f['Group/GroupPos'][:]
            if 'Group/Group_M_Crit200' in f:
                halo_mass = f['Group/Group_M_Crit200'][:]
            if 'Group/Group_R_Crit200' in f:
                halo_radii = f['Group/Group_R_Crit200'][:]
    
    if len(halo_mass) > 0:
        # Convert units
        halo_pos = np.array(halo_pos) / 1000.0  # kpc -> Mpc
        halo_mass = np.array(halo_mass) * 1e10   # 10^10 M_sun -> M_sun
        halo_radii = np.array(halo_radii) / 1000.0 if len(halo_radii) > 0 else np.zeros(len(halo_mass))
        
        # Apply mass threshold
        mask = halo_mass > mass_threshold
        return halo_pos[mask], halo_mass[mask], halo_radii[mask]
    
    return np.array([]), np.array([]), np.array([])


def project_particles_2d(positions: np.ndarray, masses: np.ndarray, 
                         box_size: float, resolution: int, 
                         axis: int = 2) -> np.ndarray:
    """
    Project 3D particle data onto a 2D plane using CIC interpolation.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3) in Mpc/h
    masses : np.ndarray
        Particle masses, shape (N,) in M_sun/h
    box_size : float
        Box size in Mpc/h
    resolution : int
        Output grid resolution (e.g., 1024 for 1024x1024)
    axis : int
        Projection axis (0=x, 1=y, 2=z, default: 2)
        
    Returns
    -------
    np.ndarray
        2D mass field, shape (resolution, resolution)
    """
    try:
        import MAS_library as MASL
    except ImportError:
        raise ImportError("MAS_library (pylians3) required for projection")
    
    # Select projection axes
    axes = [0, 1, 2]
    proj_axes = axes[:axis] + axes[axis+1:]
    
    # Prepare arrays for MASL (must be contiguous float32)
    pos_2d = np.ascontiguousarray(positions.astype(np.float32))[:, proj_axes]
    mass_arr = np.ascontiguousarray(masses.astype(np.float32))
    
    # Create output field and run CIC
    field = np.zeros((resolution, resolution), dtype=np.float32)
    MASL.MA(pos_2d, field, box_size, MAS='CIC', W=mass_arr, verbose=False)
    
    return field


def voxelize_particles_3d(positions: np.ndarray, masses: np.ndarray,
                          box_size: float, resolution: int) -> np.ndarray:
    """
    Voxelize 3D particle data using CIC interpolation.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3) in Mpc/h
    masses : np.ndarray
        Particle masses, shape (N,) in M_sun/h
    box_size : float
        Box size in Mpc/h
    resolution : int
        Output grid resolution (e.g., 512 for 512^3)
        
    Returns
    -------
    np.ndarray
        3D mass field, shape (resolution, resolution, resolution)
    """
    try:
        import MAS_library as MASL
    except ImportError:
        raise ImportError("MAS_library (pylians3) required for voxelization")
    
    # Prepare arrays for MASL (must be contiguous float32)
    pos_arr = np.ascontiguousarray(positions.astype(np.float32))
    mass_arr = np.ascontiguousarray(masses.astype(np.float32))
    
    # Create output field and run CIC
    field = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    MASL.MA(pos_arr, field, box_size, MAS='CIC', W=mass_arr, verbose=False)
    
    return field


def apply_periodic_boundary(positions: np.ndarray, center: np.ndarray, 
                           box_size: float = 50.0) -> np.ndarray:
    """
    Apply minimum image convention for periodic boundaries.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3)
    center : np.ndarray
        Reference center position, shape (3,)
    box_size : float
        Box size in same units as positions
        
    Returns
    -------
    np.ndarray
        Positions relative to center with minimum image applied
    """
    delta = positions - center
    delta = delta - box_size * np.round(delta / box_size)
    return delta
