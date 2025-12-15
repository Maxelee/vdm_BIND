"""
Abstract interfaces for simulation data loading.

This module defines abstract base classes that allow VDM-BIND to work with
arbitrary simulation data formats. Users can implement these interfaces for
their specific simulation codes.

Supported out-of-box:
- CAMELS (IllustrisTNG, SIMBA)
- Generic HDF5 (Gadget/AREPO format)

Example usage:
    from vdm.data_interface import CAMELSLoader, SubFindCatalog
    
    # Load simulation
    loader = CAMELSLoader('/path/to/sim', snapnum=90)
    dm_pos, dm_mass = loader.load_particles('dm')
    
    # Load halos
    halos = SubFindCatalog('/path/to/fof_subhalo_tab_090.hdf5')
    positions, masses, radii = halos.load_halos(mass_threshold=1e13)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List, Any
from pathlib import Path
import numpy as np
import warnings


class SimulationLoader(ABC):
    """
    Abstract base class for loading simulation particle data.
    
    Implement this interface to support new simulation formats.
    
    Args:
        path: Path to simulation snapshot file or directory
        snapnum: Snapshot number (optional, depends on format)
        **kwargs: Additional format-specific arguments
    
    Example implementation:
        class MySimLoader(SimulationLoader):
            def load_particles(self, particle_type):
                # Load from your format
                return positions, masses
            
            def get_box_size(self):
                return self._box_size
    """
    
    def __init__(self, path: str, snapnum: Optional[int] = None, **kwargs):
        self.path = Path(path)
        self.snapnum = snapnum
        self._header = None
        self._box_size = None
    
    @abstractmethod
    def load_particles(
        self, 
        particle_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load particle positions and masses.
        
        Args:
            particle_type: Type of particles to load.
                Common values: 'dm', 'gas', 'stars', 'bh'
        
        Returns:
            positions: Particle positions in Mpc/h, shape (N, 3)
            masses: Particle masses in M_sun/h, shape (N,)
        """
        pass
    
    @abstractmethod
    def get_box_size(self) -> float:
        """
        Get simulation box size.
        
        Returns:
            Box size in Mpc/h
        """
        pass
    
    def get_cosmology(self) -> Optional[Dict[str, float]]:
        """
        Get cosmological parameters (optional).
        
        Returns:
            Dictionary with keys like 'Omega_m', 'Omega_b', 'h', 'sigma_8', etc.
            Returns None if not available.
        """
        return None
    
    def get_redshift(self) -> Optional[float]:
        """
        Get snapshot redshift.
        
        Returns:
            Redshift value or None if not available.
        """
        return None


class HaloCatalogLoader(ABC):
    """
    Abstract base class for loading halo catalogs.
    
    Implement this interface to support new halo finder formats.
    
    Args:
        path: Path to halo catalog file
        **kwargs: Additional format-specific arguments
    """
    
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
    
    @abstractmethod
    def load_halos(
        self, 
        mass_threshold: float = 1e13
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load halo positions, masses, and radii.
        
        Args:
            mass_threshold: Minimum halo mass in M_sun/h
        
        Returns:
            positions: Halo center positions in Mpc/h, shape (N, 3)
            masses: Halo masses in M_sun/h, shape (N,)
            radii: Halo virial radii in Mpc/h, shape (N,)
        """
        pass
    
    def get_halo_properties(
        self, 
        properties: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Get additional halo properties (optional).
        
        Args:
            properties: List of property names to retrieve
        
        Returns:
            Dictionary mapping property names to arrays
        """
        return {}


# =============================================================================
# Concrete Implementations
# =============================================================================

class GadgetHDF5Loader(SimulationLoader):
    """
    Loader for Gadget/AREPO HDF5 snapshots.
    
    Supports both single-file and multi-file snapshots.
    
    Args:
        path: Path to snapshot file or snapdir directory
        snapnum: Snapshot number (for snapdir format)
    """
    
    PARTICLE_TYPES = {
        'gas': 0,
        'dm': 1,
        'disk': 2,  # Rarely used
        'bulge': 3,  # Rarely used
        'stars': 4,
        'bh': 5,
    }
    
    def __init__(self, path: str, snapnum: Optional[int] = None, **kwargs):
        super().__init__(path, snapnum, **kwargs)
        self._load_header()
    
    def _get_snapshot_files(self) -> List[Path]:
        """Get list of snapshot files."""
        import h5py
        
        if self.path.is_file():
            return [self.path]
        
        # Multi-file snapshot in directory
        if self.path.is_dir():
            # Try snapdir format
            if self.snapnum is not None:
                snap_dir = self.path / f'snapdir_{self.snapnum:03d}'
                if snap_dir.exists():
                    files = sorted(snap_dir.glob(f'snap_{self.snapnum:03d}.*.hdf5'))
                    if files:
                        return files
            
            # Try direct glob
            files = sorted(self.path.glob('snap_*.hdf5'))
            if not files:
                files = sorted(self.path.glob('snapshot_*.hdf5'))
            return files
        
        raise FileNotFoundError(f"No snapshot files found at {self.path}")
    
    def _load_header(self):
        """Load header from first snapshot file."""
        import h5py
        
        files = self._get_snapshot_files()
        if not files:
            raise FileNotFoundError(f"No snapshot files found at {self.path}")
        
        with h5py.File(files[0], 'r') as f:
            self._header = dict(f['Header'].attrs.items())
            
            # Get box size
            box_size = self._header.get('BoxSize', None)
            if box_size is None:
                raise ValueError("Could not determine BoxSize from header")
            
            # Convert from code units if needed (kpc/h -> Mpc/h)
            if box_size > 1000:
                box_size = box_size / 1000.0
            
            self._box_size = float(box_size)
    
    def load_particles(
        self, 
        particle_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load particles from Gadget/AREPO snapshot."""
        import h5py
        
        ptype_num = self.PARTICLE_TYPES.get(particle_type.lower())
        if ptype_num is None:
            raise ValueError(f"Unknown particle type: {particle_type}. "
                           f"Valid types: {list(self.PARTICLE_TYPES.keys())}")
        
        ptype_key = f'PartType{ptype_num}'
        
        positions = []
        masses = []
        
        for fpath in self._get_snapshot_files():
            with h5py.File(fpath, 'r') as f:
                if ptype_key not in f:
                    continue
                
                # Load positions
                pos = f[f'{ptype_key}/Coordinates'][:]
                positions.append(pos)
                
                # Load masses
                if f'{ptype_key}/Masses' in f:
                    mass = f[f'{ptype_key}/Masses'][:]
                else:
                    # Use mass table
                    mass_table = self._header.get('MassTable', 
                                                  f['Header'].attrs.get('MassTable'))
                    particle_mass = mass_table[ptype_num]
                    mass = np.full(len(pos), particle_mass)
                
                masses.append(mass)
        
        if not positions:
            return np.array([]).reshape(0, 3), np.array([])
        
        positions = np.concatenate(positions)
        masses = np.concatenate(masses)
        
        # Convert units (kpc/h -> Mpc/h)
        if positions.max() > 1000:
            positions = positions / 1000.0
        
        # Convert masses (10^10 M_sun/h -> M_sun/h)
        if masses.max() < 1e6:
            masses = masses * 1e10
        
        return positions.astype(np.float32), masses.astype(np.float32)
    
    def get_box_size(self) -> float:
        return self._box_size
    
    def get_cosmology(self) -> Optional[Dict[str, float]]:
        """Get cosmological parameters from header."""
        if self._header is None:
            return None
        
        cosmo = {}
        param_map = {
            'Omega0': 'Omega_m',
            'OmegaLambda': 'Omega_Lambda',
            'OmegaBaryon': 'Omega_b',
            'HubbleParam': 'h',
        }
        
        for gadget_key, cosmo_key in param_map.items():
            if gadget_key in self._header:
                cosmo[cosmo_key] = float(self._header[gadget_key])
        
        return cosmo if cosmo else None
    
    def get_redshift(self) -> Optional[float]:
        if self._header is None:
            return None
        return float(self._header.get('Redshift', None))


class CAMELSLoader(GadgetHDF5Loader):
    """
    Specialized loader for CAMELS simulations.
    
    Handles CAMELS-specific directory structure and parameter files.
    
    Args:
        sim_path: Path to simulation directory (e.g., /path/to/CV/CV_0)
        snapnum: Snapshot number (default: 90 for z=0)
        suite: Simulation suite ('IllustrisTNG' or 'SIMBA')
    """
    
    def __init__(
        self, 
        sim_path: str, 
        snapnum: int = 90, 
        suite: str = 'IllustrisTNG',
        **kwargs
    ):
        self.sim_path = Path(sim_path)
        self.suite = suite
        
        # Determine snapshot path
        snap_path = self._find_snapshot_path(snapnum)
        super().__init__(snap_path, snapnum, **kwargs)
    
    def _find_snapshot_path(self, snapnum: int) -> Path:
        """Find snapshot path in CAMELS directory structure."""
        # Try different formats
        candidates = [
            self.sim_path / f'snap_{snapnum:03d}.hdf5',
            self.sim_path / f'snapdir_{snapnum:03d}',
            self.sim_path / 'output' / f'snapdir_{snapnum:03d}',
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        raise FileNotFoundError(
            f"Snapshot {snapnum} not found in {self.sim_path}. "
            f"Tried: {candidates}"
        )
    
    def get_cosmology(self) -> Optional[Dict[str, float]]:
        """Get CAMELS parameters including astrophysical ones."""
        cosmo = super().get_cosmology() or {}
        
        # Try to load CAMELS parameter file
        param_file = self.sim_path / 'CosmoAstro_params.txt'
        if param_file.exists():
            with open(param_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        cosmo[parts[0]] = float(parts[1])
        
        return cosmo if cosmo else None


class SubFindCatalog(HaloCatalogLoader):
    """
    Loader for SubFind/FOF halo catalogs (AREPO/Gadget format).
    
    Args:
        path: Path to fof_subhalo_tab_*.hdf5 file
    """
    
    def __init__(self, path: str, **kwargs):
        super().__init__(path, **kwargs)
        self._load_catalog()
    
    def _load_catalog(self):
        """Load catalog from HDF5."""
        import h5py
        
        with h5py.File(self.path, 'r') as f:
            # Try Group (FOF) first, then Subhalo
            if 'Group' in f:
                self._positions = f['Group/GroupPos'][:]
                self._masses = f['Group/GroupMass'][:]
                
                if 'Group/Group_R_Crit200' in f:
                    self._radii = f['Group/Group_R_Crit200'][:]
                elif 'Group/Group_R_Mean200' in f:
                    self._radii = f['Group/Group_R_Mean200'][:]
                else:
                    # Estimate from mass
                    self._radii = self._estimate_r200(self._masses)
            
            elif 'Subhalo' in f:
                self._positions = f['Subhalo/SubhaloPos'][:]
                self._masses = f['Subhalo/SubhaloMass'][:]
                self._radii = f['Subhalo/SubhaloHalfmassRad'][:]
            
            else:
                raise ValueError("No Group or Subhalo data found in catalog")
            
            # Get header for unit conversion
            header = dict(f['Header'].attrs.items())
            self._box_size = header.get('BoxSize', 50000)
        
        # Convert units
        if self._positions.max() > 1000:
            self._positions = self._positions / 1000.0
        if self._radii.max() > 1000:
            self._radii = self._radii / 1000.0
        if self._masses.max() < 1e6:
            self._masses = self._masses * 1e10
    
    def _estimate_r200(self, masses: np.ndarray) -> np.ndarray:
        """Estimate R200 from mass using spherical collapse."""
        # R200 ~ (M / (200 * rho_crit * 4/3 * pi))^(1/3)
        # Simplified: R200 ~ 0.78 * (M / 1e14)^(1/3) Mpc/h
        return 0.78 * (masses / 1e14)**(1/3)
    
    def load_halos(
        self, 
        mass_threshold: float = 1e13
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load halos above mass threshold."""
        mask = self._masses >= mass_threshold
        
        return (
            self._positions[mask].astype(np.float32),
            self._masses[mask].astype(np.float32),
            self._radii[mask].astype(np.float32),
        )


class RockstarCatalog(HaloCatalogLoader):
    """
    Loader for Rockstar halo catalogs.
    
    Args:
        path: Path to Rockstar ASCII output file (e.g., halos_0.0.ascii)
    """
    
    def __init__(self, path: str, **kwargs):
        super().__init__(path, **kwargs)
        self._load_catalog()
    
    def _load_catalog(self):
        """Load Rockstar ASCII catalog."""
        import pandas as pd
        
        # Read header to get column names
        with open(self.path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    if 'scale' in line.lower() or 'id' in line.lower():
                        # This is the column header line
                        columns = line[1:].strip().split()
                        break
                else:
                    break
        
        # Read data
        self._data = pd.read_csv(
            self.path,
            comment='#',
            delim_whitespace=True,
            names=columns if columns else None
        )
        
        # Map column names
        self._pos_cols = ['x', 'y', 'z']
        self._mass_col = 'Mvir' if 'Mvir' in self._data.columns else 'mvir'
        self._radius_col = 'Rvir' if 'Rvir' in self._data.columns else 'rvir'
    
    def load_halos(
        self, 
        mass_threshold: float = 1e13
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load halos above mass threshold."""
        mask = self._data[self._mass_col] >= mass_threshold
        data = self._data[mask]
        
        positions = data[self._pos_cols].values
        masses = data[self._mass_col].values
        radii = data[self._radius_col].values
        
        # Rockstar outputs in Mpc/h and M_sun/h by default
        # Radii might be in kpc/h
        if radii.max() > 10:  # Likely kpc/h
            radii = radii / 1000.0
        
        return (
            positions.astype(np.float32),
            masses.astype(np.float32),
            radii.astype(np.float32),
        )


class CSVCatalog(HaloCatalogLoader):
    """
    Loader for generic CSV halo catalogs.
    
    Expected columns: x, y, z, mass, radius (optional)
    
    Args:
        path: Path to CSV file
        pos_cols: Column names for positions (default: ['x', 'y', 'z'])
        mass_col: Column name for mass (default: 'mass')
        radius_col: Column name for radius (default: 'radius')
        units: Dictionary with unit conversion factors
    """
    
    def __init__(
        self, 
        path: str,
        pos_cols: List[str] = None,
        mass_col: str = 'mass',
        radius_col: str = 'radius',
        units: Dict[str, float] = None,
        **kwargs
    ):
        super().__init__(path, **kwargs)
        self.pos_cols = pos_cols or ['x', 'y', 'z']
        self.mass_col = mass_col
        self.radius_col = radius_col
        self.units = units or {}
        self._load_catalog()
    
    def _load_catalog(self):
        """Load CSV catalog."""
        import pandas as pd
        self._data = pd.read_csv(self.path)
    
    def load_halos(
        self, 
        mass_threshold: float = 1e13
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load halos above mass threshold."""
        # Apply unit conversions
        pos_factor = self.units.get('position', 1.0)
        mass_factor = self.units.get('mass', 1.0)
        radius_factor = self.units.get('radius', 1.0)
        
        masses = self._data[self.mass_col].values * mass_factor
        mask = masses >= mass_threshold
        
        data = self._data[mask]
        masses = masses[mask]
        
        positions = data[self.pos_cols].values * pos_factor
        
        if self.radius_col in data.columns:
            radii = data[self.radius_col].values * radius_factor
        else:
            # Estimate from mass
            radii = 0.78 * (masses / 1e14)**(1/3)
        
        return (
            positions.astype(np.float32),
            masses.astype(np.float32),
            radii.astype(np.float32),
        )


# =============================================================================
# Factory Functions
# =============================================================================

def get_simulation_loader(
    path: str,
    format: str = 'auto',
    **kwargs
) -> SimulationLoader:
    """
    Factory function to get appropriate simulation loader.
    
    Args:
        path: Path to simulation
        format: Format type ('auto', 'gadget', 'camels')
        **kwargs: Additional arguments passed to loader
    
    Returns:
        SimulationLoader instance
    """
    if format == 'auto':
        # Try to auto-detect
        path = Path(path)
        if 'CV' in str(path) or 'SB' in str(path) or '1P' in str(path):
            format = 'camels'
        else:
            format = 'gadget'
    
    loaders = {
        'gadget': GadgetHDF5Loader,
        'camels': CAMELSLoader,
    }
    
    if format not in loaders:
        raise ValueError(f"Unknown format: {format}. Available: {list(loaders.keys())}")
    
    return loaders[format](path, **kwargs)


def get_halo_catalog(
    path: str,
    format: str = 'auto',
    **kwargs
) -> HaloCatalogLoader:
    """
    Factory function to get appropriate halo catalog loader.
    
    Args:
        path: Path to halo catalog
        format: Format type ('auto', 'subfind', 'rockstar', 'csv')
        **kwargs: Additional arguments passed to loader
    
    Returns:
        HaloCatalogLoader instance
    """
    path = Path(path)
    
    if format == 'auto':
        # Auto-detect from extension and content
        if path.suffix in ['.hdf5', '.h5']:
            format = 'subfind'
        elif path.suffix in ['.ascii', '.list', '.txt']:
            format = 'rockstar'
        elif path.suffix == '.csv':
            format = 'csv'
        else:
            raise ValueError(f"Cannot auto-detect format for {path}")
    
    loaders = {
        'subfind': SubFindCatalog,
        'rockstar': RockstarCatalog,
        'csv': CSVCatalog,
    }
    
    if format not in loaders:
        raise ValueError(f"Unknown format: {format}. Available: {list(loaders.keys())}")
    
    return loaders[format](path, **kwargs)
