import numpy as np
import torch
import h5py
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import torch.nn.functional as F
from workflow_utils import ConfigLoader, ModelManager, sample
from vdm.constants import norms_256 as norms
import MAS_library as MASL

class BIND:
    """
    A unified class for the BIND (Baryonic Inference from N-body Data) pipeline.
    Combines voxelization, halo extraction, diffusion-based generation, and halo pasting.
    """
    
    def __init__(self, simulation_path: str, snapnum: int, boxsize: float, 
                 gridsize: int = 1024, subimage_size: int = 128, 
                 mass_threshold: float = 1e13, config_path: str = None, 
                 output_dir: str = '/mnt/home/mlee1/ceph/BIND3d/sim_output',
                 r_in_factor: float = 1.0, r_out_factor: float = 4.0,
                 device: str = 'cuda', verbose: bool = True, dim: str = '3d', axis: int = 2):
        """
        Initialize the BIND pipeline.
        
        Args:
            simulation_path (str): Path to the simulation directory.
            snapnum (int): Snapshot number.
            boxsize (float): Size of the simulation box in kpc/h.
            gridsize (int): Initial grid size for voxelization (e.g., 1024).
            subimage_size (int): Size of extracted subimages (e.g., 32).
            mass_threshold (float): Minimum halo mass in 10^10 M_sun/h.
            config_path (str): Path to diffusion model config.
            output_dir (str): Directory for outputs.
            r_in_factor (float): Inner radius factor for pasting.
            r_out_factor (float): Outer radius factor for pasting.
            device (str): Device for computations ('cuda' or 'cpu').
            verbose (bool): Enable verbose output.
        """
        self.simulation_path = simulation_path
        self.snapnum = snapnum
        self.boxsize = boxsize
        self.gridsize = gridsize
        self.subimage_size = subimage_size
        self.mass_threshold = mass_threshold
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.r_in_factor = r_in_factor
        self.r_out_factor = r_out_factor
        self.device = device
        self.verbose = verbose
        
        # Add dim and axis
        self.dim = dim
        self.axis = axis
        
        # Load config to get cropsize if config_path is provided
        self.cropsize = 128  # default
        self.num_large_scales = 0  # default: no large-scale conditioning
        self.quantile_path = None  # default: no quantile normalization
        self.use_quantile_normalization = False  # default: use Z-score for stellar channel
        
        if self.config_path is not None:
            try:
                config = ConfigLoader(self.config_path, verbose=False)
                self.cropsize = getattr(config, 'cropsize', 128)
                self.num_large_scales = getattr(config, 'large_scale_channels', 0)
                self.quantile_path = getattr(config, 'quantile_path', None)
                
                # Enable quantile normalization if path is provided
                if self.quantile_path is not None:
                    self.use_quantile_normalization = True
                    if self.verbose:
                        print(f"[BIND] 🌟 Quantile normalization enabled: {self.quantile_path}")
                
                if self.verbose and self.num_large_scales > 0:
                    print(f"[BIND] Loaded large_scale_channels from config: {self.num_large_scales}")
            except Exception as e:
                if self.verbose:
                    print(f"[BIND] Could not load config, using defaults: {e}")
        
        self.subimage_size = self.cropsize  # Set subimage size to cropsize to avoid resizing
        self.paste_gridsize = self.gridsize  # No resizing, use original gridsize
        
        # Load normalization stats from .npz files (matches training data)
        # These files are in the project root
        from .workflow_utils import load_normalization_stats
        norm_stats = load_normalization_stats()  # Uses project root by default
        
        # Set normalization parameters
        # Input (DM condition): use DM stats
        self.input_mean = norms['IllustrisTNG'][6]
        self.input_std = norms['IllustrisTNG'][7]

        # Targets (DM, Gas, Stars): use respective stats
        self.target_means = np.array([
            norm_stats['dm_mag_mean'],    # DM target
            norm_stats['gas_mag_mean'],   # Gas target
            norm_stats['star_mag_mean']   # Stellar target
        ])
        self.target_stds = np.array([
            norm_stats['dm_mag_std'],     # DM target
            norm_stats['gas_mag_std'],    # Gas target
            norm_stats['star_mag_std']    # Stellar target
        ])
        
        if self.verbose:
            print(f"[BIND] Loaded normalization from .npz files:")
            print(f"  DM condition: mean={self.input_mean:.6f}, std={self.input_std:.6f}")
            print(f"  DM target:   mean={norm_stats['dm_mag_mean']:.6f}, std={norm_stats['dm_mag_std']:.6f}")
            print(f"  Gas target:  mean={norm_stats['gas_mag_mean']:.6f}, std={norm_stats['gas_mag_std']:.6f}")
            if self.use_quantile_normalization:
                print(f"  Stars target: QUANTILE TRANSFORMATION ({self.quantile_path})")
            else:
                print(f"  Stars target: mean={norm_stats['star_mag_mean']:.6f}, std={norm_stats['star_mag_std']:.6f}")
        
        # Initialize components
        self.sim_grid = None
        self.halo_catalog = None
        self.extracted = None
        self.generated_images = None
        self.final_maps = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("[BIND] Initialized BIND pipeline.")
    
    def voxelize_simulation(self) -> np.ndarray:
        """
        Voxelize the N-body simulation into a 3D grid or project to 2D using CIC interpolation.
        
        Returns:
            np.ndarray: 3D grid of shape (gridsize, gridsize, gridsize) or 2D (gridsize, gridsize).
        """
        if self.verbose:
            print("[BIND] Voxelizing simulation...")
        
        if self.dim == '3d':
            # Read coordinates and mass
            snappath = f"{self.simulation_path}/snap_{self.snapnum:03d}.hdf5"
            with h5py.File(snappath, 'r') as f:
                coordinates = f['PartType1']['Coordinates'][:]
                header = dict(f['Header'].attrs.items())
                dm_mass = header['MassTable'][1] * 1e10
            
            # Convert units to match process_simulations.py: kpc/h -> Mpc/h
            pos = coordinates / 1000.0
            boxsize_mpc = self.boxsize / 1000.0
            
            # Use MASL.MA for 3D voxelization (same as process_simulations.py)
            pos = np.ascontiguousarray(pos.astype(np.float32))
            mass = np.full(len(pos), dm_mass, dtype=np.float32)
            mass = np.ascontiguousarray(mass)
            
            self.sim_grid = np.zeros((self.gridsize, self.gridsize, self.gridsize), dtype=np.float32)
            MASL.MA(pos, self.sim_grid, np.float32(boxsize_mpc), MAS='CIC', W=mass, verbose=False)
        else:
            # 2D projection
            snappath = f"{self.simulation_path}/snap_{self.snapnum:03d}.hdf5"
            with h5py.File(snappath, 'r') as f:
                coordinates = f['PartType1']['Coordinates'][:]
                header = dict(f['Header'].attrs.items())
                dm_mass = header['MassTable'][1] * 1e10
            coords = coordinates / 1000.0  # to Mpc/h
            boxsize_mpc = self.boxsize / 1000.0
            pos = coords.astype(np.float32)
            mass = np.full(len(pos), dm_mass, dtype=np.float32)
            axes = [0,1,2]
            proj_axes = axes[:self.axis] + axes[self.axis+1:]
            pos_proj = pos[:, proj_axes]
            self.sim_grid = np.zeros((self.gridsize, self.gridsize), dtype=np.float32)
            MASL.MA(pos_proj, self.sim_grid, boxsize_mpc, MAS='CIC', W=mass, verbose=False)
        
        if self.verbose:
            print(f"[BIND] {'Voxelization' if self.dim == '3d' else 'Projection'} complete. Grid shape: {self.sim_grid.shape}")
        return self.sim_grid
    
    def load_halo_catalog(self) -> Dict:
        """
        Load and filter halo catalog.
        
        Returns:
            Dict: Filtered halo data.
        """
        halopath = self.simulation_path.replace('Sims', 'FOF_Subfind')
        hdf5_path = f"{halopath}/fof_subhalo_tab_{self.snapnum:03d}.hdf5"
        
        if self.verbose:
            print(f"[BIND] Loading halo catalog from {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            positions = f['Group/GroupPos'][:]
            masses = f['Group/Group_M_Crit200'][:]
            radii = f['Group/Group_R_Crit200'][:]
        
        mask = masses * 1e10 >= self.mass_threshold
        self.halo_catalog = {
            'positions': positions[mask],
            'masses': masses[mask],
            'radii': radii[mask],
            'indices': np.where(mask)[0]
        }
        
        if self.verbose:
            print(f"[BIND] Loaded {len(self.halo_catalog['positions'])} halos above mass threshold.")
        return self.halo_catalog
    
    def extract_halos(self, omega_m, use_large_scale: bool = True, 
                      num_large_scales: int = None, target_res: int = 128) -> Dict:
        """
        Extract halos from the gridded simulation.
        
        IMPORTANT: This follows the exact same multi-scale extraction as process_simulations2_cpu.py
        to ensure compatibility with the trained diffusion model.
        
        Args:
            omega_m: Omega_m parameter for normalization
            use_large_scale: Whether to extract large-scale context (default: True)
            num_large_scales: Number of large-scale maps to extract (default: None, uses value from config).
                            Scales are fixed to match training: [6.25, 12.5, 25.0, 50.0] Mpc/h
                            - num_large_scales=0: Only 6.25 Mpc condition (no large-scale)
                            - num_large_scales=3: All scales [6.25, 12.5, 25.0, 50.0] Mpc/h
            target_res: Target resolution for all scales (default: 128)
        
        Returns:
            Dict: Extraction results with metadata and conditions.
        """
        if self.sim_grid is None:
            raise ValueError("Simulation grid not available. Run voxelize_simulation() first.")
        if self.halo_catalog is None:
            self.load_halo_catalog()
        
        # Use config value if num_large_scales not specified
        if num_large_scales is None:
            num_large_scales = self.num_large_scales
            if self.verbose and num_large_scales > 0:
                print(f"[BIND] Using num_large_scales from config: {num_large_scales}")
        
        # If use_large_scale is False, set num_large_scales to 0
        if not use_large_scale:
            num_large_scales = 0
        
        # Fixed scales to match training data generation
        scales_mpc = [6.25, 12.5, 25.0, 50.0]
        
        if self.verbose:
            print("[BIND] Extracting halos...")
            if num_large_scales > 0:
                # Large-scale maps are the scales AFTER the first one (6.25 Mpc is the condition)
                large_scale_sizes = scales_mpc[1:num_large_scales+1]
                print(f"[BIND] Extracting condition at 6.25 Mpc + {num_large_scales} large-scale map(s) at {large_scale_sizes} Mpc, {target_res}^{2 if self.dim == '2d' else 3} resolution...")
            else:
                print("[BIND] Extracting condition at 6.25 Mpc only (no large-scale conditioning)")
        
        # Convert halo positions from kpc/h to Mpc/h (to match process_simulations.py)
        halo_positions_mpc = self.halo_catalog['positions'] / 1000.0
        boxsize_mpc = self.boxsize / 1000.0
        
        extraction_results = {
            'halo_count': len(halo_positions_mpc),
            'file_paths': {},
            'metadata': [],
            'conditions': [],
            'large_scale_conditions': [] if num_large_scales > 0 else None,
            'num_large_scales': num_large_scales
        }
        
        for i, pos in enumerate(halo_positions_mpc):
            # Extract all scales (matching process_simulations2_cpu.py extract_multiscale_cutouts)
            # scales_mpc = [6.25, 12.5, 25.0, 50.0]
            all_scale_maps = []
            
            for scale_idx, scale_size in enumerate(scales_mpc):
                if self.dim == '3d':
                    scale_map = self._extract_region_3d(self.sim_grid, pos, scale_size, 
                                                        boxsize_mpc, self.gridsize)
                    # Downsample to target_res if needed
                    if scale_map.shape[0] != target_res:
                        scale_map = self._downsample_3d(scale_map, target_res)
                else:
                    scale_map = self._extract_projection_2d(self.sim_grid, pos, scale_size, 
                                                            boxsize_mpc, self.gridsize, self.axis)
                    # Downsample to target_res if needed
                    if scale_map.shape[0] != target_res:
                        scale_map = self._downsample_2d(scale_map, target_res)
                    # scale_map /= omega_m  # Scale by omega_m for 2D
                
                scale_map_normalized = self._normalize_condition(scale_map)
                all_scale_maps.append(scale_map_normalized)
            
            # First scale (6.25 Mpc) is the condition
            condition_32 = all_scale_maps[0]
            
            # Remaining scales are large-scale context (if requested)
            large_scale_maps = []
            if num_large_scales > 0:
                large_scale_maps = all_scale_maps[1:num_large_scales+1]
            
            # Compute center pixel from position
            pix_size = boxsize_mpc / self.gridsize
            if self.dim == '3d':
                center_pixel = (pos / pix_size).astype(int)
            else:
                # Project to 2D
                axes = [0, 1, 2]
                proj_axes = axes[:self.axis] + axes[self.axis+1:]
                center_pixel = (pos[proj_axes] / pix_size).astype(int)
            
            halo_metadata = {
                'halo_id': i,
                'original_index': self.halo_catalog['indices'][i],
                'position': self.halo_catalog['positions'][i],
                'mass': self.halo_catalog['masses'][i],
                'radius': self.halo_catalog['radii'][i],
                'center_pixel': center_pixel,
            }
            
            # Save with large_scale if available
            output_path = self.output_dir / f"halo_{i}.npz"
            if num_large_scales > 0 and large_scale_maps:
                # Stack all large-scale maps into a single array (num_scales, H, W) or (num_scales, H, W, D)
                if self.dim == '3d':
                    large_scale_stacked = np.stack(large_scale_maps, axis=0)  # (num_scales, H, W, D)
                else:
                    large_scale_stacked = np.stack(large_scale_maps, axis=0)  # (num_scales, H, W)
                np.savez(output_path, condition=condition_32, large_scale=large_scale_stacked, 
                        metadata=halo_metadata)
            else:
                np.savez(output_path, condition=condition_32, metadata=halo_metadata)
            
            extraction_results['file_paths'][i] = {'condition': str(output_path)}
            extraction_results['metadata'].append(halo_metadata)
            extraction_results['conditions'].append(condition_32)
            if num_large_scales > 0:
                extraction_results['large_scale_conditions'].append(large_scale_maps)
        
        self.extracted = extraction_results
        if self.verbose:
            print(f"[BIND] Extracted {len(extraction_results['conditions'])} halos.")
        return self.extracted
    
    def _downsample_2d(self, field: np.ndarray, target_res: int) -> np.ndarray:
        """
        Downsample 2D field to target resolution using averaging.
        
        Args:
            field: 2D array to downsample
            target_res: Target resolution
            
        Returns:
            Downsampled 2D array
        """
        current_res = field.shape[0]
        factor = current_res // target_res
        if factor <= 1:
            return field
        
        # Reshape and average
        downsampled = field.reshape(target_res, factor, target_res, factor).mean(axis=(1, 3))
        return downsampled
    
    def _downsample_3d(self, field: np.ndarray, target_res: int) -> np.ndarray:
        """
        Downsample 3D field to target resolution using averaging.
        
        Args:
            field: 3D array to downsample
            target_res: Target resolution
            
        Returns:
            Downsampled 3D array
        """
        current_res = field.shape[0]
        factor = current_res // target_res
        if factor <= 1:
            return field
        
        # Reshape and average
        downsampled = field.reshape(target_res, factor, target_res, factor, target_res, factor).mean(axis=(1, 3, 5))
        return downsampled
    
    def _extract_region_3d(self, field_3d: np.ndarray, center: np.ndarray, size: float, box_size: float, resolution: int) -> np.ndarray:
        """
        Extract 3D region around center with periodic boundaries.
        Matches process_simulations.py extract_region function.
        
        Args:
            field_3d: 3D field to extract from
            center: Center position in Mpc/h
            size: Size of region in Mpc
            box_size: Box size in Mpc/h
            resolution: Grid resolution
        """
        pix_size = box_size / resolution
        half_size_pix = int(size / (2 * pix_size))
        full_size_pix = 2 * half_size_pix
        center_pix = (center / pix_size).astype(int)
        
        # Periodic indices for each dimension
        ix = []
        for d in range(3):
            start = center_pix[d] - half_size_pix
            indices = (start + np.arange(full_size_pix)) % resolution
            ix.append(indices)
        
        extracted = field_3d[np.ix_(ix[0], ix[1], ix[2])]
        return extracted
    
    def _extract_projection_2d(self, field_2d: np.ndarray, center: np.ndarray, size: float, box_size: float, resolution: int, axis: int = 2) -> np.ndarray:
        """
        Extract 2D projection around center with periodic boundaries.
        Matches process_simulations.py extract_projection function.
        
        Args:
            field_2d: 2D field to extract from
            center: Center position in Mpc/h (3D position)
            size: Size of region in Mpc
            box_size: Box size in Mpc/h
            resolution: Grid resolution
            axis: Projection axis (0=x, 1=y, 2=z)
        """
        pix_size = box_size / resolution
        half_size_pix = int(size / (2 * pix_size))
        full_size_pix = 2 * half_size_pix
        
        if axis == 2:
            center_2d = center[:2]  # x,y
        elif axis == 1:
            center_2d = center[[0, 2]]  # x,z
        elif axis == 0:
            center_2d = center[[1, 2]]  # y,z
        
        center_pix = (center_2d / pix_size).astype(int)
        
        # Periodic indices for each dimension
        ix = []
        for d in range(2):
            start = center_pix[d] - half_size_pix
            indices = (start + np.arange(full_size_pix)) % resolution
            ix.append(indices)
        
        extracted = field_2d[np.ix_(ix[0], ix[1])]
        return extracted
    
    def _extract_subimage_3d(self, simulation_data: np.ndarray, center_pixel: Tuple) -> np.ndarray:
        """Extract 3D subvolume with periodic boundaries."""
        half_size = self.subimage_size // 2
        grid_size = simulation_data.shape[0]
        start = [(center - half_size) % grid_size for center in center_pixel]
        stop = [(center + half_size) % grid_size for center in center_pixel]
        
        if all(start[axis] < stop[axis] for axis in range(3)):
            subvolume = simulation_data[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
        else:
            subvolume = np.zeros((self.subimage_size, self.subimage_size, self.subimage_size))
            for i in range(self.subimage_size):
                for j in range(self.subimage_size):
                    for k in range(self.subimage_size):
                        x_idx = (start[0] + i) % grid_size
                        y_idx = (start[1] + j) % grid_size
                        z_idx = (start[2] + k) % grid_size
                        subvolume[i, j, k] = simulation_data[x_idx, y_idx, z_idx]
        return subvolume
    
    def _extract_subimage_2d(self, simulation_data: np.ndarray, center_pixel: Tuple) -> np.ndarray:
        half_size = self.subimage_size // 2
        grid_size = simulation_data.shape[0]
        start = [(center - half_size) % grid_size for center in center_pixel]
        stop = [(center + half_size) % grid_size for center in center_pixel]
        
        if all(start[axis] < stop[axis] for axis in range(2)):
            subvolume = simulation_data[start[0]:stop[0], start[1]:stop[1]]
        else:
            subvolume = np.zeros((self.subimage_size, self.subimage_size))
            for i in range(self.subimage_size):
                for j in range(self.subimage_size):
                    x_idx = (start[0] + i) % grid_size
                    y_idx = (start[1] + j) % grid_size
                    subvolume[i, j] = simulation_data[x_idx, y_idx]
        return subvolume
    
    def _normalize_condition(self, condition: np.ndarray) -> np.ndarray:
        """Normalize condition: log10(condition + 1), then standardize."""
        condition_log = np.log10(condition + 1)
        return (condition_log - self.input_mean) / self.input_std
    
    def generate_halos(self, batch_size: int = 10, conditional_params: np.ndarray = None, 
                      cleanup_conditions: bool = True, use_large_scale: bool = None,
                      conserve_mass: bool = True) -> List:
        """
        Generate halos using the diffusion model.
        
        Args:
            batch_size (int): Batch size for generation.
            conditional_params (np.ndarray): Conditional parameters.
            cleanup_conditions (bool): If True, delete individual halo_*.npz files after generation.
            use_large_scale (bool): Whether to use large-scale conditioning. If None, auto-detect from extracted data.
            conserve_mass (bool): If True, normalize generated halos so total mass (DM+Gas+Stars) equals DMO cutout mass.
        
        Returns:
            List: Generated halo images.
        """
        if self.extracted is None:
            raise ValueError("Halos not extracted. Run extract_halos() first.")
        if self.config_path is None:
            raise ValueError("Config path not provided.")
        
        # Auto-detect large_scale usage if not specified
        if use_large_scale is None:
            use_large_scale = self.extracted.get('large_scale_conditions') is not None
        
        if self.verbose:
            print("[BIND] Generating halos with diffusion model...")
            if use_large_scale:
                print("[BIND] Using large-scale conditioning")
            if conserve_mass:
                print(f"[BIND] Mass conservation enabled: generated (DM+Gas+Stars) will match DMO cutout mass")
        
        # Initialize diffusion processor
        config = ConfigLoader(self.config_path, verbose=self.verbose)
        # Skip dataset loading for inference (much faster!)
        hydro, vdm_model = ModelManager.initialize(config, verbose=self.verbose, skip_data_loading=True)
        vdm_model.to(self.device)
        
        # Prepare conditions
        conditions = np.array(self.extracted['conditions'])
        conditions = torch.tensor(conditions, dtype=torch.float32).to(self.device)
        
        # Prepare large_scale conditions if available
        if use_large_scale and self.extracted.get('large_scale_conditions') is not None:
            # large_scale_conditions is a list of lists: [halo_i][scale_j] -> array
            # Convert to numpy array of shape (N_halos, num_scales, H, W) or (N_halos, num_scales, H, W, D)
            large_scale_list = self.extracted['large_scale_conditions']
            num_halos = len(large_scale_list)
            num_scales = self.extracted.get('num_large_scales', 1)
            
            # Stack all scales for all halos
            if self.dim == '3d':
                # Each element is (H, W, D), stack to (num_halos, num_scales, H, W, D)
                large_scale_stacked = np.array([[large_scale_list[i][j] for j in range(num_scales)] 
                                                for i in range(num_halos)])
            else:
                # Each element is (H, W), stack to (num_halos, num_scales, H, W)
                large_scale_stacked = np.array([[large_scale_list[i][j] for j in range(num_scales)] 
                                                for i in range(num_halos)])
            
            large_scale_conditions = torch.tensor(large_scale_stacked, dtype=torch.float32).to(self.device)
            
            # Concatenate along channel dimension
            # Base condition: (N, H, W) or (N, H, W, D)
            # Large-scale: (N, num_scales, H, W) or (N, num_scales, H, W, D)
            # Result: (N, 1+num_scales, H, W) or (N, 1+num_scales, H, W, D)
            conditions = conditions.unsqueeze(1)  # (N, 1, H, W[, D])
            
            # Concatenate: (N, 1+num_scales, H, W[, D])
            conditions = torch.cat([conditions, large_scale_conditions], dim=1)
        else:
            # No large_scale, just add channel dimension
            if self.dim == '3d':
                conditions = conditions.unsqueeze(1)  # (N, 1, H, W, D)
            else:
                conditions = conditions.unsqueeze(1)  # (N, 1, H, W)
        
        conditional_params = torch.tensor(conditional_params, dtype=torch.float32).to(self.device) if conditional_params is not None else None
        
        # Extract DMO cutouts for mass normalization if needed
        dmo_cutouts = []
        if conserve_mass:
            if self.verbose:
                print("[BIND] Extracting DMO cutouts for mass normalization...")
            # Extract DMO cutouts for each halo
            boxsize_mpc = self.boxsize / 1000.0
            omega_m = 1#conditional_params[0, 0].to('cpu').numpy() if conditional_params is not None else 0.3
            for i, meta in enumerate(self.extracted['metadata']):
                pos_mpc = meta['position'] / 1000.0
                if self.dim == '3d':
                    dmo_cutout = self._extract_region_3d(self.sim_grid, pos_mpc, 6.25, boxsize_mpc, self.gridsize)
                else:
                    dmo_cutout = self._extract_projection_2d(self.sim_grid, pos_mpc, 6.25, boxsize_mpc, self.gridsize, self.axis)
                    dmo_cutout /= omega_m
                dmo_cutouts.append(dmo_cutout)
        
        generated_samples = sample(vdm_model, conditions, batch_size=batch_size, conditional_params=conditional_params)
        print(generated_samples.shape)
        generated_halos = []
        for i in range(len(generated_samples)):
            unnormalized = self._unnormalize_target(generated_samples[i].numpy())
            # unnormalized[:, 0] *= conditional_params[0, 0].to('cpu').numpy()
            # # print(conditional_params[0, 0].to('cpu').numpy())
            # unnormalized[:, 1] *= conditional_params[0, 6].to('cpu').numpy()
            # unnormalized[:, 2] *= conditional_params[0, 6].to('cpu').numpy()
            
            # Apply mass conservation normalization if enabled
            if conserve_mass:
                unnormalized = self._normalize_generated_mass(unnormalized, dmo_cutouts[i])
            
            generated_halos.append(unnormalized)
        
        self.generated_images = generated_halos
        
        # Clean up individual halo condition files if requested
        if cleanup_conditions:
            if self.verbose:
                print(f"[BIND] Cleaning up {len(self.extracted['file_paths'])} halo condition files...")
            for i in range(len(self.extracted['file_paths'])):
                halo_file = self.output_dir / f"halo_{i}.npz"
                if halo_file.exists():
                    halo_file.unlink()
        
        if self.verbose:
            print(f"[BIND] Generated {len(generated_halos)} halo sets.")
        return self.generated_images
    
    def _unnormalize_target(self, target_norm: np.ndarray) -> np.ndarray:
        """
        Unnormalize target from z-normalized log10 field back to physical units.
        
        Supports both Z-score normalization (legacy) and quantile normalization 
        for the stellar channel (new).
        
        Args:
            target_norm: Normalized target (n_realizations, 3, H, W[, D])
                        Channels are [dm, gas, stars]
        
        Returns:
            Unnormalized field in physical units (M_sun)
        """
        import joblib
        
        # Use pre-loaded normalization stats
        # Unnormalize each channel separately
        field_unnorm = np.empty_like(target_norm)
        
        # Unnormalize DM and Gas channels using Z-score normalization
        # target_means and target_stds are arrays of shape (3,) for [dm, gas, stars]
        field_unnorm[:, 0] = target_norm[:, 0] * self.target_stds[0] + self.target_means[0]
        field_unnorm[:, 1] = target_norm[:, 1] * self.target_stds[1] + self.target_means[1]
        
        # Unnormalize stellar channel
        if self.use_quantile_normalization and self.quantile_path is not None:
            # Use quantile transformer for stellar channel
            quantile_transformer = joblib.load(self.quantile_path)
            
            # Apply inverse quantile transformation for each realization
            for r in range(target_norm.shape[0]):
                original_shape = target_norm[r, 2].shape
                stellar_flat = target_norm[r, 2].flatten().reshape(-1, 1)
                stellar_unnorm_flat = quantile_transformer.inverse_transform(stellar_flat)
                field_unnorm[r, 2] = stellar_unnorm_flat.reshape(original_shape)
            
            # Note: quantile transformer outputs are already in log10 space,
            # so we apply the exponential transformation below
        else:
            # Use Z-score normalization for stellar channel
            field_unnorm[:, 2] = target_norm[:, 2] * self.target_stds[2] + self.target_means[2]
        
        # Convert from log10 back to physical units
        return 10 ** field_unnorm - 1
    
    def _normalize_generated_mass(self, generated_halo: np.ndarray, dmo_cutout: np.ndarray) -> np.ndarray:
        """
        Normalize generated halo to conserve total mass.
        
        This addresses the issue where the diffusion model doesn't conserve mass, 
        which affects large-scale power spectrum. We normalize the generated halo 
        so its total mass (DM + Gas + Stars) matches the total mass in the DMO cutout.
        
        Args:
            generated_halo: Generated halo (n_realizations, n_components, H, W[, D])
            dmo_cutout: DMO cutout for this halo (H, W[, D])
        
        Returns:
            Normalized generated halo with conserved mass
        """
        # Total mass in DMO cutout - this is our reference
        dmo_total_mass = np.sum(dmo_cutout)
        
        # For each realization, normalize so that sum of all three channels equals DMO mass
        normalized_halos = []
        for realization in generated_halo:
            # Total generated mass across all three components (DM + Gas + Stars)
            generated_total_mass = np.sum(realization)
            
            if generated_total_mass > 0:
                # Compute normalization factor
                normalization_factor = dmo_total_mass / generated_total_mass
                
                # Apply normalization to all components equally
                normalized_realization = realization * normalization_factor
                normalized_halos.append(normalized_realization)
            else:
                # If generated mass is zero, keep as is
                normalized_halos.append(realization)
        
        return np.array(normalized_halos)
    


    def paste_halos(self, realizations: int = 10, use_enhanced: bool = False) -> List[np.ndarray]:
        """
        Paste generated halos back into the simulation grid.
        
        Args:
            realizations (int): Number of realizations to generate.
            use_enhanced (bool): Use enhanced pasting with weighting for overlaps (2D only).
        
        Returns:
            List[np.ndarray]: Final blended maps.
        """
        if self.sim_grid is None or self.extracted is None or self.generated_images is None:
            raise ValueError("Required data not available. Run previous steps first.")
        
        if self.verbose:
            print("[BIND] Pasting halos...")
        
        # Resize sim_grid to 256^3 if needed
        sim_grid_resized = self.sim_grid
        
        # Get center pixels from metadata (already computed during extraction)
        center_pixels = [meta['center_pixel'] for meta in self.extracted['metadata']]
        masses = [meta['mass'] for meta in self.extracted['metadata']]
        radii = [meta['radius'] for meta in self.extracted['metadata']]
        
        # Sort by mass
        ids = np.argsort(masses)
        
        final_maps = []
        for realization in range(realizations):
            if self.dim == '3d':
                paster = HaloPaster3D(box_size_kpc=self.boxsize, r_in_factor=self.r_in_factor, r_out_factor=self.r_out_factor)
                final_map, _, _, _, _ = paster.enhanced_halo_pasting_3d(
                    original_map=sim_grid_resized,
                    halo_patches=self.generated_images,
                    centers_pixels=center_pixels,
                    radii_kpc=radii,
                    realization=realization
                )
            else:
                if use_enhanced:
                    paster = HaloPaster2D(box_size_kpc=self.boxsize, r_in_factor=self.r_in_factor, r_out_factor=self.r_out_factor)
                    final_map, _, _, _, _ = paster.enhanced_halo_pasting_2d(
                        original_map=sim_grid_resized,
                        halo_patches=self.generated_images,
                        centers_pixels=center_pixels,
                        radii_kpc=radii,
                        realization=realization
                    )
                else:
                    final_map = sim_grid_resized.copy()
                    grid_size = final_map.shape[0]
                    
                    for id in ids:
                        cp = center_pixels[id]
                        generated_halo = self.generated_images[id][realization].sum(axis=0)
                        patch_size = generated_halo.shape[0]
                        half_patch = patch_size // 2
                        
                        for i in range(patch_size):
                            for j in range(patch_size):
                                x_idx = (cp[0] - half_patch + i) % grid_size
                                y_idx = (cp[1] - half_patch + j) % grid_size
                                final_map[x_idx, y_idx] = generated_halo[i, j]
            
            final_maps.append(final_map)
        
        self.final_maps = final_maps
        if self.verbose:
            print(f"[BIND] Pasting complete. Generated {len(final_maps)} realizations.")
        return self.final_maps
    
    def run(self, batch_size: int = 10, realizations: int = 10, conditional_params: np.ndarray = None, 
            use_enhanced_pasting: bool = False, use_large_scale: bool = True,
            num_large_scales: int = None, target_res: int = 128) -> List[np.ndarray]:
        """
        Run the full BIND pipeline.
        
        Args:
            batch_size (int): Batch size for generation.
            realizations (int): Number of pasting realizations.
            conditional_params (np.ndarray): Conditional parameters.
            use_enhanced_pasting (bool): Use enhanced pasting with weighting for overlaps (2D only).
            use_large_scale (bool): Whether to extract and use large-scale conditioning (default: True).
            num_large_scales (int): Number of large-scale maps to extract (default: None, uses value from config).
                                   Scales are fixed to match training: [6.25, 12.5, 25.0, 50.0] Mpc/h
                                   - num_large_scales=0: Only 6.25 Mpc condition (no large-scale)
                                   - num_large_scales=3: All scales [6.25, 12.5, 25.0, 50.0] Mpc/h
            target_res (int): Target resolution for all scales (default: 128).
        
        Returns:
            List[np.ndarray]: Final blended maps.
        """
        
        self.voxelize_simulation()
        if conditional_params is not None:
            omega_m = 1#conditional_params[0]
        else:
            omega_m = 1#0.315  # Default value
        self.extract_halos(omega_m, use_large_scale=use_large_scale, 
                          num_large_scales=num_large_scales, target_res=target_res)
        self.generate_halos(batch_size=batch_size, conditional_params=conditional_params, 
                           use_large_scale=use_large_scale)
        return self.paste_halos(realizations=realizations, use_enhanced=use_enhanced_pasting)


class HaloPaster3D:
    """Halo pasting class for 3D overlap handling."""
    
    def __init__(self, box_size_kpc: float, r_in_factor: float = 1.0, r_out_factor: float = 3.5):
        self.box_size_kpc = box_size_kpc
        self.r_in_factor = r_in_factor
        self.r_out_factor = r_out_factor
    
    def _create_weight_function(self, r: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
        W = np.zeros_like(r)
        mask_inner = r <= r_in
        W[mask_inner] = 1.0
        mask_transition = (r > r_in) & (r <= r_out)
        W[mask_transition] = 0.5 * (1 + np.cos(np.pi * (r[mask_transition] - r_in) / (r_out - r_in)))
        return W
    
    def enhanced_halo_pasting_3d(self, original_map: np.ndarray, halo_patches: List, 
                                centers_pixels: List, radii_kpc: List, realization: int = 0) -> Tuple:
        pixel_scale_kpc = self.box_size_kpc / original_map.shape[0]
        map_size = original_map.shape[0]
        
        total_weight_map = np.zeros_like(original_map)
        individual_weights = []
        individual_halos = []
        valid_regions = []
        
        for idx in range(len(halo_patches)):
            halo_patch = halo_patches[idx][realization].sum(axis=0)
            center_pixel = centers_pixels[idx]
            radius_kpc = radii_kpc[idx]
            radius_pixels = radius_kpc / pixel_scale_kpc
            
            cx, cy, cz = int(center_pixel[0]), int(center_pixel[1]), int(center_pixel[2])
            patch_size = halo_patch.shape[0]
            half_patch = patch_size // 2
            
            r_in = self.r_in_factor * radius_pixels
            max_r_out_from_patch = half_patch - 2
            r_out_unlimited = self.r_out_factor * radius_pixels
            r_out = min(r_out_unlimited, max_r_out_from_patch)
            if r_out <= r_in:
                r_out = r_in + 2.0
            
            # Create mesh grid for radial distance calculation
            x_coords, y_coords, z_coords = np.meshgrid(
                np.arange(patch_size) - half_patch,
                np.arange(patch_size) - half_patch,
                np.arange(patch_size) - half_patch,
                indexing='ij'
            )
            r_coords = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
            W = self._create_weight_function(r_coords, r_in, r_out)
            
            # Store for this halo
            individual_weights.append(W)
            individual_halos.append(halo_patch)
            valid_regions.append((cx, cy, cz, half_patch))
            
            # Add weights to total weight map with periodic boundaries (vectorized)
            i_indices = (cx - half_patch + np.arange(patch_size)) % map_size
            j_indices = (cy - half_patch + np.arange(patch_size)) % map_size
            k_indices = (cz - half_patch + np.arange(patch_size)) % map_size
            
            # Use advanced indexing with meshgrid
            ii, jj, kk = np.meshgrid(i_indices, j_indices, k_indices, indexing='ij')
            np.add.at(total_weight_map, (ii, jj, kk), W)
        
        normalized_total_weight = np.zeros_like(original_map)
        normalized_halo_sum = np.zeros_like(original_map)
        
        # Apply normalized weights and paste halos with periodic boundaries (vectorized)
        for i, (W, halo, (cx, cy, cz, half_patch)) in enumerate(zip(
            individual_weights, individual_halos, valid_regions
        )):
            patch_size = halo.shape[0]
            
            # Generate periodic indices
            i_indices = (cx - half_patch + np.arange(patch_size)) % map_size
            j_indices = (cy - half_patch + np.arange(patch_size)) % map_size
            k_indices = (cz - half_patch + np.arange(patch_size)) % map_size
            
            # Create meshgrid for indexing
            ii, jj, kk = np.meshgrid(i_indices, j_indices, k_indices, indexing='ij')
            
            # Get total weights at these locations
            region_total_weight = total_weight_map[ii, jj, kk]
            normalization_factor = np.maximum(1.0, region_total_weight)
            W_norm = W / normalization_factor
            
            # Add contributions using advanced indexing
            np.add.at(normalized_total_weight, (ii, jj, kk), W_norm)
            np.add.at(normalized_halo_sum, (ii, jj, kk), halo * W_norm)
        
        dmo_contrib = original_map * (1 - normalized_total_weight)
        final_blended_map = dmo_contrib + normalized_halo_sum
        
        return final_blended_map, normalized_total_weight, normalized_halo_sum, dmo_contrib, total_weight_map


class HaloPaster2D:
    """Halo pasting class for 2D overlap handling."""
    
    def __init__(self, box_size_kpc: float, r_in_factor: float = 1.0, r_out_factor: float = 3.5):
        self.box_size_kpc = box_size_kpc
        self.r_in_factor = r_in_factor
        self.r_out_factor = r_out_factor
    
    def _create_weight_function(self, r: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
        W = np.zeros_like(r)
        mask_inner = r <= r_in
        W[mask_inner] = 1.0
        mask_transition = (r > r_in) & (r <= r_out)
        W[mask_transition] = 0.5 * (1 + np.cos(np.pi * (r[mask_transition] - r_in) / (r_out - r_in)))
        return W
    
    def enhanced_halo_pasting_2d(self, original_map: np.ndarray, halo_patches: List, 
                                centers_pixels: List, radii_kpc: List, realization: int = 0) -> Tuple:
        pixel_scale_kpc = self.box_size_kpc / original_map.shape[0]
        map_size = original_map.shape[0]
        
        total_weight_map = np.zeros_like(original_map)
        individual_weights = []
        individual_halos = []
        valid_regions = []
        
        for idx in range(len(halo_patches)):
            halo_patch = halo_patches[idx][realization].sum(axis=0)
            center_pixel = centers_pixels[idx]
            radius_kpc = radii_kpc[idx]
            radius_pixels = radius_kpc / pixel_scale_kpc
            
            cx, cy = int(center_pixel[0]), int(center_pixel[1])
            patch_size = halo_patch.shape[0]
            half_patch = patch_size // 2
            
            r_in = self.r_in_factor * radius_pixels
            max_r_out_from_patch = half_patch - 2
            r_out_unlimited = self.r_out_factor * radius_pixels
            r_out = min(r_out_unlimited, max_r_out_from_patch)
            if r_out <= r_in:
                r_out = r_in + 2.0
            
            # Create mesh grid for radial distance calculation
            x_coords, y_coords = np.meshgrid(
                np.arange(patch_size) - half_patch,
                np.arange(patch_size) - half_patch,
                indexing='ij'
            )
            r_coords = np.sqrt(x_coords**2 + y_coords**2)
            W = self._create_weight_function(r_coords, r_in, r_out)
            
            # Store for this halo
            individual_weights.append(W)
            individual_halos.append(halo_patch)
            valid_regions.append((cx, cy, half_patch))
            
            # Add weights to total weight map with periodic boundaries (vectorized)
            i_indices = (cx - half_patch + np.arange(patch_size)) % map_size
            j_indices = (cy - half_patch + np.arange(patch_size)) % map_size
            
            # Use advanced indexing with meshgrid
            ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')
            np.add.at(total_weight_map, (ii, jj), W)
        
        normalized_total_weight = np.zeros_like(original_map)
        normalized_halo_sum = np.zeros_like(original_map)
        
        # Apply normalized weights and paste halos with periodic boundaries (vectorized)
        for i, (W, halo, (cx, cy, half_patch)) in enumerate(zip(
            individual_weights, individual_halos, valid_regions
        )):
            patch_size = halo.shape[0]
            
            # Generate periodic indices
            i_indices = (cx - half_patch + np.arange(patch_size)) % map_size
            j_indices = (cy - half_patch + np.arange(patch_size)) % map_size
            
            # Create meshgrid for indexing
            ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')
            
            # Get total weights at these locations
            region_total_weight = total_weight_map[ii, jj]
            normalization_factor = np.maximum(1.0, region_total_weight)
            W_norm = W / normalization_factor
            
            # Add contributions using advanced indexing
            np.add.at(normalized_total_weight, (ii, jj), W_norm)
            np.add.at(normalized_halo_sum, (ii, jj), halo * W_norm)
        
        dmo_contrib = original_map * (1 - normalized_total_weight)
        final_blended_map = dmo_contrib + normalized_halo_sum
        
        return final_blended_map, normalized_total_weight, normalized_halo_sum, dmo_contrib, total_weight_map