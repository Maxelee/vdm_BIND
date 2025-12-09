"""
Unified BIND pipeline for all simulation suites (CV, SB35, 1P).
Hierarchical output structure to avoid redundant computation.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from bind.bind import BIND
import pandas as pd
import h5py
import os
import glob
import MAS_library as MASL
import Pk_library as PKL
import argparse
from bind.analyses import run_all_analyses, plot_parameter_extremes_comparison

# Import consolidated I/O utilities
from vdm.io_utils import load_simulation, project_particles_2d as project_2d


def extract_halo_cutout(field_2d, center, size_mpc, box_size, resolution, axis=2):
    """
    Extract 2D projection around center with periodic boundaries.
    Matches BIND._extract_projection_2d method.
    
    Args:
        field_2d: 2D field to extract from
        center: Center position in Mpc/h (3D position)
        size_mpc: Size of region in Mpc
        box_size: Box size in Mpc/h
        resolution: Grid resolution
        axis: Projection axis (0=x, 1=y, 2=z)
    """
    pix_size = box_size / resolution
    half_size_pix = int(size_mpc / (2 * pix_size))
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


def compute_power_spectrum(final_maps, sim_grid, full_hydro, boxsize, output_dir, sim_label, img_base_path):
    """Compute power spectra and create plots."""
    print(f"Computing power spectra for {sim_label}")
    
    box_size = boxsize / 1000.0  # in Mpc/h
    delta_hydro = full_hydro / np.mean(full_hydro, dtype=np.float64); delta_hydro -= 1.0
    delta_dmo = sim_grid / np.mean(sim_grid, dtype=np.float64); delta_dmo -= 1.0
    
    Pk2D_hydro = PKL.Pk_plane(delta_hydro, box_size, 'CIC', 0)
    Pk2D_dmo = PKL.Pk_plane(delta_dmo, box_size, 'CIC', 0)
    
    Pks_bind = []
    for final_map in final_maps:
        delta_bind = final_map / np.mean(final_map, dtype=np.float64); delta_bind -= 1.0
        Pk2D_bind = PKL.Pk_plane(delta_bind, box_size, 'CIC', 0)
        Pks_bind.append(Pk2D_bind.Pk)
    
    k = Pk2D_dmo.k
    ratio_hydro = Pk2D_hydro.Pk / Pk2D_dmo.Pk
    ratio_bind = np.mean(np.array(Pks_bind) / Pk2D_dmo.Pk, axis=0)
    std_bind = np.std(np.array(Pks_bind) / Pk2D_dmo.Pk, axis=0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(k[1:], ratio_hydro[1:], 'r-', label='Hydro/DMO', linewidth=2.5, alpha=0.8)
    plt.semilogx(k[1:], ratio_bind[1:], 'b-', label='Mean BIND/DMO', linewidth=2.5, alpha=0.8)
    plt.fill_between(k[1:], (ratio_bind - std_bind)[1:], (ratio_bind + std_bind)[1:], 
                     alpha=0.3, color='b', label='BIND Std')
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('Power Spectrum Ratio', fontsize=12)
    plt.title(f'Power Spectrum Suppression - {sim_label}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.1, 1.5)
    plt.tight_layout()
    
    os.makedirs(img_base_path, exist_ok=True)
    plt.savefig(os.path.join(img_base_path, f'power_spec_{sim_label}.png'), dpi=150)
    plt.close()
    
    # Save power spectra
    np.savez(os.path.join(output_dir, 'power_spec.npz'), 
             k=k, Pk_dmo=Pk2D_dmo.Pk, Pk_hydro=Pk2D_hydro.Pk, Pk_bind=Pks_bind)


def process_simulation(sim_info, args):
    """
    Process a single simulation through the hierarchical pipeline.
    
    sim_info: dict containing simulation metadata (suite, sim_num, paths, params, etc.)
    args: command line arguments
    """
    suite = sim_info['suite']
    sim_num = sim_info['sim_num']
    simulation_path = sim_info['dmo_path']
    hydro_snapdir = sim_info['hydro_snapdir']
    base_params = sim_info['params']
    om0 = sim_info['om0']
    
    print(f"\n{'='*80}")
    print(f"Processing {suite}_{sim_num}")
    print(f"{'='*80}")
    
    # Define directory structure
    # Note: 1P uses direct sim names (e.g., "1P_p1_0") without "sim_" prefix
    # CV and SB35 use "sim_{num}" format
    if suite == '1P':
        sim_base_dir = os.path.join(args.base_outpath, suite, f"{sim_num}")
    else:
        sim_base_dir = os.path.join(args.base_outpath, suite, f"sim_{sim_num}")
    snap_dir = os.path.join(sim_base_dir, f"snap_{args.snapnum}")
    mt_dir = os.path.join(snap_dir, f"mass_threshold_{int(np.log10(args.mass_threshold))}")
    model_dir = os.path.join(mt_dir, args.model_name)
    paste_dir = os.path.join(model_dir, f"ue_{int(args.use_enhanced)}")
    
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(mt_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(paste_dir, exist_ok=True)
    
    # Step 1: Compute/Load full DMO and Hydro maps (snapshot level)
    sim_grid_path = os.path.join(snap_dir, 'sim_grid.npy')
    full_hydro_path = os.path.join(snap_dir, 'full_hydro.npy')
    hydro_dm_2d_path = os.path.join(snap_dir, 'hydro_dm_2d.npy')
    gas_2d_path = os.path.join(snap_dir, 'gas_2d.npy')
    star_2d_path = os.path.join(snap_dir, 'star_2d.npy')
    
    # Try to load from pre-computed projections first
    projection_loaded = False
    if suite in ['SB35']:
        # For CV and SB35, load from projected_images directory
        proj_dir = '/mnt/home/mlee1/ceph/train_data_1024/projected_images'
        if suite == 'CV':
            proj_file = os.path.join(proj_dir, f'projections_xy_{sim_num}.npz')
        else:  # SB35
            proj_file = os.path.join(proj_dir, f'projections_xy_{sim_num}.npz')
        
        if os.path.exists(proj_file):
            print(f"Loading from pre-computed projections: {proj_file}")
            proj_data = np.load(proj_file)
            sim_grid = proj_data['dm'].astype(np.float32)
            hydro_dm_2d = proj_data['hydro_dm'].astype(np.float32)
            gas_2d = proj_data['gas'].astype(np.float32)
            star_2d = proj_data['star'].astype(np.float32)
            full_hydro = hydro_dm_2d + gas_2d + star_2d
            projection_loaded = True
    elif suite == '1P':
        # For 1P, load from projected_images_1P directory
        proj_dir = '/mnt/home/mlee1/ceph/train_data_1024/projected_images_1P'
        proj_file = os.path.join(proj_dir, f'projections_xy_{sim_num}.npz')  # FIX: sim_num already has '1P_' prefix
        
        if os.path.exists(proj_file):
            print(f"Loading from pre-computed projections: {proj_file}")
            proj_data = np.load(proj_file)
            sim_grid = proj_data['dm'].astype(np.float32)
            hydro_dm_2d = proj_data['hydro_dm'].astype(np.float32)
            gas_2d = proj_data['gas'].astype(np.float32)
            star_2d = proj_data['star'].astype(np.float32)
            full_hydro = hydro_dm_2d + gas_2d + star_2d
            projection_loaded = True
    
    # If pre-computed projections not found, check if already processed and cached
    if not projection_loaded and (os.path.exists(sim_grid_path) and os.path.exists(full_hydro_path) 
        and os.path.exists(hydro_dm_2d_path) and os.path.exists(gas_2d_path) 
        and os.path.exists(star_2d_path) and not args.regenerate_all):
        print(f"Loading existing full maps from cache: {snap_dir}")
        sim_grid = np.load(sim_grid_path)
        full_hydro = np.load(full_hydro_path)
        hydro_dm_2d = np.load(hydro_dm_2d_path)
        gas_2d = np.load(gas_2d_path)
        star_2d = np.load(star_2d_path)
        projection_loaded = True
    
    # Last resort: compute from scratch
    if not projection_loaded:
        print(f"Computing full maps for {suite}_{sim_num} (pre-computed projections not found)")
        dm_pos, dm_mass, hydro_dm_pos, hydro_dm_mass, gas_pos, gas_mass, star_pos, star_mass = load_simulation(
            simulation_path, hydro_snapdir
        )
        
        box_size = args.boxsize / 1000.0  # Convert to Mpc/h
        sim_grid = project_2d(dm_pos, dm_mass, box_size, args.gridsize, axis=args.axis)
        hydro_dm_2d = project_2d(hydro_dm_pos, hydro_dm_mass, box_size, args.gridsize, axis=args.axis) if len(hydro_dm_pos) > 0 else np.zeros((args.gridsize, args.gridsize), dtype=np.float32)
        gas_2d = project_2d(gas_pos, gas_mass, box_size, args.gridsize, axis=args.axis) if len(gas_pos) > 0 else np.zeros((args.gridsize, args.gridsize), dtype=np.float32)
        star_2d = project_2d(star_pos, star_mass, box_size, args.gridsize, axis=args.axis) if len(star_pos) > 0 else np.zeros((args.gridsize, args.gridsize), dtype=np.float32)
        full_hydro = hydro_dm_2d + gas_2d + star_2d
        
        # Clean up large particle data from memory
        del dm_pos, dm_mass, hydro_dm_pos, hydro_dm_mass, gas_pos, gas_mass, star_pos, star_mass
        import gc
        gc.collect()
    
    # Save to cache for next time
    np.save(sim_grid_path, sim_grid)
    np.save(full_hydro_path, full_hydro)
    np.save(hydro_dm_2d_path, hydro_dm_2d)
    np.save(gas_2d_path, gas_2d)
    np.save(star_2d_path, star_2d)
    print(f"Saved full maps and component maps to {snap_dir}")
    
    # Step 2: Extract/Load halo metadata and raw cutouts (mass threshold level)
    # Raw cutouts (same for all models) are saved here
    # Normalized cutouts (model-specific conditions) are created later as temporary halo_*.npz files
    halo_metadata_path = os.path.join(mt_dir, 'halo_metadata.npz')
    dmo_cutouts_path = os.path.join(mt_dir, 'dmo_cutouts.npy')
    hydro_cutouts_path = os.path.join(mt_dir, 'hydro_cutouts.npy')
    
    if (os.path.exists(halo_metadata_path) and os.path.exists(dmo_cutouts_path) 
        and os.path.exists(hydro_cutouts_path) and not args.regenerate_all):
        print(f"Loading existing halo metadata and raw cutouts for {suite}_{sim_num}")
        metadata_data = np.load(halo_metadata_path)
        metadata = {
            'masses': metadata_data['masses'],
            'radii': metadata_data['radii'],
            'positions': metadata_data['positions'],
            'center_pixels': metadata_data['center_pixels']
        }
        dmo_cutouts = np.load(dmo_cutouts_path)
        hydro_cutouts = np.load(hydro_cutouts_path)
    else:
        print(f"Extracting halo metadata and raw cutouts for {suite}_{sim_num}")
        # Initialize BIND to extract halo catalog
        bind = BIND(
            simulation_path=simulation_path,
            snapnum=args.snapnum,
            boxsize=args.boxsize,
            gridsize=args.gridsize,
            config_path=args.config_path,
            output_dir=model_dir,  # Temporary
            verbose=False,
            dim='2d',
            axis=args.axis,
            r_in_factor=args.r_in_factor,
            r_out_factor=args.r_out_factor,
            mass_threshold=args.mass_threshold
        )
        
        bind.sim_grid = sim_grid  # Use precomputed grid
        bind.load_halo_catalog()
        
        # Save the metadata
        metadata = {
            'masses': bind.halo_catalog['masses'],
            'radii': bind.halo_catalog['radii'],
            'positions': bind.halo_catalog['positions'],
            'indices': bind.halo_catalog['indices'],
            'center_pixels': None  # Will be computed during extraction
        }
        
        # Compute center pixels
        halo_positions_mpc = bind.halo_catalog['positions'] / 1000.0
        box_size_mpc = args.boxsize / 1000.0
        pix_size = box_size_mpc / args.gridsize
        axes = [0, 1, 2]
        proj_axes = axes[:args.axis] + axes[args.axis+1:]
        center_pixels = (halo_positions_mpc[:, proj_axes] / pix_size).astype(int)
        metadata['center_pixels'] = center_pixels
        
        # Extract RAW cutouts (not normalized - same for all models)
        # For hydro, extract DM, Gas, Stars separately (3 channels)
        print(f"Extracting {len(metadata['masses'])} raw DMO and hydro cutouts...")
        dmo_cutouts = []
        hydro_cutouts = []
        cutout_size_mpc = 6.25  # Match BIND's extraction size
        for pos in halo_positions_mpc:
            dmo_cut = extract_halo_cutout(sim_grid, pos, cutout_size_mpc, box_size_mpc, 
                                         args.gridsize, axis=args.axis)
            # Extract 3-channel hydro cutout: [DM, Gas, Stars]
            hydro_dm_cut = extract_halo_cutout(hydro_dm_2d, pos, cutout_size_mpc, box_size_mpc,
                                              args.gridsize, axis=args.axis)
            gas_cut = extract_halo_cutout(gas_2d, pos, cutout_size_mpc, box_size_mpc,
                                          args.gridsize, axis=args.axis)
            star_cut = extract_halo_cutout(star_2d, pos, cutout_size_mpc, box_size_mpc,
                                           args.gridsize, axis=args.axis)
            hydro_cut = np.stack([hydro_dm_cut, gas_cut, star_cut], axis=0)  # Shape: (3, L, W)
            
            dmo_cutouts.append(dmo_cut)
            hydro_cutouts.append(hydro_cut)
        
        dmo_cutouts = np.array(dmo_cutouts)
        hydro_cutouts = np.array(hydro_cutouts)  # Shape: (n_halos, 3, L, W)
        
        # Save everything
        np.savez(halo_metadata_path, 
                 masses=metadata['masses'],
                 radii=metadata['radii'],
                 positions=metadata['positions'],
                 center_pixels=metadata['center_pixels'])
        np.save(dmo_cutouts_path, dmo_cutouts)
        np.save(hydro_cutouts_path, hydro_cutouts)
        print(f"Saved halo metadata and raw cutouts to {mt_dir}")
        print(f"  DMO cutouts shape: {dmo_cutouts.shape}")
        print(f"  Hydro cutouts shape: {hydro_cutouts.shape} [n_halos, 3 (DM/Gas/Stars), L, W]")
    
    # Early exit if prep_only mode - just save maps and cutouts, no BIND generation
    if args.prep_only:
        print(f"\n[PREP ONLY] Finished preparing data for {suite}_{sim_num}")
        print(f"  Full maps saved to: {snap_dir}")
        print(f"  Halo cutouts saved to: {mt_dir}")
        return True
    
    # Step 3: Hydro replacement (mass threshold level)
    if args.do_hydro_replace:
        hydro_replaced_path = os.path.join(mt_dir, 'hydro_replaced.npy')
        
        # Try to load from pre-existing hydro replacement files first
        hydro_replaced_loaded = False
        if suite == 'CV':
            hydro_replaced_source = f'/mnt/home/mlee1/ceph/BIND2d/hydro_replace/CV/sim_{sim_num}/hydro_replace/final_map_hydro_replace.npy'
        elif suite == 'SB35':
            hydro_replaced_source = f'/mnt/home/mlee1/ceph/BIND2d/hydro_replace/SB35/sim_{sim_num}/hydro_replace/final_map_hydro_replace.npy'
        elif suite == '1P':
            # For 1P, sim_num already includes "1P_" prefix (e.g., "1P_p1_0")
            hydro_replaced_source = f'/mnt/home/mlee1/ceph/BIND2d/hydro_replace/1P/{sim_num}/hydro_replace/final_map_hydro_replace.npy'
        else:
            hydro_replaced_source = None
        
        # Priority 1: Load from cache
        if os.path.exists(hydro_replaced_path) and not args.regenerate_all:
            print(f"Loading existing hydro replacement from cache: {hydro_replaced_path}")
            hydro_replaced = np.load(hydro_replaced_path)
            hydro_replaced_loaded = True
        # Priority 2: Load from pre-computed BIND2d files
        elif hydro_replaced_source and os.path.exists(hydro_replaced_source) and not args.regenerate_all:
            print(f"Loading existing hydro replacement from: {hydro_replaced_source}")
            hydro_replaced = np.load(hydro_replaced_source)
            # Save to cache for next time
            np.save(hydro_replaced_path, hydro_replaced)
            print(f"Cached to {hydro_replaced_path}")
            hydro_replaced_loaded = True
        
        # Priority 3: Compute from scratch
        if not hydro_replaced_loaded:
            print(f"Computing hydro replacement for {suite}_{sim_num}")
            # Use the raw hydro cutouts we already extracted
            bind = BIND(
                simulation_path=simulation_path,
                snapnum=args.snapnum,
                boxsize=args.boxsize,
                gridsize=args.gridsize,
                config_path=args.config_path,
                output_dir=model_dir,  # Use model dir for temp halo files
                verbose=False,
                dim='2d',
                axis=args.axis,
                r_in_factor=args.r_in_factor,
                r_out_factor=args.r_out_factor,
                mass_threshold=args.mass_threshold
            )
            
            bind.sim_grid = sim_grid
            
            # Set up metadata for pasting
            bind.extracted = {
                'metadata': [
                    {
                        'mass': metadata['masses'][i],
                        'radius': metadata['radii'][i],
                        'center_pixel': metadata['center_pixels'][i],
                    }
                    for i in range(len(metadata['masses']))
                ]
            }
            
            # Use raw hydro cutouts as "generated" halos (sum all 3 components)
            # Shape: (n_halos, 1, 1, H, W) where we sum DM+Gas+Stars
            bind.generated_images = hydro_cutouts.sum(axis=1, keepdims=True)[:, None, :, :]
            hydro_replaced_maps = bind.paste_halos(realizations=1, use_enhanced=True)
            hydro_replaced = hydro_replaced_maps[0]
            
            np.save(hydro_replaced_path, hydro_replaced)
            print(f"Saved hydro replacement to {hydro_replaced_path}")
    
    # Step 4: Generate halos with model (model level)
    # This step creates halo_*.npz files (with model-specific conditional params)
    # in the model directory, then generates halos and cleans them up
    generated_path = os.path.join(model_dir, 'generated_halos.npz')
    
    if os.path.exists(generated_path) and not args.regenerate:
        print(f"Loading existing generated halos for {suite}_{sim_num}")
        loaded = np.load(generated_path)
        generated_halos = loaded['generated']
    else:
        print(f"Generating halos for {suite}_{sim_num}")
        bind = BIND(
            simulation_path=simulation_path,
            snapnum=args.snapnum,
            boxsize=args.boxsize,
            gridsize=args.gridsize,
            config_path=args.config_path,
            output_dir=model_dir,  # Model-specific halo_*.npz files go here
            verbose=False,
            dim='2d',
            axis=args.axis,
            r_in_factor=args.r_in_factor,
            r_out_factor=args.r_out_factor,
            mass_threshold=args.mass_threshold
        )
        
        bind.sim_grid = sim_grid
        
        # Extract halos with model-specific conditional parameters
        # This saves halo_*.npz files in model_dir
        bind.extract_halos(omega_m=om0)
        
        num_halos = len(metadata['masses'])
        conditional_params = np.reshape(base_params * num_halos, (num_halos, -1))
        
        # Generate halos and clean up halo_*.npz files
        bind.generate_halos(batch_size=args.batch_size, conditional_params=conditional_params,
                           cleanup_conditions=True)
        generated_halos = np.array(bind.generated_images)
        
        # Save generated halos with metadata
        np.savez(generated_path, generated=generated_halos, **metadata)
        print(f"Saved generated halos to {generated_path}")
        
        # Explicit cleanup after generation
        del bind
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Step 5: Paste halos and compute power spectra (paste method level)
    final_map_0_path = os.path.join(paste_dir, 'final_map_0.npy')
    
    if os.path.exists(final_map_0_path) and not args.repaste:
        print(f"Loading existing final maps for {suite}_{sim_num}")
        final_maps = [np.load(os.path.join(paste_dir, f'final_map_{i}.npy')) 
                      for i in range(args.realizations)]
    else:
        print(f"Pasting halos for {suite}_{sim_num}")
        bind = BIND(
            simulation_path=simulation_path,
            snapnum=args.snapnum,
            boxsize=args.boxsize,
            gridsize=args.gridsize,
            config_path=args.config_path,
            output_dir=paste_dir,
            verbose=False,
            dim='2d',
            axis=args.axis,
            r_in_factor=args.r_in_factor,
            r_out_factor=args.r_out_factor,
            mass_threshold=args.mass_threshold
        )
        
        bind.sim_grid = sim_grid
        bind.extracted = {
            'metadata': [
                {
                    'mass': metadata['masses'][i],
                    'radius': metadata['radii'][i],
                    'center_pixel': metadata['center_pixels'][i],
                }
                for i in range(len(metadata['masses']))
            ]
        }
        bind.generated_images = generated_halos
        
        final_maps = bind.paste_halos(realizations=args.realizations, use_enhanced=args.use_enhanced)
        
        for i, fm in enumerate(final_maps):
            np.save(os.path.join(paste_dir, f'final_map_{i}.npy'), fm)
        print(f"Saved final maps to {paste_dir}")
    
    # Step 6: Compute power spectra
    if not os.path.exists(os.path.join(paste_dir, 'power_spec.npz')) or args.regenerate or args.repaste:
        img_path = os.path.join('/mnt/home/mlee1/BIND3d/imgs_new', args.model_name, suite)
        compute_power_spectrum(final_maps, sim_grid, full_hydro, args.boxsize, 
                             paste_dir, f"{suite}_{sim_num}", img_path)
    
    # Step 7: Run comprehensive analyses (if requested)
    if args.run_analyses:
        img_base_dir = os.path.join('/mnt/home/mlee1/BIND3d/paper_plots', args.model_name, suite, f'sim_{sim_num}')
        
        # Load hydro_replaced if not already loaded
        if 'hydro_replaced' not in locals():
            # Try standard cache location first
            hydro_replaced_path = os.path.join(mt_dir, 'hydro_replaced.npy')
            
            # Construct suite-specific alternate path
            if suite == 'CV':
                hydro_replaced_alt_path = f'/mnt/home/mlee1/ceph/BIND2d/hydro_replace/CV/sim_{sim_num}/hydro_replace/final_map_hydro_replace.npy'
            elif suite == 'SB35':
                hydro_replaced_alt_path = f'/mnt/home/mlee1/ceph/BIND2d/hydro_replace/SB35/sim_{sim_num}/hydro_replace/final_map_hydro_replace.npy'
            elif suite == '1P':
                # For 1P, sim_num already includes "1P_" prefix (e.g., "1P_p1_0")
                hydro_replaced_alt_path = f'/mnt/home/mlee1/ceph/BIND2d/hydro_replace/1P/{sim_num}/hydro_replace/final_map_hydro_replace.npy'
            else:
                hydro_replaced_alt_path = None
            
            if os.path.exists(hydro_replaced_path):
                print(f"Loading hydro_replaced from {hydro_replaced_path}")
                hydro_replaced = np.load(hydro_replaced_path)
            elif hydro_replaced_alt_path and os.path.exists(hydro_replaced_alt_path):
                print(f"Loading hydro_replaced from {hydro_replaced_alt_path}")
                hydro_replaced = np.load(hydro_replaced_alt_path)
            else:
                print(f"Warning: hydro_replaced not found at {hydro_replaced_path} or {hydro_replaced_alt_path}")
                print(f"Using full_hydro as fallback")
                hydro_replaced = full_hydro
        
        # Convert radii from kpc to pixels
        halo_radii = metadata['radii'] * args.gridsize / (args.boxsize / 1000.0) / 1000.0
        halo_radii_list = list(halo_radii)
        
        # Call comprehensive analysis function matching BIND_usage_CV.ipynb
        run_all_analyses(
            full_dmo=sim_grid,
            full_hydro=full_hydro, 
            hydro_replaced=hydro_replaced,
            final_maps=final_maps,
            dmo_cutouts=dmo_cutouts,
            hydro_cutouts=hydro_cutouts,
            gen_cutouts=generated_halos,
            halo_radii=halo_radii_list,
            boxsize=args.boxsize / 1000.0,  # Convert to Mpc/h
            output_dir=img_base_dir,
            model_name=args.model_name
        )
    
    print(f"Completed {suite}_{sim_num}")
    return True


def get_cv_simulation_info(sim_num, args):
    """Get simulation info for CV suite."""
    metadata = pd.read_csv('/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv')
    base_params = list(metadata['FiducialVal'])
    om0 = base_params[0]
    
    base_simpath = '/mnt/ceph/users/camels/Sims/IllustrisTNG_DM/L50n512/'
    dmo_path = f'{base_simpath}/CV/CV_{sim_num}'
    hydro_path = dmo_path.replace('IllustrisTNG_DM', 'IllustrisTNG')
    hydro_snapdir = os.path.join(hydro_path, 'snapdir_090')
    
    return {
        'suite': 'CV',
        'sim_num': sim_num,
        'dmo_path': dmo_path,
        'hydro_snapdir': hydro_snapdir,
        'params': base_params,
        'om0': om0
    }


def get_sb35_simulation_info(sim_num, args):
    """Get simulation info for SB35 suite."""
    metadata = pd.read_csv('/mnt/home/mlee1/50Mpc_boxes/data/param_df.csv')
    base_params = list(metadata.iloc[sim_num])
    om0 = base_params[0]
    
    base_simpath = '/mnt/ceph/users/camels/Sims/IllustrisTNG_DM/L50n512/'
    dmo_path = f'{base_simpath}/SB35/SB35_{sim_num}'
    hydro_path = dmo_path.replace('IllustrisTNG_DM', 'IllustrisTNG')
    hydro_snapdir = os.path.join(hydro_path, 'snapdir_090')
    
    return {
        'suite': 'SB35',
        'sim_num': sim_num,
        'dmo_path': dmo_path,
        'hydro_snapdir': hydro_snapdir,
        'params': base_params,
        'om0': om0
    }


def get_1p_simulation_info(sim_name, args):
    """Get simulation info for 1P suite."""
    param_file = '/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt'
    oneP_params = pd.read_csv(param_file, delim_whitespace=True)
    
    base_params = list(oneP_params[oneP_params['#Name'] == sim_name].iloc[0, 1:-1])
    if sim_name.split('_')[1] != 'p15':
        base_params = np.array(base_params)
        base_params[14] = 0
        base_params = list(base_params)
    om0 = base_params[0]
    
    # Determine which N-body simulation to use
    param_val = sim_name.split('_')[-1]
    try:
        param_val = int(param_val)
        fiducial = (param_val == 0)
    except ValueError:
        fiducial = False
    
    cosmological_param_ids = {1, 2, 7, 8, 9}
    
    if fiducial:
        nbody_sim = '1P_p1_0'
    else:
        parts = sim_name.split('_')
        param_id = int(parts[1][1:])
        if param_id in cosmological_param_ids:
            nbody_sim = sim_name
        else:
            nbody_sim = '1P_p1_0'
    
    nbody_base = '/mnt/ceph/users/camels/Sims/IllustrisTNG_DM/L50n512/1P'
    hydro_base = '/mnt/ceph/users/camels/Sims/IllustrisTNG/L50n512/1P'
    
    dmo_path = f'{nbody_base}/{nbody_sim}'
    hydro_snapdir = f'{hydro_base}/{sim_name}/snapdir_090'
    
    return {
        'suite': '1P',
        'sim_num': sim_name,
        'dmo_path': dmo_path,
        'hydro_snapdir': hydro_snapdir,
        'params': base_params,
        'om0': om0
    }


def main():
    parser = argparse.ArgumentParser(
        description='Unified BIND pipeline for all simulation suites'
    )
    
    # Suite selection
    parser.add_argument('--suite', type=str, default='all', 
                       choices=['all', 'cv', 'sb35', '1p'],
                       help='Which simulation suite to process (default: all)')
    parser.add_argument('--sim_nums', type=str, default=None,
                       help='Comma-separated list of simulation numbers to process (default: all)')
    
    # Simulation parameters
    parser.add_argument('--snapnum', type=int, default=90, 
                       help='Snapshot number (default: 90)')
    parser.add_argument('--boxsize', type=float, default=50000.0, 
                       help='Box size in kpc/h (default: 50000.0)')
    parser.add_argument('--gridsize', type=int, default=1024, 
                       help='Grid size (default: 1024)')
    parser.add_argument('--axis', type=int, default=2, choices=[0, 1, 2],
                       help='Projection axis (default: 2 for z-axis)')
    
    # Halo parameters
    parser.add_argument('--mass_threshold', type=float, default=1e13, 
                       help='Mass threshold in Msun/h (default: 1e13)')
    parser.add_argument('--r_in_factor', type=float, default=2.5,
                       help='Inner radius factor (default: 2.5)')
    parser.add_argument('--r_out_factor', type=float, default=3.0,
                       help='Outer radius factor (default: 3.0)')

    # Model parameters
    parser.add_argument('--config_path', type=str, 
                       default='/mnt/home/mlee1/vdm_BIND/configs/dmo2hydro_1024_2.ini',
                       help='Path to model config file')
    parser.add_argument('--model_name', type=str, default='standard',
                       help='Model name for output directory (default: standard)')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for generation (reduce if OOM errors occur, default: 10)')
    
    # Pasting parameters
    parser.add_argument('--use_enhanced', action='store_const', const=True, default=True, 
                       help='Use enhanced halo pasting (default: True)')
    parser.add_argument('--no-use_enhanced', action='store_const', const=False, dest='use_enhanced',
                       help='Disable enhanced halo pasting')
    parser.add_argument('--realizations', type=int, default=10,
                       help='Number of paste realizations (default: 10)')
    
    # Output paths
    parser.add_argument('--base_outpath', type=str, 
                       default='/mnt/home/mlee1/ceph/BIND2d',
                       help='Base output path (default: /mnt/home/mlee1/ceph/BIND2d)')
    
    # Control flags
    parser.add_argument('--regenerate', action='store_true', 
                       help='Force regeneration of model outputs (generated halos, final maps, power spectra)')
    parser.add_argument('--regenerate_all', action='store_true',
                       help='Force regeneration of ALL data including DMO/hydro grids and cutouts')
    parser.add_argument('--repaste', action='store_true', 
                       help='Force repasting even if final maps exist')
    parser.add_argument('--do_hydro_replace', action='store_true',
                       help='Also compute hydro replacement baseline')
    parser.add_argument('--run_analyses', action='store_true',
                       help='Run additional analyses (profiles, mass comparisons)')
    parser.add_argument('--run_1p_extremes', action='store_true',
                       help='Generate parameter extremes comparison plot for 1P suite')
    parser.add_argument('--prep_only', action='store_true',
                       help='Only prepare data (full maps, cutouts, metadata) without running BIND generation')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("UNIFIED BIND PIPELINE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Suite: {args.suite}")
    print(f"  Model: {args.model_name}")
    print(f"  Grid size: {args.gridsize}")
    print(f"  Mass threshold: {args.mass_threshold:.1e}")
    print(f"  Enhanced pasting: {args.use_enhanced}")
    print(f"  Realizations: {args.realizations}")
    print(f"  Output path: {args.base_outpath}")
    print(f"  Hydro replacement: {args.do_hydro_replace}")
    print(f"  Run analyses: {args.run_analyses}")
    if args.prep_only:
        print(f"  PREP ONLY MODE: Generating maps and cutouts only (no BIND generation)")
    if args.run_1p_extremes:
        print(f"  1P extremes plot: Yes")
    if args.regenerate:
        print(f"  Regenerate mode: Model outputs only (keeps grids/cutouts)")
    if args.regenerate_all:
        print(f"  Regenerate mode: ALL data (grids, cutouts, model outputs)")
    print("=" * 80)
    
    # Determine which simulations to process
    sim_infos = []
    
    if args.suite in ['all', 'cv']:
        if args.sim_nums:
            sim_nums = [int(x) for x in args.sim_nums.split(',')]
        else:
            sim_nums = range(25)
        
        for sim_num in sim_nums:
            sim_infos.append(get_cv_simulation_info(sim_num, args))
    
    if args.suite in ['all', 'sb35']:
        if args.sim_nums:
            sim_nums = [int(x) for x in args.sim_nums.split(',')]
        else:
            # Get all SB35 test simulations
            test_dirs = sorted(glob.glob('/mnt/home/mlee1/ceph/train_data_1024/test_2d/*'))
            sim_nums = [int(t.split('_')[-1]) for t in test_dirs]
        
        for sim_num in sim_nums:
            sim_infos.append(get_sb35_simulation_info(sim_num, args))
    
    if args.suite in ['all', '1p']:
        param_file = '/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt'
        oneP_params = pd.read_csv(param_file, delim_whitespace=True)
        names = oneP_params['#Name'].to_list()
        
        if args.sim_nums:
            # Assume comma-separated sim names for 1P
            sim_names = args.sim_nums.split(',')
        else:
            sim_names = names
        
        for sim_name in sim_names:
            sim_infos.append(get_1p_simulation_info(sim_name, args))
    
    # Process all simulations
    print(f"\nProcessing {len(sim_infos)} simulations...")
    
    # Track which 1P simulations were successfully processed (for extremes plot)
    successful_1p_sims = []
    
    for sim_info in sim_infos:
        try:
            success = process_simulation(sim_info, args)
            
            # Track successful 1P simulations for extremes plot (don't load data yet)
            if args.run_1p_extremes and sim_info['suite'] == '1P' and success:
                successful_1p_sims.append(sim_info)
            
            # Explicit memory cleanup after each simulation
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"ERROR processing {sim_info['suite']}_{sim_info['sim_num']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate 1P parameter extremes comparison if requested
    # Load data from disk only when needed (memory efficient)
    if args.run_1p_extremes and len(successful_1p_sims) > 0:
        print("\nGenerating 1P parameter extremes comparison plot...")
        print("Loading data from disk for extremes plot...")
        
        generated_halos_dict = {}
        hydro_cutouts_dict = {}
        params_dict = {}
        
        for sim_info in successful_1p_sims:
            sim_name = sim_info['sim_num']
            
            mt_dir = os.path.join(args.base_outpath, sim_info['suite'], f"sim_{sim_name}",
                                 f"snap_{args.snapnum}", f"mass_threshold_{int(np.log10(args.mass_threshold))}")
            model_dir = os.path.join(mt_dir, args.model_name)
            
            generated_path = os.path.join(model_dir, 'generated_halos.npz')
            hydro_cutouts_path = os.path.join(mt_dir, 'hydro_cutouts.npy')
            
            if os.path.exists(generated_path) and os.path.exists(hydro_cutouts_path):
                loaded = np.load(generated_path)
                generated_halos_dict[sim_name] = loaded['generated']
                hydro_cutouts_dict[sim_name] = np.load(hydro_cutouts_path)
                params_dict[sim_name] = sim_info['params']
        
        img_dir = os.path.join('/mnt/home/mlee1/BIND3d/imgs_new', args.model_name, '1P', 'analyses')
        plot_parameter_extremes_comparison(successful_1p_sims, generated_halos_dict, hydro_cutouts_dict,
                                          params_dict, img_dir, args.model_name)
    
    print("\n" + "=" * 80)
    print("All simulations processed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
