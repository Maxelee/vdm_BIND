"""
Process halos with mass between 1e12 and 1e13 Msun.

This is a follow-up to process_simulations2_cpu.py which processed halos > 1e13.
It reuses the same functions but:
  - Only processes halos with 1e12 < M <= 1e13
  - Uses 1 rotation per halo (not 10)
  - Saves with 'lowmass_halo_' prefix to avoid filename conflicts

Output goes into the SAME train/test directories so the training pipeline
can pick up all .npz files seamlessly.
"""

import numpy as np
import h5py
import os
from scipy.spatial.transform import Rotation
import MAS_library as MASL
import pandas as pd
import argparse
import random
import mpi4py.MPI as MPI

# Command-line arguments
parser = argparse.ArgumentParser(description='Process low-mass halos (1e12 < M <= 1e13) from IllustrisTNG.')
parser.add_argument('--resolution', type=int, default=128)
parser.add_argument('--total_sims', type=int, default=1024)
parser.add_argument('--start_sim', type=int, default=0, help='Starting simulation index')
parser.add_argument('--end_sim', type=int, default=None, help='Ending simulation index (exclusive)')
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1993)
parser.add_argument('--output_base_root', type=str, default='/mnt/home/mlee1/ceph')
parser.add_argument('--hydro_base', type=str, default='/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35')
parser.add_argument('--nbody_base', type=str, default='/mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35')
parser.add_argument('--fof_nbody_base', type=str, default='/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/SB35')
parser.add_argument('--param_file', type=str, default='/mnt/home/mlee1/50Mpc_boxes/data/param_df.csv')
parser.add_argument('--num_rotations', type=int, default=1,
                    help='Number of rotations per halo (default: 1)')
parser.add_argument('--mass_low', type=float, default=1e12,
                    help='Lower mass threshold in Msun (default: 1e12)')
parser.add_argument('--mass_high', type=float, default=1e13,
                    help='Upper mass threshold in Msun - halos above this are skipped (default: 1e13)')

args = parser.parse_args()

# Load parameters
metadata = pd.read_csv(args.param_file)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set variables
resolution = args.resolution
total_sims = args.total_sims
test_size = int(args.test_frac * total_sims)
train_size = total_sims - test_size
output_base_root = args.output_base_root
hydro_base = args.hydro_base
nbody_base = args.nbody_base
fof_nbody_base = args.fof_nbody_base
num_rotations = args.num_rotations
mass_low = args.mass_low
mass_high = args.mass_high
BOX_SIZE = 50.0  # Mpc/h

# Determine sim range
start_sim = args.start_sim
end_sim = args.end_sim if args.end_sim is not None else total_sims

# Train/test split (same seed as original run to get identical split)
random.seed(args.seed)
all_sims = list(range(total_sims))
random.shuffle(all_sims)
test_sims = set(all_sims[:test_size])
train_sims = set(all_sims[test_size:])

# ========== Data loading functions (identical to process_simulations2_cpu.py) ==========

def sim_dir(i):
    return f'SB35_{i}'

def load_params(sim_id):
    return metadata.iloc[sim_id].to_dict()

def load_simulation(nbody_path, hydro_snapdir):
    """Load particle data from nbody and hydro simulations."""
    dm_pos = []
    dm_mass = []
    with h5py.File(os.path.join(nbody_path, 'snap_090.hdf5'), 'r') as f:
        dm_pos.append(f['PartType1/Coordinates'][:])
        mass_table = f['Header'].attrs['MassTable']
        dm_particle_mass = mass_table[1]
        num_dm = len(f['PartType1/Coordinates'][:])
        dm_mass.append(np.full(num_dm, dm_particle_mass))
    dm_pos = np.concatenate(dm_pos)
    dm_mass = np.concatenate(dm_mass)
    dm_pos /= 1000.0
    dm_mass *= 1e10

    hydro_dm_pos, hydro_dm_mass = [], []
    gas_pos, gas_mass = [], []
    star_pos, star_mass = [], []

    for i in range(16):
        fname = os.path.join(hydro_snapdir, f'snap_090.{i}.hdf5')
        if not os.path.exists(fname):
            continue
        with h5py.File(fname, 'r') as f:
            if 'PartType1/Coordinates' in f:
                hydro_dm_pos.append(f['PartType1/Coordinates'][:])
                if 'PartType1/Masses' in f:
                    hydro_dm_mass.append(f['PartType1/Masses'][:])
                else:
                    mass_table = f['Header'].attrs['MassTable']
                    dm_particle_mass = mass_table[1]
                    num_dm = len(f['PartType1/Coordinates'][:])
                    hydro_dm_mass.append(np.full(num_dm, dm_particle_mass))
            if 'PartType0/Coordinates' in f:
                gas_pos.append(f['PartType0/Coordinates'][:])
                gas_mass.append(f['PartType0/Masses'][:])
            if 'PartType4/Coordinates' in f:
                star_pos.append(f['PartType4/Coordinates'][:])
                star_mass.append(f['PartType4/Masses'][:])

    hydro_dm_pos = np.concatenate(hydro_dm_pos) if hydro_dm_pos else np.array([])
    hydro_dm_mass = np.concatenate(hydro_dm_mass) if hydro_dm_mass else np.array([])
    gas_pos = np.concatenate(gas_pos) if gas_pos else np.array([])
    gas_mass = np.concatenate(gas_mass) if gas_mass else np.array([])
    star_pos = np.concatenate(star_pos) if star_pos else np.array([])
    star_mass = np.concatenate(star_mass) if star_mass else np.array([])

    if len(hydro_dm_pos) > 0:
        hydro_dm_pos /= 1000.0
        hydro_dm_mass *= 1e10
    if len(gas_pos) > 0:
        gas_pos /= 1000.0
        gas_mass *= 1e10
    if len(star_pos) > 0:
        star_pos /= 1000.0
        star_mass *= 1e10

    return dm_pos, dm_mass, hydro_dm_pos, hydro_dm_mass, gas_pos, gas_mass, star_pos, star_mass

def load_halos_in_mass_range(fof_file, mass_low=1e12, mass_high=1e13):
    """
    Load halos with mass_low < M <= mass_high from FOF catalog.
    
    Returns positions, masses, and the index of each halo within the
    low-mass-only filtered array (for unique file naming).
    """
    if not os.path.exists(fof_file):
        return np.array([]), np.array([]), np.array([], dtype=int)
    
    with h5py.File(fof_file, 'r') as f:
        if 'Group/GroupPos' not in f or 'Group/Group_M_Crit200' not in f:
            return np.array([]), np.array([]), np.array([], dtype=int)
        halo_pos = f['Group/GroupPos'][:]
        halo_mass = f['Group/Group_M_Crit200'][:]
    
    halo_pos = halo_pos / 1000.0   # kpc/h -> Mpc/h
    halo_mass = halo_mass * 1e10    # 1e10 Msun -> Msun
    
    # Select halos in the target mass range: mass_low < M <= mass_high
    mask = (halo_mass > mass_low) & (halo_mass <= mass_high)
    
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([], dtype=int)
    
    selected_pos = halo_pos[mask]
    selected_mass = halo_mass[mask]
    # Sequential indices within this selection (0, 1, 2, ...)
    indices = np.arange(np.sum(mask))
    
    return selected_pos, selected_mass, indices


# ========== Processing functions (identical to process_simulations2_cpu.py) ==========

def apply_periodic_boundary_minimum_image(positions, halo_center, box_size=50.0):
    delta = positions - halo_center
    delta = delta - box_size * np.round(delta / box_size)
    return delta

def create_periodic_copies_for_rotation(positions, masses, box_size=50.0, buffer=5.0):
    all_positions = [positions]
    all_masses = [masses]
    half_box = box_size / 2.0
    edge_threshold = half_box - buffer

    for axis in range(3):
        near_pos_edge = positions[:, axis] > edge_threshold
        if np.any(near_pos_edge):
            copied_pos = positions[near_pos_edge].copy()
            copied_pos[:, axis] -= box_size
            all_positions.append(copied_pos)
            all_masses.append(masses[near_pos_edge])

        near_neg_edge = positions[:, axis] < -edge_threshold
        if np.any(near_neg_edge):
            copied_pos = positions[near_neg_edge].copy()
            copied_pos[:, axis] += box_size
            all_positions.append(copied_pos)
            all_masses.append(masses[near_neg_edge])

    all_positions = np.vstack(all_positions)
    all_masses = np.concatenate(all_masses)
    return all_positions, all_masses

def pixelize_z_projection(positions, masses, box_size=50.0, npix=1024):
    pos_ = np.ascontiguousarray(positions.astype(np.float32))[:, [0, 1]]
    mass_ = np.ascontiguousarray(masses.astype(np.float32))
    field = np.zeros((npix, npix), dtype=np.float32)
    MASL.MA(pos_, field, box_size, MAS='CIC', W=mass_, verbose=False)
    return field

def process_halo_with_full_periodic_tiling(positions, masses, halo_center,
                                           box_size=50.0, npix=1024, seed=None):
    if seed is not None:
        np.random.seed(seed)

    centered_pos = apply_periodic_boundary_minimum_image(positions, halo_center, box_size)

    margin = box_size
    in_cube = np.all(np.abs(centered_pos) < margin, axis=1)
    extracted_pos = centered_pos[in_cube]
    extracted_mass = masses[in_cube]

    tiled_pos, tiled_mass = create_periodic_copies_for_rotation(
        extracted_pos, extracted_mass, box_size, buffer=10.0
    )

    rot = Rotation.from_euler('xyz', np.random.uniform(0, 2*np.pi, 3))
    rot_matrix = rot.as_matrix()
    rotated_pos = tiled_pos @ rot_matrix.T

    shifted_pos = rotated_pos + box_size / 2.0
    in_final_box = np.all((shifted_pos >= 0) & (shifted_pos < box_size), axis=1)
    final_pos = shifted_pos[in_final_box]
    final_mass = tiled_mass[in_final_box]

    mass_map = pixelize_z_projection(final_pos, final_mass, box_size, npix)
    return mass_map, rot_matrix

def extract_multiscale_cutouts(field_2d_full, box_size, target_resolution=128):
    scales_mpc = [6.25, 12.5, 25.0, 50.0]
    multiscale = np.zeros((4, target_resolution, target_resolution), dtype=np.float32)

    full_resolution = field_2d_full.shape[0]
    pix_size = box_size / full_resolution
    center_pix = full_resolution // 2

    for i, scale_size in enumerate(scales_mpc):
        if scale_size >= box_size:
            factor = full_resolution // target_resolution
            if factor > 1:
                downsampled = field_2d_full.reshape(
                    target_resolution, factor, target_resolution, factor
                ).mean(axis=(1, 3))
                multiscale[i] = downsampled
            else:
                multiscale[i] = field_2d_full
        else:
            half_size_pix = int(scale_size / (2 * pix_size))
            start = center_pix - half_size_pix
            end = center_pix + half_size_pix

            cutout = field_2d_full[start:end, start:end]
            cutout_size_pix = cutout.shape[0]

            if cutout_size_pix == target_resolution:
                multiscale[i] = cutout
            elif cutout_size_pix > target_resolution:
                factor = cutout_size_pix // target_resolution
                downsampled = cutout.reshape(
                    target_resolution, factor, target_resolution, factor
                ).mean(axis=(1, 3))
                multiscale[i] = downsampled
            else:
                from scipy.ndimage import zoom
                zoom_factor = target_resolution / cutout_size_pix
                multiscale[i] = zoom(cutout, zoom_factor, order=1)

    return multiscale


# ========== Resume helpers ==========

def check_halo_complete(output_dir, sim_id, halo_idx, num_rotations):
    """Check if all rotations for a low-mass halo have been generated."""
    for rot_idx in range(num_rotations):
        output_file = os.path.join(output_dir, f'sim_{sim_id}_lowmass_halo_{halo_idx}_rot_{rot_idx}.npz')
        if not os.path.exists(output_file):
            return False
    return True


# ========== Main processing ==========

def process_single_simulation(sim_id, output_path, box_size=50.0, npix=1024):
    """Process low-mass halos for a single simulation."""
    print(f"Rank {rank}: Sim {sim_id}: Loading data...")
    sim_path = os.path.join(hydro_base, f'SB35_{sim_id}')
    nbody_path = os.path.join(nbody_base, f'SB35_{sim_id}')

    voxel_resolution = npix
    BOX_SIZE = box_size

    dm_pos, dm_mass, hydro_dm_pos, hydro_dm_mass, gas_pos, gas_mass, star_pos, star_mass = \
        load_simulation(nbody_path, os.path.join(sim_path, 'snapdir_090'))

    params = load_params(sim_id)
    fof_file = os.path.join(fof_nbody_base, f'SB35_{sim_id}', 'fof_subhalo_tab_090.hdf5')
    
    # Load only halos in the low-mass range
    halo_pos, halo_mass, halo_indices = load_halos_in_mass_range(
        fof_file, mass_low=mass_low, mass_high=mass_high
    )

    if len(halo_pos) == 0:
        print(f"Rank {rank}: Sim {sim_id}: No halos in range ({mass_low:.0e}, {mass_high:.0e}], skipping")
        return

    print(f"Rank {rank}: Sim {sim_id}: Processing {len(halo_pos)} low-mass halos "
          f"({mass_low:.0e} < M <= {mass_high:.0e}) with {num_rotations} rotation(s) each")

    halos_processed = 0
    halos_skipped = 0

    for halo_idx in range(len(halo_pos)):
        # Check if already complete (resume capability)
        if check_halo_complete(output_path, sim_id, halo_idx, num_rotations):
            halos_skipped += 1
            continue

        halo_center = halo_pos[halo_idx]

        for rotation_idx in range(num_rotations):
            output_file = os.path.join(
                output_path, f'sim_{sim_id}_lowmass_halo_{halo_idx}_rot_{rotation_idx}.npz'
            )

            # Unique seed: offset by 50000 to avoid collision with original run's seeds
            seed = 50000 + sim_id * 1000 + halo_idx * 100 + rotation_idx

            # Process all particle species
            nbody_map, rot_matrix = process_halo_with_full_periodic_tiling(
                dm_pos, dm_mass, halo_center, BOX_SIZE, voxel_resolution, seed=seed
            )
            gas_map, _ = process_halo_with_full_periodic_tiling(
                gas_pos, gas_mass, halo_center, BOX_SIZE, voxel_resolution, seed=seed
            )
            star_map, _ = process_halo_with_full_periodic_tiling(
                star_pos, star_mass, halo_center, BOX_SIZE, voxel_resolution, seed=seed
            )
            hydro_dm_map, _ = process_halo_with_full_periodic_tiling(
                hydro_dm_pos, hydro_dm_mass, halo_center, BOX_SIZE, voxel_resolution, seed=seed
            )

            center_pix = voxel_resolution // 2
            stretch = resolution // 2

            nbody = extract_multiscale_cutouts(nbody_map, BOX_SIZE, target_resolution=resolution)
            target_star = star_map[center_pix - stretch : center_pix + stretch,
                                   center_pix - stretch : center_pix + stretch]
            target_gas = gas_map[center_pix - stretch : center_pix + stretch,
                                 center_pix - stretch : center_pix + stretch]
            target_hydro_dm = hydro_dm_map[center_pix - stretch : center_pix + stretch,
                                           center_pix - stretch : center_pix + stretch]
            target = np.stack([target_hydro_dm, target_gas, target_star], axis=0)

            np.savez_compressed(
                output_file,
                condition=nbody[0],
                target=target,
                large_scale=nbody[1:],
                params=np.array(list(params.values())),
                halo_mass=halo_mass[halo_idx],
                halo_center=halo_center,
            )

            del nbody_map, gas_map, star_map, hydro_dm_map, nbody
            del target_star, target_gas, target_hydro_dm, target, rot_matrix

        halos_processed += 1

    print(f"Rank {rank}: Sim {sim_id}: Complete ({halos_processed} processed, {halos_skipped} skipped)")


if __name__ == '__main__':
    # Write into the SAME output directory as the original run
    output_dir_base = os.path.join(output_base_root, 'train_data_rotated2_128_cpu')

    if rank == 0:
        os.makedirs(os.path.join(output_dir_base, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir_base, 'test'), exist_ok=True)
        print(f"Output directory: {output_dir_base}")
        print(f"Mass range: ({mass_low:.0e}, {mass_high:.0e}]")
        print(f"Rotations per halo: {num_rotations}")
        print(f"Total simulations: {start_sim} to {end_sim}")
        print(f"Using {size} MPI ranks")
        print()

        # Find incomplete simulations
        print("Scanning for incomplete simulations...")
        incomplete_sims = []
        complete_sims = []

        for sim_id in range(start_sim, end_sim):
            sim_path = os.path.join(hydro_base, f'SB35_{sim_id}')
            nbody_path = os.path.join(nbody_base, f'SB35_{sim_id}')
            if not os.path.exists(sim_path) or not os.path.exists(nbody_path):
                continue

            split = 'test' if sim_id in test_sims else 'train'
            output_path = os.path.join(output_dir_base, split, f'sim_{sim_id}')

            # Count how many lowmass halo files exist for this sim
            if os.path.exists(output_path):
                lowmass_files = [f for f in os.listdir(output_path)
                                 if f.startswith(f'sim_{sim_id}_lowmass_halo_') and f.endswith('.npz')]
            else:
                lowmass_files = []

            # Check FOF catalog to see how many low-mass halos there should be
            fof_file = os.path.join(fof_nbody_base, f'SB35_{sim_id}', 'fof_subhalo_tab_090.hdf5')
            halo_pos, halo_mass, _ = load_halos_in_mass_range(fof_file, mass_low, mass_high)
            expected_files = len(halo_pos) * num_rotations

            if expected_files == 0:
                complete_sims.append(sim_id)
            elif len(lowmass_files) >= expected_files:
                complete_sims.append(sim_id)
            else:
                incomplete_sims.append(sim_id)
                if len(lowmass_files) > 0:
                    print(f"  Sim {sim_id}: Partially complete ({len(lowmass_files)}/{expected_files} files)")

        print(f"\nFound {len(complete_sims)} complete/empty simulations")
        print(f"Found {len(incomplete_sims)} incomplete simulations")
        print(f"Will process {len(incomplete_sims)} simulations across {size} ranks")
        print(f"  → ~{len(incomplete_sims)/size:.1f} simulations per rank\n")
    else:
        incomplete_sims = None

    # Broadcast list of incomplete sims to all ranks
    incomplete_sims = comm.bcast(incomplete_sims, root=0)

    # Distribute incomplete simulations across ranks
    my_sim_ids = incomplete_sims[rank::size]

    if len(my_sim_ids) == 0:
        print(f"Rank {rank}: No simulations assigned")
    else:
        print(f"Rank {rank}: Assigned {len(my_sim_ids)} simulations: "
              f"{my_sim_ids[:5]}{'...' if len(my_sim_ids) > 5 else ''}")

    comm.Barrier()

    voxel_resolution = int(resolution * BOX_SIZE / 6.25)  # 1024

    for sim_id in my_sim_ids:
        split = 'test' if sim_id in test_sims else 'train'
        output_path = os.path.join(output_dir_base, split, f'sim_{sim_id}')
        os.makedirs(output_path, exist_ok=True)

        print(f"Rank {rank}: Starting sim {sim_id} ({split})")
        process_single_simulation(sim_id, output_path, BOX_SIZE, voxel_resolution)
        print(f"Rank {rank}: Finished sim {sim_id}\n")

    comm.Barrier()

    if rank == 0:
        print("All low-mass halo processing complete!")
