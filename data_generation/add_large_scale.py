import numpy as np
import os
import h5py
import argparse
import mpi4py.MPI as MPI

# Command-line arguments
parser = argparse.ArgumentParser(description='Add large_scale projections to existing train_2d data.')
parser.add_argument('--data_dir', type=str, default='/mnt/home/mlee1/ceph/train_data_1024/train_2d',
                    help='Directory containing train_2d data')
parser.add_argument('--projected_images_dir', type=str, default='/mnt/home/mlee1/ceph/train_data_1024/projected_images',
                    help='Directory containing projected images')
parser.add_argument('--fof_nbody_base', type=str, default='/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/SB35',
                    help='Base path for FOF halo catalogs')
parser.add_argument('--large_region_size', type=float, default=12.5,
                    help='Size of large scale regions in Mpc (default: 12.5)')
parser.add_argument('--target_res', type=int, default=128,
                    help='Target resolution for large_scale (default: 128)')

args = parser.parse_args()

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data_dir = args.data_dir
projected_images_dir = args.projected_images_dir
fof_nbody_base = args.fof_nbody_base
large_region_size = args.large_region_size
target_res = args.target_res
box_size = 50.0  # Mpc/h
resolution = 1024  # Full box resolution

def sim_dir(i):
    return f'SB35_{i}'

def load_halos(fof_file):
    # Load halo catalog from nbody FOF
    halo_pos = []
    halo_mass = []
    if os.path.exists(fof_file):
        with h5py.File(fof_file, 'r') as f:
            if 'Group/GroupPos' in f:
                halo_pos = f['Group/GroupPos'][:]
            if 'Group/Group_M_Crit200' in f:
                halo_mass = f['Group/Group_M_Crit200'][:]
    if len(halo_mass) > 0:
        halo_pos = np.array(halo_pos) / 1000.0  # kpc/h -> Mpc/h
        halo_mass = np.array(halo_mass) * 1e10  # convert from 1e10 M_sun units to M_sun
        mask = halo_mass > 1e13
        return halo_pos[mask], halo_mass[mask]
    else:
        return np.array([]), np.array([])

def extract_large_projection(field_2d, center, size, box_size, resolution, axis=2, target_res=128):
    # Extract cube of size 'size' around center with periodic boundaries
    pix_size = box_size / resolution
    half_size_pix = int(size / (2 * pix_size))
    full_size_pix = 2 * half_size_pix
    center_pix = (center / pix_size).astype(int)
    
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
    
    # Downsample to target_res x target_res
    factor = full_size_pix // target_res
    if factor > 1:
        # Reshape and average
        extracted = extracted.reshape(target_res, factor, target_res, factor).mean(axis=(1, 3))
    
    return extracted

# Get list of sim subdirs
sim_subdirs = [d for d in os.listdir(data_dir) if d.startswith('sim_')]
sim_subdirs.sort(key=lambda x: int(x.split('_')[1]))

# Distribute simulations across ranks
total_sims = len(sim_subdirs)
sims_per_rank = total_sims // size
remainder = total_sims % size
start = rank * sims_per_rank + min(rank, remainder)
end = start + sims_per_rank + (1 if rank < remainder else 0)

print(f"Rank {rank}: Processing sims {start} to {end-1}")

for sim_idx in range(start, end):
    sim_subdir = sim_subdirs[sim_idx]
    sim_id = int(sim_subdir.split('_')[1])
    sim_path = os.path.join(data_dir, sim_subdir)
    
    # Load projected images for this sim
    proj_xy_file = os.path.join(projected_images_dir, f'projections_xy_{sim_id}.npz')
    proj_xz_file = os.path.join(projected_images_dir, f'projections_xz_{sim_id}.npz')
    proj_yz_file = os.path.join(projected_images_dir, f'projections_yz_{sim_id}.npz')
    
    if not os.path.exists(proj_xy_file):
        continue
    
    proj_xy = np.load(proj_xy_file)
    dm_2d_xy = proj_xy['dm']
    
    proj_xz = np.load(proj_xz_file)
    dm_2d_xz = proj_xz['dm']
    
    proj_yz = np.load(proj_yz_file)
    dm_2d_yz = proj_yz['dm']
    
    # Load halos
    fof_file = os.path.join(fof_nbody_base, sim_dir(sim_id), 'fof_subhalo_tab_090.hdf5')
    halo_pos, _ = load_halos(fof_file)
    
    # Get list of halo files
    halo_files_xy = [f for f in os.listdir(sim_path) if f.startswith('halo_') and f.endswith('_xy.npz')]
    halo_files_xy.sort(key=lambda x: int(x.split('_')[1]))
    
    for halo_file in halo_files_xy:
        halo_idx = int(halo_file.split('_')[1])
        if halo_idx >= len(halo_pos):
            continue
        pos = halo_pos[halo_idx]
        
        # Extract large_scale for xy
        large_scale_xy = extract_large_projection(dm_2d_xy, pos, large_region_size, box_size, resolution, axis=2, target_res=target_res)
        
        # Load existing npz
        halo_path = os.path.join(sim_path, halo_file)
        data = np.load(halo_path)
        existing_data = {key: data[key] for key in data.keys()}
        
        # Add large_scale
        existing_data['large_scale'] = large_scale_xy
        
        # Save back
        np.savez_compressed(halo_path, **existing_data)
    
    # Similarly for xz
    halo_files_xz = [f for f in os.listdir(sim_path) if f.startswith('halo_') and f.endswith('_xz.npz')]
    halo_files_xz.sort(key=lambda x: int(x.split('_')[1]))
    
    for halo_file in halo_files_xz:
        halo_idx = int(halo_file.split('_')[1])
        if halo_idx >= len(halo_pos):
            continue
        pos = halo_pos[halo_idx]
        
        large_scale_xz = extract_large_projection(dm_2d_xz, pos, large_region_size, box_size, resolution, axis=1, target_res=target_res)
        
        halo_path = os.path.join(sim_path, halo_file)
        data = np.load(halo_path)
        existing_data = {key: data[key] for key in data.keys()}
        existing_data['large_scale'] = large_scale_xz
        np.savez_compressed(halo_path, **existing_data)
    
    # Similarly for yz
    halo_files_yz = [f for f in os.listdir(sim_path) if f.startswith('halo_') and f.endswith('_yz.npz')]
    halo_files_yz.sort(key=lambda x: int(x.split('_')[1]))
    
    for halo_file in halo_files_yz:
        halo_idx = int(halo_file.split('_')[1])
        if halo_idx >= len(halo_pos):
            continue
        pos = halo_pos[halo_idx]
        
        large_scale_yz = extract_large_projection(dm_2d_yz, pos, large_region_size, box_size, resolution, axis=0, target_res=target_res)
        
        halo_path = os.path.join(sim_path, halo_file)
        data = np.load(halo_path)
        existing_data = {key: data[key] for key in data.keys()}
        existing_data['large_scale'] = large_scale_yz
        np.savez_compressed(halo_path, **existing_data)
    
    if (sim_idx - start) % 10 == 0:
        print(f"Rank {rank}: Processed {sim_idx - start + 1}/{end - start} sims")

print(f"Rank {rank}: Finished processing.")

if rank == 0:
    print("Large scale addition complete.")