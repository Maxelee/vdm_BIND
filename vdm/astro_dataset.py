import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from lightning.pytorch import LightningDataModule
import h5py
import os
import glob
import joblib

from .augmentation import Translate, Permutate, Flip, Normalize, Resize, RandomRotate
from .constants import norms_256 as norms

class AstroDataset(TensorDataset):
    def __init__(self, file_paths, transform=None):  # Add halo_mass parameters
        """
        Args:
            file_paths: List of file paths to data
            transform: Data transformations
            field: Field to use (for compatibility)
            use_omega_m: Whether to include Omega_m conditioning
            omega_m_value: Fixed Omega_m value or function to generate it
        """
        self.file_paths = file_paths
        self.transform = transform
        

    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self, index):
        try:
            with np.load(self.file_paths[index]) as data:
                if 'condition' not in data or 'target' not in data or 'params' not in data:
                    raise KeyError("Missing required keys")
                m_dm = data['condition']
                m_target = data['target']
                conditions = data['params']  # Use all parameters (not [:-1])
                large_scale = data['large_scale']
                if large_scale.shape[0] == 4:
                    large_scale = large_scale[1:]
                
                # # Normalize large_scale - handle both (H, W) and (N, H, W) cases
                # if large_scale.ndim == 2:
                #     large_scale /= conditions[0]
                # elif large_scale.ndim == 3:
                #     large_scale /= conditions[0]
                # else:
                #     raise ValueError(f"Unexpected large_scale shape: {large_scale.shape}")
                
                # m_dm /= conditions[0]
                # m_target[0] /= conditions[0]
                # m_target[1] /= conditions[6]
                # m_target[2] /= conditions[6]
                # conditions_mask = np.ones(35)
                # conditions_mask[[0, 1, 6,7,8]] = 0
                # conditions = conditions[conditions_mask==1]
        except Exception as e:
            print(f"Skipping corrupted file: {self.file_paths[index]} ({e})")
            # Try next index (wrap around if at end)
            next_index = (index + 1) % len(self.file_paths)
            return self.__getitem__(next_index)

        m_dm = torch.from_numpy(m_dm).unsqueeze(0).float()
        
        # Handle large_scale shape: (H, W) -> (1, H, W) or (N, H, W) -> (N, H, W)
        large_scale = torch.from_numpy(large_scale).float()
        if large_scale.ndim == 2:
            large_scale = large_scale.unsqueeze(0)
        
        m_target = torch.from_numpy(m_target).float()
        conditions = torch.from_numpy(conditions).float()

        if self.transform:
            m_dm, large_scale, m_target = self.transform((m_dm, large_scale, m_target))

        result = [m_dm, large_scale, m_target, conditions]
        return tuple(result)

class AstroDataModule(LightningDataModule):
    def __init__(
            self, 
            train_transforms=None, 
            test_transforms=None, 
            batch_size=1,
            num_workers=1,
            dataset='illustris',
            data_root = '/mnt/home/mlee1/ceph/50Mpc_boxes/subimages/snapdir_090',
            limit_train_samples=None,  # NEW: Limit training samples for fast ablation
            limit_val_samples=None,    # NEW: Limit validation samples
        ):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.data_root = data_root
        self.limit_train_samples = limit_train_samples
        self.limit_val_samples = limit_val_samples


    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            # Find all NPZ files recursively
            pattern = os.path.join(self.data_root, "**", "*halo*.npz")
            self.all_files = sorted(glob.glob(pattern, recursive=True))
            print(f"Found {len(self.all_files)} total files")
            
            # Optionally limit dataset size for fast ablation studies with RANDOM sampling
            if self.limit_train_samples is not None:
                print(f"⚡ FAST ABLATION MODE: Randomly selecting {self.limit_train_samples} training samples")
                # Set random seed for reproducibility
                rng = np.random.RandomState(seed=42)
                # Randomly sample files spanning the full parameter space
                indices = rng.choice(len(self.all_files), size=self.limit_train_samples, replace=False)
                self.all_files = [self.all_files[i] for i in sorted(indices)]
                print(f"  → Selected {len(self.all_files)} random samples from full dataset")
            
            # Split into train/val/test
            total = len(self.all_files)

            data = AstroDataset(
                self.all_files, 
                transform=self.train_transforms, 
            )
            train_set_size = int(total * 0.8)
            valid_set_size = total - train_set_size
            generator = torch.Generator().manual_seed(342)
            self.train_data, self.valid_data = random_split(
                data, [train_set_size, valid_set_size], generator=generator
            )
            
            # Optionally limit validation set for faster epochs
            if self.limit_val_samples is not None and self.limit_val_samples < len(self.valid_data):
                print(f"⚡ FAST ABLATION MODE: Limiting to {self.limit_val_samples} validation samples")
                # Create subset of validation data
                val_indices = list(range(self.limit_val_samples))
                self.valid_data = torch.utils.data.Subset(self.valid_data, val_indices)

        if stage == "test" or stage is None:
            # Find all NPZ files recursively
            pattern = os.path.join(self.data_root, "**", "*halo*.npz")
            self.all_files = sorted(glob.glob(pattern, recursive=True))

            self.test_data = AstroDataset(
                self.all_files, 
                transform=self.test_transforms, 
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            pin_memory=True,  # PERFORMANCE: Enable for faster GPU transfer (adds ~100MB overhead)
            persistent_workers=True if self.num_workers > 0 else False,  # PERFORMANCE: Keep workers alive (CRITICAL for 400k files!)
            prefetch_factor=2 if self.num_workers > 0 else None,  # PERFORMANCE: Prefetch 2 batches per worker (conservative)
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,  # PERFORMANCE: Enable for faster GPU transfer (adds ~100MB overhead)
            persistent_workers=True if self.num_workers > 0 else False,  # PERFORMANCE: Keep workers alive (CRITICAL for 400k files!)
            prefetch_factor=2 if self.num_workers > 0 else None,  # PERFORMANCE: Prefetch 2 batches per worker (conservative)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=1, num_workers=self.num_workers,
        )


def astro_normalizations(dataset, stellar_stats_path=None, quantile_path=None):
    """
    Create normalization transforms for astro data.
    
    3-channel mode (CleanVDM):
        - All channels log-transformed: log10(x + 1)
        - DM and Gas: Z-score normalized
        - Stars: Z-score normalized OR quantile transformed (if quantile_path provided)
    
    Args:
        dataset: Dataset name (for DM/Gas normalization stats)
        stellar_stats_path: Path to stellar normalization stats file (required if not using quantile)
        quantile_path: Path to quantile transformer .pkl file (optional, alternative to Z-score)
    """
    # Log transform for ALL channels (including Stars)
    # Use 1e-3 offset as in notebook for stellar channel
    if quantile_path is not None:
        # For quantile normalization, use 1e-3 offset for stellar channel only
        log_transform = transforms.Lambda(
            lambda x: (
                torch.log10(x[0] + 1),  # DM conditioning
                torch.log10(x[1] + 1),  # Large-scale
                torch.cat([
                    torch.log10(x[2][0:1] + 1),  # DM target
                    torch.log10(x[2][1:2] + 1),  # Gas target
                    torch.log10(x[2][2:3] + 1),  # Stars target (1e-3 offset for quantile compatibility)
                ], dim=0)
            )
        )
    elif stellar_stats_path is not None:
        # For Z-score normalization, use 1e-3 offset for stellar channel only   
        log_transform = transforms.Lambda(
            lambda x: (
                torch.log10(x[0] + 1),  # DM conditioning
                torch.log10(x[1] + 1),  # Large-scale
                torch.cat([
                    torch.log10(x[2][0:1] + 1),  # DM target
                    torch.log10(x[2][1:2] + 1),  # Gas target
                    torch.log10(x[2][2:3] + 1),  # Stars target (1e-3 offset for quantile compatibility)
                ], dim=0)
            )
        )
    else:
        raise ValueError(
            f"⚠️  Stellar normalization required for 3-channel mode!\n"
            f"   Please provide either:\n"
            f"   - quantile_path: {quantile_path} (for quantile normalization)\n"
            f"   - stellar_stats_path: {stellar_stats_path} (for Z-score normalization)"
        )
    
    # Load stellar normalization stats (required for 3-channel mode)
    star_mag_mean = None
    star_mag_std = None
    
    # Load DM normalization stats
    dm_stats_path = '/mnt/home/mlee1/vdm_BIND/dark_matter_normalization_stats.npz'
    if os.path.exists(dm_stats_path):
        stats = np.load(dm_stats_path)
        dm_mag_mean = float(stats['dm_mag_mean'])
        dm_mag_std = float(stats['dm_mag_std'])
        print(f"✓ Loaded DM stats: mean={dm_mag_mean:.6f}, std={dm_mag_std:.6f}")
    else:
        # Fallback to constants
        dm_mag_mean = norms[dataset][4]
        dm_mag_std = norms[dataset][5]
        print(f"⚠️  Using fallback DM stats from constants: mean={dm_mag_mean:.6f}, std={dm_mag_std:.6f}")
    
    # Load Gas normalization stats
    gas_stats_path = '/mnt/home/mlee1/vdm_BIND/gas_normalization_stats.npz'
    if os.path.exists(gas_stats_path):
        stats = np.load(gas_stats_path)
        gas_mag_mean = float(stats['gas_mag_mean'])
        gas_mag_std = float(stats['gas_mag_std'])
        print(f"✓ Loaded Gas stats: mean={gas_mag_mean:.6f}, std={gas_mag_std:.6f}")
    else:
        # Fallback to constants
        gas_mag_mean = norms[dataset][2]
        gas_mag_std = norms[dataset][3]
        print(f"⚠️  Using fallback Gas stats from constants: mean={gas_mag_mean:.6f}, std={gas_mag_std:.6f}")
    
    # Load stellar normalization: quantile OR Z-score
    quantile_transformer = None
    star_mag_mean = None
    star_mag_std = None
    
    if quantile_path and os.path.exists(quantile_path):
        # Use quantile transformation
        print(f"✓ Loading quantile transformer from: {quantile_path}")
        import joblib
        quantile_transformer = joblib.load(quantile_path)
        print(f"  Quantile transformer loaded (n_quantiles={len(quantile_transformer.quantiles_)})")
        print(f"  Output distribution: {quantile_transformer.output_distribution}")
    elif stellar_stats_path and os.path.exists(stellar_stats_path):
        # Use Z-score normalization
        print(f"✓ Loading stellar Z-score stats from: {stellar_stats_path}")
        stats = np.load(stellar_stats_path)
        star_mag_mean = float(stats['star_mag_mean'])
        star_mag_std = float(stats['star_mag_std'])
        print(f"  mean={star_mag_mean:.6f}, std={star_mag_std:.6f}")
    else:
        raise ValueError(
            f"⚠️  Stellar normalization required for 3-channel mode!\n"
            f"   Please provide either:\n"
            f"   - quantile_path: {quantile_path} (for quantile normalization)\n"
            f"   - stellar_stats_path: {stellar_stats_path} (for Z-score normalization)"
        )
    
    norm = Normalize(
        mean_input=norms[dataset][6],
        std_input=norms[dataset][7],
        mean_target_dmh=dm_mag_mean,
        std_target_dmh=dm_mag_std,
        mean_target_gas=gas_mag_mean,
        std_target_gas=gas_mag_std,
        mean_target_stars=star_mag_mean,
        std_target_stars=star_mag_std,
        quantile_transformer=quantile_transformer,
    )
    
    return transforms.Compose([log_transform, norm])


def get_astro_data(
    dataset, 
    data_root, 
    num_workers=1, 
    batch_size=10, 
    stage=None, 
    resize=None,
    limit_train_samples=None,
    limit_val_samples=None,
    stellar_stats_path='/mnt/home/mlee1/vdm_BIND/stellar_normalization_stats.npz',  # Z-score normalization
    quantile_path=None,  # Quantile normalization (alternative to stellar_stats_path)
):
    """
    Get astro data with optional sample limiting for fast ablation studies
    
    3-channel mode (CleanVDM):
        - All channels log-transformed
        - DM and Gas: Z-score normalized
        - Stars: Z-score OR quantile normalized (if quantile_path provided)
    
    Args:
        dataset: Dataset name
        data_root: Root directory for data
        num_workers: Number of workers for data loading
        batch_size: Batch size
        stage: Training stage
        resize: Resize dimensions
        limit_train_samples: Limit training samples (e.g., 5000 for fast ablation)
        limit_val_samples: Limit validation samples (e.g., 500 for faster epochs)
        stellar_stats_path: Path to stellar Z-score normalization stats
        quantile_path: Path to quantile transformer .pkl (alternative to stellar_stats_path)
    """
    # Create transforms with stellar normalization (Z-score OR quantile)
    train_transforms = [
        astro_normalizations(dataset, stellar_stats_path=stellar_stats_path, quantile_path=quantile_path)
    ]
    test_transforms = [
        astro_normalizations(dataset, stellar_stats_path=stellar_stats_path, quantile_path=quantile_path)
    ]
    # train_transforms = [
    #         astro_normalizations(dataset)
    #         # RandomRotate(180)
    #         # Translate(),
    #         #Flip(2)
    #         #Permutate(2)
    #     ]

    # test_transforms = [astro_normalizations(dataset)]

    if resize is not None:
        train_transforms += [Resize(resize)]
        test_transforms += [Resize(resize)]

    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    dm = AstroDataModule(
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=num_workers,
        batch_size=batch_size,
        dataset=dataset,
        data_root=data_root,
        limit_train_samples=limit_train_samples,
        limit_val_samples=limit_val_samples,
    )
    dm.setup(stage=stage)
    return dm
