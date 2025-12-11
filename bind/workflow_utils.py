"""
Workflow utilities for BIND inference.

Provides ConfigLoader, ModelManager, and sampling functions.
Supports both 'clean' (3-channel) and 'triple' (3 separate 1-channel) model types.
"""
from lightning.pytorch import Trainer, seed_everything
import sys
import os

# Import vdm package - assumes vdm_BIND is installed or in path
from vdm.astro_dataset import get_astro_data
from vdm import vdm_model_clean as vdm_module, networks_clean as networks
from vdm import vdm_model_triple as vdm_triple_module
from vdm.utils import draw_figure
import torch
import re
import glob
import configparser
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

# Import centralized path configuration
from config import PROJECT_ROOT, DATA_DIR, NORMALIZATION_STATS_DIR

def load_normalization_stats(base_path=None):
    """
    Load normalization statistics from .npz files.
    
    Args:
        base_path: Directory containing the normalization .npz files.
                   Defaults to NORMALIZATION_STATS_DIR from config.py
    
    Returns:
        dict with keys:
            - dm_mag_mean, dm_mag_std
            - gas_mag_mean, gas_mag_std
            - star_mag_mean, star_mag_std
    """
    if base_path is None:
        base_path = NORMALIZATION_STATS_DIR
    
    stats = {}
    
    # Load DM stats
    dm_path = os.path.join(base_path, 'dark_matter_normalization_stats.npz')
    if os.path.exists(dm_path):
        dm_stats = np.load(dm_path)
        stats['dm_mag_mean'] = float(dm_stats['dm_mag_mean'])
        stats['dm_mag_std'] = float(dm_stats['dm_mag_std'])
        print(f"✓ Loaded DM normalization: mean={stats['dm_mag_mean']:.6f}, std={stats['dm_mag_std']:.6f}")
    else:
        raise FileNotFoundError(f"DM normalization file not found: {dm_path}")
    
    # Load Gas stats
    gas_path = os.path.join(base_path, 'gas_normalization_stats.npz')
    if os.path.exists(gas_path):
        gas_stats = np.load(gas_path)
        stats['gas_mag_mean'] = float(gas_stats['gas_mag_mean'])
        stats['gas_mag_std'] = float(gas_stats['gas_mag_std'])
        print(f"✓ Loaded Gas normalization: mean={stats['gas_mag_mean']:.6f}, std={stats['gas_mag_std']:.6f}")
    else:
        raise FileNotFoundError(f"Gas normalization file not found: {gas_path}")
    
    # Load Stellar stats
    star_path = os.path.join(base_path, 'stellar_normalization_stats.npz')
    if os.path.exists(star_path):
        star_stats = np.load(star_path)
        stats['star_mag_mean'] = float(star_stats['star_mag_mean'])
        stats['star_mag_std'] = float(star_stats['star_mag_std'])
        print(f"✓ Loaded Stellar normalization: mean={stats['star_mag_mean']:.6f}, std={stats['star_mag_std']:.6f}")
    else:
        raise FileNotFoundError(f"Stellar normalization file not found: {star_path}")
    
    return stats


class ConfigLoader:
    """Load and manage configuration parameters from an INI file."""
    
    def __init__(self, config_path='config.ini', verbose=False, train_samples=False):
        self.config_path = config_path
        self.verbose = verbose
        self.train_samples = train_samples
        self._load_config()
        self._state_initialization()

    def _load_config(self):
        """Load and parse configuration file, converting types automatically."""
        config = configparser.ConfigParser()
        config.read(self.config_path)
        params = config['TRAINING']

        # Define expected types for parameters
        int_params = {'seed', 'cropsize', 'batch_size', 'num_workers', 'embedding_dim',
                      'norm_groups', 'n_blocks', 'n_attention_heads', 'version', 'ndim', 
                      'conditioning_channels', 'large_scale_channels', 'field_weight_warmup_steps',
                      'cross_attention_heads', 'cross_attention_chunk_size', 'cross_attn_cond_downsample_factor',
                      # DDPM/score_models parameters
                      'n_params', 'nf', 'num_res_blocks', 'n_sampling_steps', 'accumulate_grad_batches',
                      'ema_update_after_step', 'ema_update_every'}
        float_params = {'gamma_min', 'gamma_max', 'learning_rate', 'mass_conservation_weight',
                        'sparsity_threshold', 'sparse_loss_weight', 'focal_alpha', 'focal_gamma',
                        'param_prediction_weight', 'cross_attention_dropout',
                        # DDPM/score_models parameters
                        'beta_min', 'beta_max', 'sigma_min', 'sigma_max', 'ema_decay', 'dropout'}
        bool_params = {'use_large_scale', 'use_fourier_features', 'fourier_legacy', 'legacy_fourier',
                       'add_attention', 'use_progressive_field_weighting', 'use_mass_conservation',
                       'use_sparsity_aware_loss', 'use_focal_loss', 'use_param_prediction', 
                       'use_auxiliary_mask', 'antithetic_time_sampling', 'use_cross_attention',
                       'use_chunked_cross_attention', 'downsample_cross_attn_cond',
                       # DDPM/score_models parameters
                       'use_param_conditioning', 'attention', 'enable_ema', 'enable_early_stopping',
                       'enable_gradient_monitoring'}
        
        # Assign attributes with correct types
        for key, value in params.items():
            if key in int_params:
                setattr(self, key, int(value))
            elif key in float_params:
                setattr(self, key, float(value))
            elif key in bool_params:
                setattr(self, key, value.lower() in ('true', '1', 'yes'))
            else:
                setattr(self, key, value)
        
        # Handle backward compatibility: legacy_fourier -> fourier_legacy
        if hasattr(self, 'legacy_fourier') and not hasattr(self, 'fourier_legacy'):
            self.fourier_legacy = self.legacy_fourier
            if self.verbose:
                print(f"[ConfigLoader] Converted legacy_fourier={self.legacy_fourier} to fourier_legacy")
        
        # UPDATED: Handle data_noise specially (can be single float or tuple)
        if 'data_noise' in params:
            data_noise_str = params['data_noise']
            if ',' in data_noise_str:
                # Per-channel: parse as tuple of floats
                self.data_noise = tuple(map(float, data_noise_str.split(',')))
            else:
                # Single value: parse as float
                self.data_noise = float(data_noise_str)
        else:
            self.data_noise = 1e-3  # Default fallback
        
        # Set defaults for optional parameters that might not be in config
        if not hasattr(self, 'lr_scheduler'):
            self.lr_scheduler = 'cosine'
        if not hasattr(self, 'lambdas'):
            self.lambdas = '1.0,1.0,1.0'
        if not hasattr(self, 'channel_weights'):
            self.channel_weights = '1.0,1.0,1.0'
        if not hasattr(self, 'focal_gamma'):
            self.focal_gamma = 2.0
        if not hasattr(self, 'param_prediction_weight'):
            self.param_prediction_weight = 0.01
        if not hasattr(self, 'use_focal_loss'):
            self.use_focal_loss = False
        if not hasattr(self, 'use_param_prediction'):
            self.use_param_prediction = False
        if not hasattr(self, 'antithetic_time_sampling'):
            self.antithetic_time_sampling = True
        if not hasattr(self, 'add_attention'):
            self.add_attention = True
        
        # Set defaults for cross-attention parameters (networks_clean.py)
        if not hasattr(self, 'use_cross_attention'):
            self.use_cross_attention = False
        if not hasattr(self, 'cross_attention_location'):
            self.cross_attention_location = 'bottleneck'
        if not hasattr(self, 'cross_attention_heads'):
            self.cross_attention_heads = 8
        if not hasattr(self, 'cross_attention_dropout'):
            self.cross_attention_dropout = 0.1
        if not hasattr(self, 'use_chunked_cross_attention'):
            self.use_chunked_cross_attention = True
        if not hasattr(self, 'cross_attention_chunk_size'):
            self.cross_attention_chunk_size = 512
        if not hasattr(self, 'downsample_cross_attn_cond'):
            self.downsample_cross_attn_cond = False
        if not hasattr(self, 'cross_attn_cond_downsample_factor'):
            self.cross_attn_cond_downsample_factor = 2
        
        # Set defaults for optional parameters
        if not hasattr(self, 'conditioning_channels') or self.conditioning_channels is None:
            # Try to auto-detect from checkpoint if available
            self.conditioning_channels = 1  # Default to 1 (base DM channel)
            if self.verbose:
                print(f"[ConfigLoader] conditioning_channels not in config or None, using default: {self.conditioning_channels}")
        else:
            if self.verbose:
                print(f"[ConfigLoader] Found conditioning_channels in config: {self.conditioning_channels}")
                print(f"[ConfigLoader] WARNING: This will be overridden by checkpoint auto-detection!")
        
        if not hasattr(self, 'large_scale_channels'):
            # Default to 0 (no large-scale conditioning)
            self.large_scale_channels = 0
        
        if not hasattr(self, 'use_large_scale'):
            self.use_large_scale = None  # Will be auto-detected
        
        # Set defaults for Fourier features
        if not hasattr(self, 'use_fourier_features'):
            self.use_fourier_features = True  # Default to True for backward compatibility
        
        if not hasattr(self, 'fourier_legacy'):
            # Auto-detect legacy mode based on checkpoint if available
            self.fourier_legacy = None  # Will be auto-detected from checkpoint

        # Load in the min/max params from the CSV file if it exists
        if 'param_norm_path' in params and params['param_norm_path']:
            self.use_param_conditioning = True
            self.param_norm_path = params['param_norm_path']
            if not glob.glob(self.param_norm_path):
                raise FileNotFoundError(f"Conditional path not found: {self.param_norm_path}")
            minmax_df = pd.read_csv(self.param_norm_path)
            self.min = np.array(minmax_df['MinVal'].values)
            self.max = np.array(minmax_df['MaxVal'].values)
            self.Nparams = len(self.min)

        # ✅ QUANTILE NORMALIZATION SUPPORT
        # Load quantile_path if specified (for stellar channel normalization)
        if 'quantile_path' in params and params['quantile_path']:
            quantile_val = params['quantile_path'].strip()
            if quantile_val.lower() not in ['none', '']:
                self.quantile_path = quantile_val
                if self.verbose:
                    print(f"[ConfigLoader] 🌟 Quantile normalization enabled: {self.quantile_path}")
            else:
                self.quantile_path = None
        else:
            self.quantile_path = None

        if getattr(self, 'verbose', False):
            print(f"[ConfigLoader] Loaded config from: {self.config_path}")
            print(f"[ConfigLoader] Parameters:")
            for key in params:
                print(f"  {key}: {getattr(self, key, None)}")

    def _state_initialization(self):
        """
        Find the best checkpoint, supporting multiple checkpoint formats:
        
        1. New format (epoch_checkpoint): epoch-epoch=XXX-step=YYY.ckpt (files)
        2. Old format (val_checkpoint): epoch=X-step=Y-val/ dirs with elbo=-X.XXX.ckpt inside
        3. Latest checkpoints: latest-epoch=X-step=Y.ckpt (files)
        
        Priority: epoch-epoch* files > val checkpoint dirs > latest files
        """
        self.tb_log_path = f"{self.tb_logs}/{self.model_name}/version_{self.version}/"
        
        # Pattern 1: New epoch checkpoint files (epoch-epoch=XXX-step=YYY.ckpt)
        ckpts = glob.glob(f'{self.tb_log_path}/checkpoints/epoch-epoch*.ckpt')
        if not ckpts:
            ckpts = glob.glob(f'{self.tb_log_path}/**/checkpoints/epoch-epoch*.ckpt', recursive=True)
        
        # Pattern 2: Old val checkpoint directories (epoch=X-step=Y-val/)
        # These contain files like elbo=-5.514.ckpt or val_loss=0.123.ckpt
        if not ckpts:
            val_dirs = glob.glob(f'{self.tb_log_path}/checkpoints/epoch=*-val')
            if not val_dirs:
                val_dirs = glob.glob(f'{self.tb_log_path}/**/checkpoints/epoch=*-val', recursive=True)
            
            if val_dirs:
                # Sort directories by epoch number
                val_dirs.sort(key=self._natural_sort_key)
                # Get the latest epoch directory
                best_dir = val_dirs[-1]
                # Find the .ckpt file inside
                inner_ckpts = glob.glob(os.path.join(best_dir, '*.ckpt'))
                if inner_ckpts:
                    # If multiple files, sort and take best (by metric value if present)
                    inner_ckpts.sort(key=self._natural_sort_key)
                    ckpts = [inner_ckpts[0]]  # Take the best one
        
        # Pattern 3: Latest checkpoint files as fallback
        if not ckpts:
            ckpts = glob.glob(f'{self.tb_log_path}/checkpoints/latest-*.ckpt')
            if not ckpts:
                ckpts = glob.glob(f'{self.tb_log_path}/**/checkpoints/latest-*.ckpt', recursive=True)
        
        if ckpts:
            ckpts.sort(key=self._natural_sort_key)
            self.best_ckpt = ckpts[-1]
        else:
            self.best_ckpt = None

        if getattr(self, 'verbose', False):
            print(f"[ConfigLoader] TensorBoard log path: {self.tb_log_path}")
            if self.best_ckpt:
                print(f"[ConfigLoader] Found best checkpoint: {self.best_ckpt}")
            else:
                print("[ConfigLoader] No checkpoint found.")

    @staticmethod
    def _natural_sort_key(s):
        """Generate key for natural sorting of strings containing numbers."""
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)
        ]

class ModelManager:
    """
    Model manager supporting multiple model types:
    
    Model Types:
    - clean: Single 3-channel model (LightCleanVDM)
    - triple: Three independent 1-channel models (LightTripleVDM)
    - ddpm: Score-based diffusion model (score_models package)
    - interpolant: Flow matching / stochastic interpolant (LightInterpolant)
    - consistency: Consistency models (Song et al., 2023) - single/few-step sampling
    - ot_flow: Optimal Transport Flow Matching (Lipman et al., 2022)
    
    The model type is auto-detected from:
    1. config.model_name containing keywords
    2. Checkpoint state_dict structure
    """
    
    @staticmethod
    def detect_model_type(config, verbose=False):
        """
        Detect model type from config and/or checkpoint.
        
        Detection order:
        1. Check config.model_name for 'consistency'
        2. Check config.model_name for 'ot_flow' or 'ot-flow' or 'optimal_transport'
        3. Check config.model_name for 'interpolant' or 'flow'
        4. Check config.model_name for 'ddpm' or 'ncsnpp' or 'score'
        5. Check config.model_name for 'triple'
        6. Check checkpoint state_dict for model type keys
        7. Default to 'clean'
        
        Returns:
            str: 'clean', 'triple', 'ddpm', 'interpolant', 'consistency', or 'ot_flow'
        """
        # Method 1: Check model_name in config
        model_name = getattr(config, 'model_name', '').lower()
        
        # Check for consistency models first (most specific)
        if 'consistency' in model_name:
            if verbose:
                print(f"[ModelManager] Detected 'consistency' model from model_name: {model_name}")
            return 'consistency'
        
        # Check for OT flow matching (before generic 'flow')
        if any(x in model_name for x in ['ot_flow', 'ot-flow', 'otflow', 'optimal_transport']):
            if verbose:
                print(f"[ModelManager] Detected 'ot_flow' model from model_name: {model_name}")
            return 'ot_flow'
        
        # Check for interpolant/flow matching
        if any(x in model_name for x in ['interpolant', 'flow']):
            if verbose:
                print(f"[ModelManager] Detected 'interpolant' model from model_name: {model_name}")
            return 'interpolant'
        
        # Check for DDPM/score models
        if any(x in model_name for x in ['ddpm', 'ncsnpp', 'score']):
            if verbose:
                print(f"[ModelManager] Detected 'ddpm' model from model_name: {model_name}")
            return 'ddpm'
        
        if 'triple' in model_name:
            if verbose:
                print(f"[ModelManager] Detected 'triple' model from model_name: {model_name}")
            return 'triple'
        
        # Method 2: Check checkpoint state_dict structure
        if config.best_ckpt and os.path.exists(config.best_ckpt):
            try:
                checkpoint = torch.load(config.best_ckpt, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('state_dict', {})
                hparams = checkpoint.get('hyper_parameters', {})
                
                # Check for consistency model structure
                # Consistency models have 'consistency_model.*' or 'target_model.*' keys
                consistency_keys = [k for k in state_dict.keys() if 'consistency_model' in k or 'target_model' in k]
                if consistency_keys or hparams.get('ct_n_steps') or hparams.get('denoising_warmup_epochs'):
                    if verbose:
                        print(f"[ModelManager] Detected 'consistency' model from checkpoint structure")
                    return 'consistency'
                
                # Check for OT flow model structure
                # OT flow models have 'ot_interpolant.*' keys or 'ot_method' in hparams
                ot_flow_keys = [k for k in state_dict.keys() if 'ot_interpolant' in k]
                if ot_flow_keys or hparams.get('ot_method') or hparams.get('ot_reg'):
                    if verbose:
                        print(f"[ModelManager] Detected 'ot_flow' model from checkpoint structure")
                    return 'ot_flow'
                
                # Check for interpolant model structure
                # Interpolant models have 'interpolant.*' keys or x0_mode in hparams
                interpolant_keys = [k for k in state_dict.keys() if 'interpolant' in k]
                if interpolant_keys or hparams.get('x0_mode'):
                    if verbose:
                        print(f"[ModelManager] Detected 'interpolant' model from checkpoint structure")
                    return 'interpolant'
                
                # Check for DDPM/score_models structure
                # score_models saves 'score_model.*' keys
                ddpm_keys = [k for k in state_dict.keys() if 'score_model' in k and 'model.score_model' not in k]
                if ddpm_keys or hparams.get('sde_type') or hparams.get('sde'):
                    if verbose:
                        print(f"[ModelManager] Detected 'ddpm' model from checkpoint structure")
                    return 'ddpm'
                
                # Triple models have keys like 'model.hydro_dm_model.*', 'model.gas_model.*', 'model.stars_model.*'
                triple_keys = [k for k in state_dict.keys() if any(x in k for x in ['hydro_dm_model', 'gas_model', 'stars_model'])]
                if triple_keys:
                    if verbose:
                        print(f"[ModelManager] Detected 'triple' model from checkpoint structure ({len(triple_keys)} keys)")
                    return 'triple'
            except Exception as e:
                if verbose:
                    print(f"[ModelManager] Could not load checkpoint for model type detection: {e}")
        
        if verbose:
            print(f"[ModelManager] Defaulting to 'clean' model type")
        return 'clean'
    
    @staticmethod
    def initialize(config, verbose=False, skip_data_loading=False):
        """
        Initialize the VDM model and optionally the dataloader.
        
        Supports 'clean', 'triple', 'ddpm', 'interpolant', 'consistency', and 'ot_flow' models.
        Model type is auto-detected from config.model_name or checkpoint structure.
        
        Args:
            config: Configuration object
            verbose: Print debug information
            skip_data_loading: If True, skip dataset loading (faster for inference-only).
                              If False, load the dataset (needed for training/validation).
                              
        Returns:
            tuple: (dataloader_or_None, model)
        """
        # Detect model type
        model_type = ModelManager.detect_model_type(config, verbose=verbose)
        
        if model_type == 'consistency':
            return ModelManager._initialize_consistency(config, verbose=verbose, skip_data_loading=skip_data_loading)
        elif model_type == 'ot_flow':
            return ModelManager._initialize_ot_flow(config, verbose=verbose, skip_data_loading=skip_data_loading)
        elif model_type == 'interpolant':
            return ModelManager._initialize_interpolant(config, verbose=verbose, skip_data_loading=skip_data_loading)
        elif model_type == 'ddpm':
            return ModelManager._initialize_ddpm(config, verbose=verbose, skip_data_loading=skip_data_loading)
        elif model_type == 'triple':
            return ModelManager._initialize_triple(config, verbose=verbose, skip_data_loading=skip_data_loading)
        else:
            return ModelManager._initialize_clean(config, verbose=verbose, skip_data_loading=skip_data_loading)
    
    @staticmethod
    def _initialize_triple(config, verbose=False, skip_data_loading=False):
        """
        Initialize a triple VDM model (3 separate 1-channel models).
        
        The LightTripleVDM requires three UNet score models to be created before loading.
        """
        if verbose:
            print("[ModelManager] Initializing TRIPLE model (3 separate 1-channel models)...")
            print(f"[ModelManager] Using seed: {config.seed}")
            print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")
        
        seed_everything(config.seed)
        
        # Load checkpoint to get hyperparameters
        if not config.best_ckpt or not os.path.exists(config.best_ckpt):
            raise ValueError(f"Triple model requires a valid checkpoint. Got: {config.best_ckpt}")
        
        checkpoint = torch.load(config.best_ckpt, map_location='cuda', weights_only=False)
        hparams = checkpoint.get('hyper_parameters', {})
        
        if verbose:
            print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
            for k, v in hparams.items():
                print(f"  {k}: {v}")
        
        # Auto-detect conditioning channels from checkpoint
        state_dict = checkpoint.get('state_dict', {})
        
        # Find a conv_in key to detect channels (triple models have per-model UNets)
        conv_in_keys = [k for k in state_dict.keys() if 'conv_in.weight' in k]
        if conv_in_keys:
            first_conv_in = state_dict[conv_in_keys[0]]
            total_in_channels = first_conv_in.shape[1]
            input_channels = 1  # Triple models use 1-channel input
            
            # Check for Fourier features - three modes:
            # 1. Legacy: fourier_features.freqs (exponential frequencies, first channel only)
            # 2. New multi-scale: fourier_features_halo.frequencies + fourier_features_largescale.frequencies
            # 3. None: no Fourier features
            legacy_keys = [k for k in state_dict.keys() if 'fourier_features.freqs' in k]
            new_multiscale_keys = [k for k in state_dict.keys() if 'fourier_features_halo.frequencies' in k or 'fourier_features_largescale.frequencies' in k]
            
            if legacy_keys:
                config.fourier_legacy = True
                config.use_fourier_features = True
            elif new_multiscale_keys:
                config.fourier_legacy = False
                config.use_fourier_features = True
            else:
                config.fourier_legacy = False
                config.use_fourier_features = False
            
            if verbose:
                print(f"[ModelManager] Triple model Fourier detection:")
                print(f"  Legacy keys found: {len(legacy_keys)}")
                print(f"  New multi-scale keys found: {len(new_multiscale_keys)}")
                print(f"  fourier_legacy: {config.fourier_legacy}")
                print(f"  use_fourier_features: {config.use_fourier_features}")
            
            # Calculate large_scale_channels based on conv_in shape and Fourier mode
            if config.use_fourier_features:
                if config.fourier_legacy:
                    # Legacy: input + conditioning + 4*fourier + large_scale
                    fourier_channels = 4
                    base_channels = input_channels + 1 + fourier_channels  # 1=conditioning
                    config.large_scale_channels = total_in_channels - base_channels
                    config.conditioning_channels = 1
                else:
                    # New multi-scale: 
                    # total = input + conditioning*(1+8) + large_scale*(1+8)
                    # total = 1 + 1*9 + N*9 = 10 + 9*N
                    # => N = (total - 10) / 9
                    config.conditioning_channels = 1
                    config.large_scale_channels = (total_in_channels - 10) // 9
            else:
                # No Fourier: input + conditioning + large_scale
                config.conditioning_channels = 1
                config.large_scale_channels = total_in_channels - input_channels - 1
            
            if verbose:
                print(f"[ModelManager] Triple model channel configuration:")
                print(f"  Total conv_in channels: {total_in_channels}")
                print(f"  conditioning_channels: {config.conditioning_channels}")
                print(f"  large_scale_channels: {config.large_scale_channels}")
        
        # Get UNet parameters from hparams
        unet_params = {
            'input_channels': 1,  # Triple models use 1-channel input
            'conditioning_channels': config.conditioning_channels,
            'large_scale_channels': config.large_scale_channels,
            'gamma_min': hparams.get('gamma_min', config.gamma_min),
            'gamma_max': hparams.get('gamma_max', config.gamma_max),
            'embedding_dim': getattr(config, 'embedding_dim', 256),
            'norm_groups': getattr(config, 'norm_groups', 32),
            'n_blocks': getattr(config, 'n_blocks', 4),
            'add_attention': getattr(config, 'add_attention', True),
            'n_attention_heads': getattr(config, 'n_attention_heads', 8),
            'use_fourier_features': config.use_fourier_features,
            'legacy_fourier': config.fourier_legacy,
            # Parameter conditioning - check config for param_norm_path
            'use_param_conditioning': getattr(config, 'use_param_conditioning', False),
            'param_min': getattr(config, 'min', None),
            'param_max': getattr(config, 'max', None),
            # Cross-attention (optional)
            'use_cross_attention': getattr(config, 'use_cross_attention', False),
            'cross_attention_location': getattr(config, 'cross_attention_location', 'bottleneck'),
            'cross_attention_heads': getattr(config, 'cross_attention_heads', 8),
            'cross_attention_dropout': getattr(config, 'cross_attention_dropout', 0.1),
            'use_chunked_cross_attention': getattr(config, 'use_chunked_cross_attention', True),
            'cross_attention_chunk_size': getattr(config, 'cross_attention_chunk_size', 512),
            'downsample_cross_attn_cond': getattr(config, 'downsample_cross_attn_cond', False),
            'cross_attn_cond_downsample_factor': getattr(config, 'cross_attn_cond_downsample_factor', 2),
        }
        
        if verbose:
            print(f"[ModelManager] Creating three UNet score models with params:")
            for k, v in unet_params.items():
                print(f"  {k}: {v}")
        
        # Create three separate UNet score models
        hydro_dm_score_model = networks.UNetVDM(**unet_params)
        gas_score_model = networks.UNetVDM(**unet_params)
        stars_score_model = networks.UNetVDM(**unet_params)
        
        if verbose:
            print(f"[ModelManager] Created three UNet models. Creating LightTripleVDM...")
        
        # Build LightTripleVDM constructor arguments from hparams
        triple_params = {
            'hydro_dm_score_model': hydro_dm_score_model,
            'gas_score_model': gas_score_model,
            'stars_score_model': stars_score_model,
            'learning_rate': hparams.get('learning_rate', getattr(config, 'learning_rate', 3e-4)),
            'lr_scheduler': hparams.get('lr_scheduler', getattr(config, 'lr_scheduler', 'cosine')),
            'noise_schedule': hparams.get('noise_schedule', getattr(config, 'noise_schedule', 'fixed_linear')),
            'gamma_min': hparams.get('gamma_min', config.gamma_min),
            'gamma_max': hparams.get('gamma_max', config.gamma_max),
            'image_shape': hparams.get('image_shape', (1, config.cropsize, config.cropsize)),
            'data_noise': hparams.get('data_noise', getattr(config, 'data_noise', 1e-3)),
            'antithetic_time_sampling': hparams.get('antithetic_time_sampling', getattr(config, 'antithetic_time_sampling', True)),
            # Channel weights
            'channel_weights': hparams.get('channel_weights', (1.0, 1.0, 1.0)),
            # Focal loss settings (typically only for stars)
            'use_focal_loss_hydro_dm': hparams.get('use_focal_loss_hydro_dm', False),
            'use_focal_loss_gas': hparams.get('use_focal_loss_gas', False),
            'use_focal_loss_stars': hparams.get('use_focal_loss_stars', getattr(config, 'use_focal_loss', False)),
            'focal_gamma': hparams.get('focal_gamma', getattr(config, 'focal_gamma', 2.0)),
            # Parameter prediction
            'use_param_prediction': hparams.get('use_param_prediction', getattr(config, 'use_param_prediction', False)),
            'param_prediction_weight': hparams.get('param_prediction_weight', getattr(config, 'param_prediction_weight', 0.01)),
        }
        
        # Create the triple VDM model
        vdm_model = vdm_triple_module.LightTripleVDM(**triple_params)
        
        if verbose:
            print(f"[ModelManager] Loading state dict into LightTripleVDM...")
        
        # Load state dict (handle missing keys like we do for clean models)
        model_state = vdm_model.state_dict()
        missing_keys = [k for k in model_state.keys() if k not in state_dict]
        if missing_keys and verbose:
            print(f"[ModelManager] Warning: checkpoint is missing {len(missing_keys)} keys. Injecting defaults.")
        for k in missing_keys:
            state_dict[k] = model_state[k]
        
        vdm_model.load_state_dict(state_dict)
        vdm_model = vdm_model.eval()
        
        if verbose:
            print("[ModelManager] Triple model loaded successfully.")
            print(f"[ModelManager] Model checkpoint: {config.best_ckpt}")
            print(f"[ModelManager] Using {config.conditioning_channels} conditioning channel(s)")
            print(f"[ModelManager] Using {config.large_scale_channels} large-scale channel(s)")
            print(f"[ModelManager] Fourier features: {'LEGACY' if config.fourier_legacy else 'MULTI-SCALE' if config.use_fourier_features else 'DISABLED'}")
        
        # Load data if requested
        if skip_data_loading:
            hydro = None
        else:
            if not config.train_samples:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test/'
            else:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/'
            hydro = get_astro_data(
                config.dataset,
                test_root,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                stage='test',
                quantile_path=getattr(config, 'quantile_path', None)
            )
            if verbose:
                print(f"[ModelManager] Dataset loaded.")
        
        return hydro, vdm_model
    
    @staticmethod
    def _initialize_ddpm(config, verbose=False, skip_data_loading=False):
        """
        Initialize a DDPM/score_models model (NCSNpp or DDPM architecture).
        
        These models use the score_models package and Denoising Score Matching loss.
        Uses the DIRECT score_models approach for proper weight loading.
        """
        if verbose:
            print("[ModelManager] Initializing DDPM/Score Model (direct score_models approach)...")
            print(f"[ModelManager] Using seed: {config.seed}")
            print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")
        
        seed_everything(config.seed)
        
        # Import score_models directly (NOT LightScoreModel wrapper)
        try:
            from score_models import ScoreModel, NCSNpp, DDPM
            SCORE_MODELS_AVAILABLE = True
        except ImportError as e:
            raise ImportError(f"DDPM model requires score_models package: {e}")
        
        # Load checkpoint to get hyperparameters
        if not config.best_ckpt or not os.path.exists(config.best_ckpt):
            raise ValueError(f"DDPM model requires a valid checkpoint. Got: {config.best_ckpt}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
        hparams = checkpoint.get('hyper_parameters', {})
        state_dict = checkpoint.get('state_dict', {})
        
        if verbose:
            print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
            for k, v in list(hparams.items())[:20]:
                print(f"  {k}: {v}")
        
        # Extract model configuration from hparams
        sde_type = hparams.get('sde_type', hparams.get('sde', 'vp')).lower()
        beta_min = hparams.get('beta_min', 0.1)
        beta_max = hparams.get('beta_max', 20.0)
        sigma_min = hparams.get('sigma_min', 0.01)
        sigma_max = hparams.get('sigma_max', 50.0)
        n_sampling_steps = hparams.get('n_sampling_steps', 1000)
        use_param_conditioning = hparams.get('use_param_conditioning', False)
        
        # Auto-detect conditioning channels from config
        conditioning_channels = getattr(config, 'conditioning_channels', 1)
        large_scale_channels = getattr(config, 'large_scale_channels', 3)
        n_params = getattr(config, 'n_params', 35) if use_param_conditioning else 0
        
        # Total spatial conditioning = 1 (DM) + large_scale_channels
        total_spatial_cond = 1 + large_scale_channels
        output_channels = 3  # [DM, Gas, Stars]
        
        # Build condition list based on what was used in training
        if use_param_conditioning:
            condition_types = ['input', 'vector']
        else:
            condition_types = ['input']
        
        # Get NCSNpp specific params from hparams or config
        nf = hparams.get('nf', getattr(config, 'nf', 96))
        ch_mult_str = hparams.get('ch_mult', getattr(config, 'ch_mult', '1,2,4,8'))
        if isinstance(ch_mult_str, str):
            ch_mult = tuple(map(int, ch_mult_str.split(',')))
        else:
            ch_mult = tuple(ch_mult_str) if hasattr(ch_mult_str, '__iter__') else (1, 2, 4, 8)
        
        if verbose:
            print(f"[ModelManager] DDPM Model Configuration:")
            print(f"  SDE type: {sde_type}")
            print(f"  nf: {nf}, ch_mult: {ch_mult}")
            print(f"  Spatial conditioning channels: {total_spatial_cond}")
            print(f"  Parameter conditioning: {use_param_conditioning} ({n_params} params)")
            print(f"  Sampling steps: {n_sampling_steps}")
        
        # Create NCSNpp network with same architecture as training
        net_kwargs = {
            'channels': output_channels,
            'dimensions': 2,
            'nf': nf,
            'ch_mult': ch_mult,
            'attention': hparams.get('attention', True),
            'condition': condition_types,
            'condition_input_channels': total_spatial_cond,
        }
        if use_param_conditioning:
            net_kwargs['condition_vector_channels'] = n_params
        
        net = NCSNpp(**net_kwargs)
        
        # Create ScoreModel directly using original score_models interface
        # This is the KEY difference - use score_models.ScoreModel directly
        if sde_type == 'vp':
            score_model = ScoreModel(
                model=net,
                sde='vp',
                beta_min=beta_min,
                beta_max=beta_max,
                T=1.0,
                epsilon=1e-5,
                device=device,
            )
        else:  # ve
            score_model = ScoreModel(
                model=net,
                sde='ve',
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                T=1.0,
                device=device,
            )
        
        if verbose:
            print(f"[ModelManager] Created ScoreModel with {sde_type.upper()}-SDE")
            print(f"[ModelManager] Loading weights from Lightning checkpoint...")
        
        # Extract network weights from Lightning checkpoint
        # Lightning saves as: 'score_model.model.xxx' or 'model.xxx'
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith('score_model.model.'):
                # Strip 'score_model.model.' prefix to get raw NCSNpp keys
                new_key = k.replace('score_model.model.', '')
                model_state[new_key] = v
            elif k.startswith('model.score_model.model.'):
                # Alternative Lightning format
                new_key = k.replace('model.score_model.model.', '')
                model_state[new_key] = v
            elif k.startswith('model.'):
                # Fallback: 'model.xxx' -> 'xxx'
                new_key = k.replace('model.', '')
                model_state[new_key] = v
        
        if len(model_state) == 0:
            raise ValueError(
                f"Could not extract model weights from checkpoint. "
                f"State dict keys: {list(state_dict.keys())[:10]}..."
            )
        
        # Load weights into the NCSNpp network
        missing, unexpected = score_model.model.load_state_dict(model_state, strict=False)
        
        if verbose:
            print(f"[ModelManager] Loaded {len(model_state)} weight tensors")
            if missing:
                print(f"[ModelManager] Missing keys: {missing[:5]}..." if len(missing) > 5 else f"[ModelManager] Missing keys: {missing}")
            if unexpected:
                print(f"[ModelManager] Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"[ModelManager] Unexpected keys: {unexpected}")
        
        score_model.model.eval()
        score_model.model.to(device)
        
        # Wrap in a simple class that provides BIND-compatible interface
        class DDPMModelWrapper:
            """Wrapper that provides BIND-compatible interface for score_models.ScoreModel."""
            
            def __init__(self, score_model, n_sampling_steps, use_param_conditioning, hparams):
                self.score_model = score_model
                self.n_sampling_steps = n_sampling_steps
                self.use_param_conditioning = use_param_conditioning
                # Store hparams as namespace for compatibility
                self.hparams = type('HParams', (), hparams)()
                self.hparams.n_sampling_steps = n_sampling_steps
                self._device = device
            
            def to(self, device):
                """Move model to device."""
                self._device = device
                self.score_model.model.to(device)
                return self
            
            def eval(self):
                """Set model to eval mode."""
                self.score_model.model.eval()
                return self
            
            def train(self, mode=True):
                """Set model to train mode."""
                self.score_model.model.train(mode)
                return self
            
            def parameters(self):
                """Return model parameters."""
                return self.score_model.model.parameters()
            
            def draw_samples(
                self,
                conditioning,
                batch_size,
                n_sampling_steps=None,
                param_conditioning=None,
                verbose=False,
            ):
                """
                BIND-compatible sampling interface.
                
                Args:
                    conditioning: Spatial conditioning tensor (B, C_cond, H, W)
                    batch_size: Number of samples to generate
                    n_sampling_steps: Sampling steps (default: self.n_sampling_steps)
                    param_conditioning: Optional parameter conditioning (B, N_params)
                    verbose: Show progress
                
                Returns:
                    samples: Generated samples (B, 3, H, W)
                """
                B, C_cond, H, W = conditioning.shape
                steps = n_sampling_steps or self.n_sampling_steps
                
                # Build condition list for score_models.sample()
                # Format: [spatial_cond, (optional) vector_cond]
                condition_list = [conditioning.to(self._device)]
                if param_conditioning is not None:
                    condition_list.append(param_conditioning.to(self._device))
                
                # Generate samples using original score_models interface
                with torch.no_grad():
                    samples = self.score_model.sample(
                        shape=[batch_size, 3, H, W],
                        steps=steps,
                        condition=condition_list,
                    )
                
                return samples
        
        # Create wrapped model
        model = DDPMModelWrapper(
            score_model=score_model,
            n_sampling_steps=n_sampling_steps,
            use_param_conditioning=use_param_conditioning,
            hparams=dict(hparams),
        )
        
        if verbose:
            n_params_total = sum(p.numel() for p in score_model.model.parameters())
            print(f"[ModelManager] ✓ DDPM model loaded successfully")
            print(f"[ModelManager] Model parameters: {n_params_total:,}")
            print(f"[ModelManager] Sampling steps: {n_sampling_steps}")
        
        # Load data if requested
        if skip_data_loading:
            hydro = None
        else:
            if not config.train_samples:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test/'
            else:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/'
            hydro = get_astro_data(
                config.dataset,
                test_root,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                stage='test',
                quantile_path=getattr(config, 'quantile_path', None)
            )
            if verbose:
                print(f"[ModelManager] Dataset loaded.")
        
        return hydro, model
    
    @staticmethod
    def _initialize_interpolant(config, verbose=False, skip_data_loading=False):
        """
        Initialize an Interpolant/Flow Matching model.
        
        These models use flow matching for the DMO -> Hydro mapping.
        """
        if verbose:
            print("[ModelManager] Initializing INTERPOLANT/Flow Matching model...")
            print(f"[ModelManager] Using seed: {config.seed}")
            print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")
        
        seed_everything(config.seed)
        
        # Import interpolant module
        from vdm.interpolant_model import LightInterpolant, VelocityNetWrapper
        
        # Load checkpoint to get hyperparameters
        if not config.best_ckpt or not os.path.exists(config.best_ckpt):
            raise ValueError(f"Interpolant model requires a valid checkpoint. Got: {config.best_ckpt}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
        hparams = checkpoint.get('hyper_parameters', {})
        state_dict = checkpoint.get('state_dict', {})
        
        if verbose:
            print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
            for k, v in list(hparams.items())[:15]:
                print(f"  {k}: {v}")
        
        # Extract model configuration from hparams
        n_sampling_steps = hparams.get('n_sampling_steps', 50)
        x0_mode = hparams.get('x0_mode', 'zeros')
        use_stochastic_interpolant = hparams.get('use_stochastic_interpolant', False)
        sigma = hparams.get('sigma', 0.0)
        learning_rate = hparams.get('learning_rate', 1e-4)
        
        # Auto-detect conditioning channels from config
        conditioning_channels = getattr(config, 'conditioning_channels', 1)
        large_scale_channels = getattr(config, 'large_scale_channels', 3)
        total_conditioning_channels = conditioning_channels + large_scale_channels
        output_channels = 3  # [DM, Gas, Stars]
        
        # Get UNet parameters from config/hparams
        embedding_dim = getattr(config, 'embedding_dim', 256)
        n_blocks = getattr(config, 'n_blocks', 32)
        norm_groups = getattr(config, 'norm_groups', 8)
        n_attention_heads = getattr(config, 'n_attention_heads', 8)
        use_fourier_features = getattr(config, 'use_fourier_features', True)
        fourier_legacy = getattr(config, 'fourier_legacy', False)
        add_attention = getattr(config, 'add_attention', True)
        use_param_conditioning = hparams.get('use_param_conditioning', getattr(config, 'use_param_conditioning', True))
        
        if verbose:
            print(f"[ModelManager] Interpolant Model Configuration:")
            print(f"  x0 mode: {x0_mode}")
            print(f"  Stochastic: {use_stochastic_interpolant} (sigma={sigma})")
            print(f"  Sampling steps: {n_sampling_steps}")
            print(f"  Conditioning channels: {total_conditioning_channels}")
            print(f"  Param conditioning: {use_param_conditioning}")
        
        # Create UNetVDM - same architecture as train_interpolant.py
        # UNetVDM takes input_channels (target) and conditioning_channels separately
        # It concatenates them internally and handles Fourier features
        from vdm.networks_clean import UNetVDM
        
        # Load param normalization if using param conditioning
        param_min = None
        param_max = None
        if use_param_conditioning:
            param_norm_path = hparams.get('param_norm_path', getattr(config, 'param_norm_path', None))
            if param_norm_path and os.path.exists(param_norm_path):
                import pandas as pd
                minmax_df = pd.read_csv(param_norm_path)
                param_min = np.array(minmax_df['MinVal'])
                param_max = np.array(minmax_df['MaxVal'])
                if verbose:
                    print(f"[ModelManager] Loaded {len(param_min)} param bounds from {param_norm_path}")
        
        unet = UNetVDM(
            input_channels=output_channels,  # Target channels (3)
            conditioning_channels=conditioning_channels,  # DM channels (1)
            large_scale_channels=large_scale_channels,  # Large-scale context (3)
            embedding_dim=embedding_dim,
            n_blocks=n_blocks,
            norm_groups=norm_groups,
            n_attention_heads=n_attention_heads,
            use_fourier_features=use_fourier_features,
            legacy_fourier=fourier_legacy,
            add_attention=add_attention,
            use_param_conditioning=use_param_conditioning,
            param_min=param_min,
            param_max=param_max,
        )
        
        # Wrap for velocity prediction
        velocity_model = VelocityNetWrapper(
            net=unet,
            output_channels=output_channels,
            conditioning_channels=total_conditioning_channels,
        )
        
        # Create LightInterpolant
        model = LightInterpolant(
            velocity_model=velocity_model,
            learning_rate=learning_rate,
            n_sampling_steps=n_sampling_steps,
            use_stochastic_interpolant=use_stochastic_interpolant,
            sigma=sigma,
            x0_mode=x0_mode,
            use_param_conditioning=use_param_conditioning,
        )
        
        if verbose:
            print(f"[ModelManager] Loading state dict into LightInterpolant...")
        
        # Load state dict
        model.load_state_dict(state_dict)
        model = model.eval().to(device)
        
        if verbose:
            n_params_total = sum(p.numel() for p in model.parameters())
            print(f"[ModelManager] ✓ Interpolant model loaded successfully")
            print(f"[ModelManager] Model parameters: {n_params_total:,}")
            print(f"[ModelManager] Sampling steps: {n_sampling_steps}")
        
        # Load data if requested
        if skip_data_loading:
            hydro = None
        else:
            if not config.train_samples:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test/'
            else:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/'
            hydro = get_astro_data(
                config.dataset,
                test_root,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                stage='test',
                quantile_path=getattr(config, 'quantile_path', None)
            )
            if verbose:
                print(f"[ModelManager] Dataset loaded.")
        
        return hydro, model

    @staticmethod
    def _initialize_consistency(config, verbose=False, skip_data_loading=False):
        """
        Initialize a Consistency Model (Song et al., 2023).
        
        Consistency models enable single-step or few-step high-quality sampling
        by learning to map any point on the diffusion trajectory directly to clean data.
        """
        if verbose:
            print("[ModelManager] Initializing CONSISTENCY model (single/few-step sampling)...")
            print(f"[ModelManager] Using seed: {config.seed}")
            print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")
        
        seed_everything(config.seed)
        
        # Import consistency module
        from vdm.consistency_model import (
            LightConsistency, ConsistencyModel, ConsistencyFunction,
            ConsistencyNoiseSchedule, ConsistencyNetWrapper
        )
        
        # Load checkpoint to get hyperparameters
        if not config.best_ckpt or not os.path.exists(config.best_ckpt):
            raise ValueError(f"Consistency model requires a valid checkpoint. Got: {config.best_ckpt}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
        hparams = checkpoint.get('hyper_parameters', {})
        state_dict = checkpoint.get('state_dict', {})
        
        if verbose:
            print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
            for k, v in list(hparams.items())[:15]:
                print(f"  {k}: {v}")
        
        # Extract model configuration from hparams
        n_sampling_steps = hparams.get('n_sampling_steps', 1)
        ct_n_steps = hparams.get('ct_n_steps', 18)
        x0_mode = hparams.get('x0_mode', 'zeros')
        sigma_min = hparams.get('sigma_min', 0.002)
        sigma_max = hparams.get('sigma_max', 80.0)
        sigma_data = hparams.get('sigma_data', 0.5)
        learning_rate = hparams.get('learning_rate', 1e-4)
        
        # Auto-detect conditioning channels from config
        conditioning_channels = getattr(config, 'conditioning_channels', 1)
        large_scale_channels = getattr(config, 'large_scale_channels', 3)
        total_conditioning_channels = conditioning_channels + large_scale_channels
        output_channels = 3  # [DM, Gas, Stars]
        
        # Get UNet parameters from config/hparams
        embedding_dim = getattr(config, 'embedding_dim', 256)
        n_blocks = getattr(config, 'n_blocks', 32)
        norm_groups = getattr(config, 'norm_groups', 8)
        n_attention_heads = getattr(config, 'n_attention_heads', 8)
        use_fourier_features = getattr(config, 'use_fourier_features', True)
        fourier_legacy = getattr(config, 'fourier_legacy', False)
        add_attention = getattr(config, 'add_attention', True)
        use_param_conditioning = hparams.get('use_param_conditioning', getattr(config, 'use_param_conditioning', True))
        
        if verbose:
            print(f"[ModelManager] Consistency Model Configuration:")
            print(f"  Sampling steps: {n_sampling_steps}")
            print(f"  CT discretization steps: {ct_n_steps}")
            print(f"  Sigma range: [{sigma_min}, {sigma_max}]")
            print(f"  Conditioning channels: {total_conditioning_channels}")
            print(f"  Param conditioning: {use_param_conditioning}")
        
        # Create UNetVDM - same architecture as train_consistency.py
        from vdm.networks_clean import UNetVDM
        
        # Load param normalization if using param conditioning
        param_min = None
        param_max = None
        if use_param_conditioning:
            param_norm_path = hparams.get('param_norm_path', getattr(config, 'param_norm_path', None))
            if param_norm_path and os.path.exists(param_norm_path):
                import pandas as pd
                minmax_df = pd.read_csv(param_norm_path)
                param_min = np.array(minmax_df['MinVal'])
                param_max = np.array(minmax_df['MaxVal'])
                if verbose:
                    print(f"[ModelManager] Loaded {len(param_min)} param bounds from {param_norm_path}")
        
        unet = UNetVDM(
            input_channels=output_channels,  # Target channels (3)
            conditioning_channels=conditioning_channels,  # DM channels (1)
            large_scale_channels=large_scale_channels,  # Large-scale context (3)
            embedding_dim=embedding_dim,
            n_blocks=n_blocks,
            norm_groups=norm_groups,
            n_attention_heads=n_attention_heads,
            use_fourier_features=use_fourier_features,
            legacy_fourier=fourier_legacy,
            add_attention=add_attention,
            use_param_conditioning=use_param_conditioning,
            param_min=param_min,
            param_max=param_max,
        )
        
        # Wrap for consistency prediction
        net_wrapper = ConsistencyNetWrapper(
            net=unet,
            output_channels=output_channels,
            conditioning_channels=total_conditioning_channels,
        )
        
        # Create consistency function and noise schedule
        noise_schedule = ConsistencyNoiseSchedule(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        
        consistency_fn = ConsistencyFunction(
            net=net_wrapper,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
        )
        
        # Create consistency model core
        consistency_model = ConsistencyModel(
            consistency_fn=consistency_fn,
            noise_schedule=noise_schedule,
            sigma_data=sigma_data,
        )
        
        # Create LightConsistency
        model = LightConsistency(
            consistency_model=consistency_model,
            learning_rate=learning_rate,
            n_sampling_steps=n_sampling_steps,
            x0_mode=x0_mode,
            use_param_conditioning=use_param_conditioning,
            ct_n_steps=ct_n_steps,
        )
        
        if verbose:
            print(f"[ModelManager] Loading state dict into LightConsistency...")
        
        # Load state dict
        model.load_state_dict(state_dict)
        model = model.eval().to(device)
        
        if verbose:
            n_params_total = sum(p.numel() for p in model.parameters())
            print(f"[ModelManager] ✓ Consistency model loaded successfully")
            print(f"[ModelManager] Model parameters: {n_params_total:,}")
            print(f"[ModelManager] Sampling steps: {n_sampling_steps}")
        
        # Load data if requested
        if skip_data_loading:
            hydro = None
        else:
            if not config.train_samples:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test/'
            else:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/'
            hydro = get_astro_data(
                config.dataset,
                test_root,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                stage='test',
                quantile_path=getattr(config, 'quantile_path', None)
            )
            if verbose:
                print(f"[ModelManager] Dataset loaded.")
        
        return hydro, model

    @staticmethod
    def _initialize_ot_flow(config, verbose=False, skip_data_loading=False):
        """
        Initialize an Optimal Transport Flow Matching model (Lipman et al., 2022).
        
        OT flow matching uses optimal transport coupling for straighter interpolation paths.
        """
        if verbose:
            print("[ModelManager] Initializing OT FLOW MATCHING model...")
            print(f"[ModelManager] Using seed: {config.seed}")
            print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")
        
        seed_everything(config.seed)
        
        # Import OT flow module
        from vdm.ot_flow_model import LightOTFlow, OTVelocityNetWrapper
        
        # Load checkpoint to get hyperparameters
        if not config.best_ckpt or not os.path.exists(config.best_ckpt):
            raise ValueError(f"OT Flow model requires a valid checkpoint. Got: {config.best_ckpt}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
        hparams = checkpoint.get('hyper_parameters', {})
        state_dict = checkpoint.get('state_dict', {})
        
        if verbose:
            print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
            for k, v in list(hparams.items())[:15]:
                print(f"  {k}: {v}")
        
        # Extract model configuration from hparams
        n_sampling_steps = hparams.get('n_sampling_steps', 50)
        x0_mode = hparams.get('x0_mode', 'zeros')
        ot_method = hparams.get('ot_method', 'exact')
        ot_reg = hparams.get('ot_reg', 0.01)
        use_stochastic_interpolant = hparams.get('use_stochastic_interpolant', False)
        sigma = hparams.get('sigma', 0.0)
        learning_rate = hparams.get('learning_rate', 1e-4)
        
        # Auto-detect conditioning channels from config
        conditioning_channels = getattr(config, 'conditioning_channels', 1)
        large_scale_channels = getattr(config, 'large_scale_channels', 3)
        total_conditioning_channels = conditioning_channels + large_scale_channels
        output_channels = 3  # [DM, Gas, Stars]
        
        # Get UNet parameters from config/hparams
        embedding_dim = getattr(config, 'embedding_dim', 256)
        n_blocks = getattr(config, 'n_blocks', 32)
        norm_groups = getattr(config, 'norm_groups', 8)
        n_attention_heads = getattr(config, 'n_attention_heads', 8)
        use_fourier_features = getattr(config, 'use_fourier_features', True)
        fourier_legacy = getattr(config, 'fourier_legacy', False)
        add_attention = getattr(config, 'add_attention', True)
        use_param_conditioning = hparams.get('use_param_conditioning', getattr(config, 'use_param_conditioning', True))
        
        if verbose:
            print(f"[ModelManager] OT Flow Model Configuration:")
            print(f"  x0 mode: {x0_mode}")
            print(f"  OT method: {ot_method}")
            print(f"  OT regularization: {ot_reg}")
            print(f"  Stochastic: {use_stochastic_interpolant} (sigma={sigma})")
            print(f"  Sampling steps: {n_sampling_steps}")
            print(f"  Conditioning channels: {total_conditioning_channels}")
            print(f"  Param conditioning: {use_param_conditioning}")
        
        # Create UNetVDM - same architecture as train_ot_flow.py
        from vdm.networks_clean import UNetVDM
        
        # Load param normalization if using param conditioning
        param_min = None
        param_max = None
        if use_param_conditioning:
            param_norm_path = hparams.get('param_norm_path', getattr(config, 'param_norm_path', None))
            if param_norm_path and os.path.exists(param_norm_path):
                import pandas as pd
                minmax_df = pd.read_csv(param_norm_path)
                param_min = np.array(minmax_df['MinVal'])
                param_max = np.array(minmax_df['MaxVal'])
                if verbose:
                    print(f"[ModelManager] Loaded {len(param_min)} param bounds from {param_norm_path}")
        
        unet = UNetVDM(
            input_channels=output_channels,  # Target channels (3)
            conditioning_channels=conditioning_channels,  # DM channels (1)
            large_scale_channels=large_scale_channels,  # Large-scale context (3)
            embedding_dim=embedding_dim,
            n_blocks=n_blocks,
            norm_groups=norm_groups,
            n_attention_heads=n_attention_heads,
            use_fourier_features=use_fourier_features,
            legacy_fourier=fourier_legacy,
            add_attention=add_attention,
            use_param_conditioning=use_param_conditioning,
            param_min=param_min,
            param_max=param_max,
        )
        
        # Wrap for velocity prediction
        velocity_model = OTVelocityNetWrapper(
            net=unet,
            output_channels=output_channels,
            conditioning_channels=total_conditioning_channels,
        )
        
        # Create LightOTFlow
        model = LightOTFlow(
            velocity_model=velocity_model,
            learning_rate=learning_rate,
            n_sampling_steps=n_sampling_steps,
            use_stochastic_interpolant=use_stochastic_interpolant,
            sigma=sigma,
            x0_mode=x0_mode,
            use_param_conditioning=use_param_conditioning,
            ot_method=ot_method,
            ot_reg=ot_reg,
        )
        
        if verbose:
            print(f"[ModelManager] Loading state dict into LightOTFlow...")
        
        # Load state dict
        model.load_state_dict(state_dict)
        model = model.eval().to(device)
        
        if verbose:
            n_params_total = sum(p.numel() for p in model.parameters())
            print(f"[ModelManager] ✓ OT Flow model loaded successfully")
            print(f"[ModelManager] Model parameters: {n_params_total:,}")
            print(f"[ModelManager] Sampling steps: {n_sampling_steps}")
        
        # Load data if requested
        if skip_data_loading:
            hydro = None
        else:
            if not config.train_samples:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test/'
            else:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/'
            hydro = get_astro_data(
                config.dataset,
                test_root,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                stage='test',
                quantile_path=getattr(config, 'quantile_path', None)
            )
            if verbose:
                print(f"[ModelManager] Dataset loaded.")
        
        return hydro, model

    @staticmethod
    def _initialize_clean(config, verbose=False, skip_data_loading=False):
        """
        Initialize a clean VDM model (single 3-channel model).
        
        This is the original ModelManager.initialize() implementation.
        """
        if getattr(config, 'verbose', False) or verbose:
            print("[ModelManager] Initializing CLEAN model (single 3-channel model)...")
            print(f"[ModelManager] Using seed: {config.seed}")
            print(f"[ModelManager] Dataset: {config.dataset}")
            print(f"[ModelManager] Data root: {config.data_root}")
            print(f"[ModelManager] Batch size: {config.batch_size}")
            if skip_data_loading:
                print("[ModelManager] ⚠️  Skipping data loading (using pre-loaded samples)")
            print(f"[ModelManager] BEFORE auto-detect: conditioning_channels = {getattr(config, 'conditioning_channels', 'NOT SET')}")
            print(f"[ModelManager] BEFORE auto-detect: large_scale_channels = {getattr(config, 'large_scale_channels', 'NOT SET')}")
            print(f"[ModelManager] BEFORE auto-detect: use_fourier_features = {getattr(config, 'use_fourier_features', 'NOT SET')}")
            print(f"[ModelManager] BEFORE auto-detect: fourier_legacy = {getattr(config, 'fourier_legacy', 'NOT SET')}")
            print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")
        
        # Auto-detect Fourier settings and conditioning channels from checkpoint if checkpoint exists
        if config.best_ckpt is not None:
            try:
                state_dict = torch.load(config.best_ckpt, map_location='cuda', weights_only=False)["state_dict"]
                
                # Check for cross-attention (separate conditioning path)
                cross_attn_keys = [k for k in state_dict.keys() if 'cross_attn' in k or 'mid_cross' in k]
                has_cross_attention = len(cross_attn_keys) > 0
                
                if has_cross_attention and verbose:
                    print(f"[ModelManager] Detected cross-attention model from checkpoint")
                
                # For cross-attention models, conditioning channels come from K/V projection, not conv_in
                if has_cross_attention:
                    # Find a cross-attention K or V projection to get conditioning channels
                    kv_keys = [k for k in state_dict.keys() if ('to_k.weight' in k or 'to_v.weight' in k) and 'cross' in k]
                    if kv_keys:
                        kv_shape = state_dict[kv_keys[0]].shape
                        # Shape is [embed_dim, cond_channels, 1, 1] for spatial cross-attention
                        total_cond_channels = kv_shape[1]
                        
                        # Total conditioning = base DM (1) + large_scale (N)
                        config.conditioning_channels = 1
                        config.large_scale_channels = total_cond_channels - 1
                        
                        if verbose:
                            print(f"[ModelManager] Cross-attention conditioning channel breakdown:")
                            print(f"  K/V projection input: {total_cond_channels}")
                            print(f"  = conditioning({config.conditioning_channels}) + large_scale({config.large_scale_channels})")
                        
                        # For cross-attention, conv_in only sees target channels (not conditioning)
                        # So skip the normal auto-detection logic below
                        has_cross_attention = True  # Mark for skipping normal logic
                    else:
                        if verbose:
                            print(f"[ModelManager] Warning: Cross-attention detected but no K/V projections found")
                        has_cross_attention = False
                else:
                    has_cross_attention = False
                
                # Check if fourier features are present in the checkpoint
                has_legacy_fourier = 'model.score_model.fourier_features.freqs_exponent' in state_dict
                has_new_fourier_halo = 'model.score_model.fourier_features_halo.frequencies' in state_dict
                has_new_fourier_largescale = 'model.score_model.fourier_features_largescale.frequencies' in state_dict
                
                # Auto-detect fourier_legacy flag ONLY if not explicitly set in config
                if config.fourier_legacy is None:
                    if has_legacy_fourier:
                        config.fourier_legacy = True
                        if verbose:
                            print(f"[ModelManager] Auto-detected LEGACY Fourier features from checkpoint")
                    elif has_new_fourier_halo or has_new_fourier_largescale:
                        config.fourier_legacy = False
                        if verbose:
                            print(f"[ModelManager] Auto-detected NEW multi-scale Fourier features from checkpoint")
                    else:
                        # No Fourier features in checkpoint
                        config.fourier_legacy = False
                        if hasattr(config, 'use_fourier_features'):
                            config.use_fourier_features = False
                        if verbose:
                            print(f"[ModelManager] No Fourier features detected in checkpoint")
                elif verbose:
                    print(f"[ModelManager] Using fourier_legacy={config.fourier_legacy} from config file (not auto-detecting)")
                
                # Check conv_in weight shape to determine conditioning channels
                # Skip this for cross-attention models (they use separate conditioning path)
                if not has_cross_attention and 'model.score_model.conv_in.weight' in state_dict:
                    conv_in_shape = state_dict['model.score_model.conv_in.weight'].shape
                    # Shape is [out_channels, in_channels, k, k]
                    # in_channels depends on Fourier mode:
                    # No Fourier: input(3) + conditioning(1) + large_scale(N)
                    # Legacy: input(3) + conditioning(1) + large_scale(N) + fourier_legacy(8)
                    # New: input(3) + conditioning(1) + fourier_halo(8) + large_scale(N) + fourier_largescale(8*N)
                    total_in_channels = conv_in_shape[1]
                    
                    input_channels = 3  # dm_hydro, gas, star
                    
                    # Determine which mode to use based on config settings
                    # Priority: use_fourier_features (highest) > fourier_legacy > auto-detect
                    
                    # Check if model was trained WITHOUT Fourier features
                    if not config.use_fourier_features:
                        # No Fourier mode: input(3) + conditioning(1) + large_scale(N)
                        total_conditioning = total_in_channels - input_channels
                        config.conditioning_channels = 1
                        config.large_scale_channels = total_conditioning - 1
                        
                        if verbose:
                            print(f"[ModelManager] No Fourier features mode channel breakdown:")
                            print(f"  Total conv_in: {total_in_channels}")
                            print(f"  = input({input_channels}) + conditioning({config.conditioning_channels}) + large_scale({config.large_scale_channels})")
                    elif config.fourier_legacy:
                        # Legacy mode: base_channels + 8 Fourier features
                        fourier_features = 8  # 4 freqs * 2 (sin/cos)
                        total_conditioning = total_in_channels - input_channels - fourier_features
                        
                        # Split into base DM (1) + large-scale (N)
                        config.conditioning_channels = 1
                        config.large_scale_channels = total_conditioning - 1
                        
                        if verbose:
                            print(f"[ModelManager] Legacy Fourier mode channel breakdown:")
                            print(f"  Total conv_in: {total_in_channels}")
                            print(f"  = input({input_channels}) + conditioning({config.conditioning_channels}) + large_scale({config.large_scale_channels}) + fourier_legacy({fourier_features})")
                    else:
                        # New mode: input(3) + conditioning(1) + fourier_halo(8) + large_scale(N) + fourier_largescale(8*N)
                        # Fourier features: 8 per channel (4 freqs * 2 for sin/cos)
                        fourier_per_channel = 8
                        
                        # Solve: total = 3 + 1 + 8 + N + 8*N = 3 + 1 + 8 + N(1 + 8) = 12 + 9*N
                        # N = (total - 12) / 9
                        if (total_in_channels - 12) % 9 == 0:
                            config.large_scale_channels = (total_in_channels - 12) // 9
                            config.conditioning_channels = 1
                            
                            if verbose:
                                print(f"[ModelManager] New multi-scale Fourier mode channel breakdown:")
                                print(f"  Total conv_in: {total_in_channels}")
                                print(f"  = input({input_channels}) + conditioning({config.conditioning_channels}) + fourier_halo({fourier_per_channel}) + large_scale({config.large_scale_channels}) + fourier_largescale({fourier_per_channel * config.large_scale_channels})")
                        else:
                            # Channel count doesn't match - this is an error!
                            # Don't override config settings
                            if verbose:
                                print(f"[ModelManager] ERROR: Channel count {total_in_channels} doesn't match expected formula for new Fourier mode")
                                print(f"[ModelManager] Expected: 12 + 9*N where N = large_scale_channels")
                                print(f"[ModelManager] This may indicate a config/checkpoint mismatch")
                            
                            # Try to calculate anyway assuming it might be no-Fourier or legacy
                            # But DON'T override explicit config settings
                            total_conditioning = total_in_channels - input_channels
                            config.conditioning_channels = 1
                            config.large_scale_channels = total_conditioning - 1
                            
                            if verbose:
                                print(f"[ModelManager] Attempting fallback calculation:")
                                print(f"  Total conv_in: {total_in_channels}")
                                print(f"  = input({input_channels}) + conditioning({config.conditioning_channels}) + large_scale({config.large_scale_channels}) + ??? (unknown Fourier channels)")
                                print(f"  WARNING: This may result in errors during model loading!")
                    
                    if verbose:
                        print(f"[ModelManager] Final channel configuration:")
                        print(f"  conditioning_channels: {config.conditioning_channels}")
                        print(f"  large_scale_channels: {config.large_scale_channels}")
                        print(f"  fourier_legacy: {config.fourier_legacy}")
            except Exception as e:
                if verbose:
                    print(f"[ModelManager] Could not auto-detect from checkpoint: {e}")
                # Fall through to manual calculation below
                config.conditioning_channels = None
        
        # If checkpoint doesn't exist or auto-detection failed, calculate from large_scale_channels
        if config.conditioning_channels is None:
            config.conditioning_channels = 1  # Always 1 for base DM
            if not hasattr(config, 'large_scale_channels'):
                config.large_scale_channels = 0  # Default to no large-scale channels
            if verbose:
                print(f"[ModelManager] Using config values: conditioning_channels={config.conditioning_channels}, large_scale_channels={config.large_scale_channels}")
        
        if verbose:
            print(f"[ModelManager] Setting up parameter conditioning...")
            
        if config.param_norm_path:
            use_param_conditioning = True
            param_norm_path = config.param_norm_path
            minmax_df = pd.read_csv(param_norm_path)
            min = np.array(minmax_df['MinVal'].values)
            max = np.array(minmax_df['MaxVal'].values)
            Nparams = len(min)
           

        if verbose:
            if not skip_data_loading:
                print(f"[ModelManager] Loading dataset (stage='test')...")
            else:
                print(f"[ModelManager] Skipping dataset loading (inference-only mode)")
                
        seed_everything(config.seed)
        
        # Skip data loading if flag is set (use pre-loaded samples)
        if skip_data_loading:
            hydro = None
        else:
            if not config.train_samples:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test/'
            else:
                test_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/'
            hydro = get_astro_data(
                config.dataset,
                test_root,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                stage='test',
                quantile_path=getattr(config, 'quantile_path', None)
            )
            if verbose:
                print(f"[ModelManager] Dataset loaded.")
        
        if verbose:
            print(f"[ModelManager] Creating UNetVDM model...")
            print(f"[ModelManager] Model configuration:")
            print(f"  - use_fourier_features: {config.use_fourier_features}")
            print(f"  - fourier_legacy: {config.fourier_legacy}")
            print(f"  - conditioning_channels: {config.conditioning_channels}")
            print(f"  - large_scale_channels: {config.large_scale_channels}")
            print(f"  - use_cross_attention: {config.use_cross_attention}")
            if config.use_cross_attention:
                print(f"    • location: {config.cross_attention_location}")
                print(f"    • heads: {config.cross_attention_heads}")
                print(f"    • dropout: {config.cross_attention_dropout}")
                print(f"    • chunked: {config.use_chunked_cross_attention} (chunk_size={config.cross_attention_chunk_size})")
            
        score_model = networks.UNetVDM(
            input_channels=3,
            gamma_min=config.gamma_min,
            gamma_max=config.gamma_max,
            embedding_dim=config.embedding_dim,
            norm_groups=config.norm_groups,
            n_blocks=config.n_blocks,
            add_attention=True,
            n_attention_heads=config.n_attention_heads,
            use_fourier_features=config.use_fourier_features,
            legacy_fourier=config.fourier_legacy,
            use_param_conditioning=use_param_conditioning,
            param_min=min,
            param_max=max,
            conditioning_channels=config.conditioning_channels,  # Base DM channel (1)
            large_scale_channels=config.large_scale_channels,  # Number of large-scale maps (N)
            # Cross-attention parameters (networks_clean.py)
            use_cross_attention=config.use_cross_attention,
            cross_attention_location=config.cross_attention_location,
            cross_attention_heads=config.cross_attention_heads,
            cross_attention_dropout=config.cross_attention_dropout,
            use_chunked_cross_attention=config.use_chunked_cross_attention,
            cross_attention_chunk_size=config.cross_attention_chunk_size,
            downsample_cross_attn_cond=config.downsample_cross_attn_cond,
            cross_attn_cond_downsample_factor=config.cross_attn_cond_downsample_factor,
        )
        
        if verbose:
            print(f"[ModelManager] UNetVDM created. Wrapping in LightCleanVDM...")
        
        # Parse channel weights
        channel_weights_str = getattr(config, 'channel_weights', '1.0,1.0,2.0')
        channel_weights = tuple(map(float, channel_weights_str.split(',')))
        
        # Parse lambdas if it's a string
        lambdas_value = getattr(config, 'lambdas', '1.0,1.0,1.0')
        if isinstance(lambdas_value, str):
            lambdas_value = lambdas_value  # Keep as string, LightCleanVDM will parse it
        
        # Image shape is always 2D for this model (channels, H, W)
        # The training data is 2D: condition (128, 128), target (3, 128, 128)
        image_shape = (3, config.cropsize, config.cropsize)
        
        vdm_hydro = vdm_module.LightCleanVDM(
            score_model=score_model,
            learning_rate=config.learning_rate,
            lr_scheduler=getattr(config, 'lr_scheduler', 'cosine'),
            gamma_min=config.gamma_min,
            gamma_max=config.gamma_max,
            image_shape=image_shape,
            noise_schedule=config.noise_schedule,
            data_noise=getattr(config, 'data_noise', 1e-5),
            antithetic_time_sampling=getattr(config, 'antithetic_time_sampling', True),
            lambdas=lambdas_value,
            channel_weights=(1, 1, 1),
            use_focal_loss=getattr(config, 'use_focal_loss', False),
            focal_gamma=getattr(config, 'focal_gamma', 2.0),
            use_param_prediction=getattr(config, 'use_param_prediction', False),
            param_prediction_weight=getattr(config, 'param_prediction_weight', 0.01),
        )
        
        if verbose:
            print(f"[ModelManager] Loading checkpoint weights from disk...")
            
        state_dict = torch.load(config.best_ckpt, map_location='cuda', weights_only=False)["state_dict"]
        
        if verbose:
            print(f"[ModelManager] Checkpoint loaded. Loading state dict into model...")
        
        # Some legacy checkpoints may omit buffers/params that newer model code
        # registers (for example, the legacy FourierFeatures registers a
        # 'freqs_exponent' buffer). To remain backwards compatible, detect any
        # keys that the freshly-initialized model expects but are missing from
        # the checkpoint state_dict, and inject the model's current defaults
        # for those keys into the loaded state dict before loading.
        model_state = vdm_hydro.state_dict()
        missing_keys = [k for k in model_state.keys() if k not in state_dict]
        if missing_keys:
            if verbose:
                print(f"[ModelManager] Warning: checkpoint is missing {len(missing_keys)} keys. Injecting defaults from model: {missing_keys}")
            for k in missing_keys:
                # Only inject if model provides a value
                state_dict[k] = model_state[k]

        vdm_hydro.load_state_dict(state_dict)
        vdm_hydro = vdm_hydro.eval()

        if getattr(config, 'verbose', False) or verbose:
            print("[ModelManager] Model and weights loaded successfully.")
            print(f"[ModelManager] Model checkpoint: {config.best_ckpt}")
            print(f"[ModelManager] Using {config.conditioning_channels} conditioning channel(s)")
            print(f"[ModelManager] Using {config.large_scale_channels} large-scale channel(s)")
            print(f"[ModelManager] Fourier features: {'LEGACY' if config.fourier_legacy else 'MULTI-SCALE' if config.use_fourier_features else 'DISABLED'}")

        return hydro, vdm_hydro

class DataHandler:
    PROJ_DIRS = ['yz', 'xz', 'xy']

    def __init__(self, base_path, sim_num, norms, data_type='test', verbose=False):
        self.base_path = base_path
        self.sim_num = sim_num
        self.data_type = data_type
        self.norms = norms
        self.verbose = verbose
        self._load_metadata()
        self.target_means = np.array([norms[4], norms[2], norms[0]])
        self.target_stds  = np.array([norms[5], norms[3], norms[1]])
        self.input_mean = norms[6]
        self.input_std = norms[7]
        if self.verbose:
            print(f"[DataHandler] Initialized for sim_num={sim_num}, data_type={data_type}")
            print(f"[DataHandler] Loaded {self.Nhalos} halos from metadata.")

    def _load_metadata(self):
        self.metadata = pd.read_csv(f'{self.base_path}/metadata/sim_{self.sim_num}.csv')
        self.halo_ids = self.metadata['nbody_index'].to_numpy().flatten()
        self.Nhalos = len(self.halo_ids)
        if getattr(self, 'verbose', False):
            print(f"[DataHandler] Metadata loaded from {self.base_path}/metadata/sim_{self.sim_num}.csv")

    def read_data(self, index):
        """Read data with optional Omega_m, halo_mass, and ASN1 return"""
        img_path = f'{self.base_path}/{self.data_type}_3d/sim_{self.sim_num}/halo_{index}_3d.npz'
        with np.load(img_path) as data:
            dm = data['dm'] + 1
            dm_hydro = data['dm_hydro'] + 1
            gas = data['gas'] + 1
            star = data['star'] + 1
            # Try to load Omega_m, halo_mass, and ASN1 if available
            conditional_params =data['conditional_params']
    
        condition = dm.astype(np.float32)
        target = np.stack([dm_hydro, gas, star]).astype(np.float32)
        
        result = [condition, target, conditional_params]
        
        return tuple(result)

    def normalize_condition(self, condition):
        condition_log = np.log10(condition + 1)
        return (condition_log - self.input_mean) / self.input_std

    def normalize_target(self, target):
        target_log = np.log10(target + 1)
        return (target_log - self.target_means[:, None, None, None]) / self.target_stds[:, None, None, None]

    def unnormalize_target(self, target_norm):
        """
        Unnormalize target with Jensen's inequality correction for log-normal bias.
        
        The correction term (0.5 * std^2 * ln(10)) accounts for the bias introduced
        when denormalizing from log-space: E[10^X] ≠ 10^E[X]
        """
        # Apply bias correction: add 0.5 * std^2 * ln(10) before exponentiation
        correction = 0.5 * (self.target_stds ** 2) * np.log(10)
        unnorm = target_norm * self.target_stds[:, None, None, None] + self.target_means[:, None, None, None] 
        return 10 ** unnorm - 1

    def batch_read_and_normalize(self, indices=None):
        """Batch read with optional Omega_m, halo_mass, and ASN1"""
        conds, tgts = [], []
        conditional_params = []
    
        if indices is None:
            indices = self.halo_ids
            
        for idx in indices:
            read_results = self.read_data(idx)
            
            
            cond, tgt, cond_params = read_results
            
            conds.append(self.normalize_condition(cond))
            tgts.append(self.normalize_target(tgt))
            conditional_params.append(cond_params)
        
        result = [np.stack(conds), np.stack(tgts), np.stack(conditional_params)]
        
        return tuple(result)

# Enhanced sampling function with ASN1 support        
def sample(vdm, conditions, batch_size=1, conditional_params=None,):
    """
    Process multiple conditions and return stacked samples with optional Omega_m, halo_mass, and ASN1 conditioning.
    
    Args:
        vdm: Your VDM model
        conditions: torch.Tensor, shape (N, C, H, W) or (N, C, H, W, D) 
                   where C is the number of conditioning channels:
                   - C=1: base DM condition only
                   - C>1: base DM + (C-1) large-scale maps
        batch_size: Number of samples per condition
        conditional_params: Optional array of conditional parameters
    
    Returns:
        torch.Tensor of shape (N, batch_size, 3, H, W) or (N, batch_size, 3, H, W, D)
    """
    # Ensure conditions has batch dimension
    # conditions should already be (N, C, H, W) or (N, C, H, W, D)
    vdm = vdm.to('cuda')
    samples = []
    
    # Determine dimensionality
    is_3d = len(conditions.shape) == 5  # (N, C, H, W, D)
    
    for i, cond in enumerate(tqdm(conditions, desc='Generating Samples')):
        # cond is now (C, H, W) or (C, H, W, D) where C = 1 + num_large_scales
        # Expand to batch_size: (batch_size, C, H, W) or (batch_size, C, H, W, D)
        if is_3d:
            cond_expanded = cond.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to('cuda')
        else:
            cond_expanded = cond.unsqueeze(0).expand(batch_size, -1, -1, -1).to('cuda')
        
        if conditional_params is not None:
            param_row = torch.tensor(conditional_params[i], dtype=torch.float32, device='cuda')
            param_expanded = param_row.unsqueeze(0).expand(batch_size, -1).to('cuda')  # (batch_size, Nparams)
        else:
            param_expanded = None
        hydro_sample = vdm.draw_samples(
            conditioning=cond_expanded,
            batch_size=batch_size,
            n_sampling_steps=getattr(vdm.hparams, 'n_sampling_steps', 1000),
            param_conditioning=param_expanded,
        )  
        
        samples.append(hydro_sample.unsqueeze(0))  # Add N dimension
    
    return torch.cat(samples, dim=0).to("cpu")  # (N, batch_size, 3, H, W) or (N, batch_size, 3, H, W, D)


# Keep all existing functions unchanged
def get_density(mass_map, radius, nbins=20, nR=2, logbins=False, physical_bins=False):    
    """
    Calculate surface density in radial shells from the center of the image.
    
    Args:
        mass_map: 2D array of mass/density values
        radius: Physical radius scale (e.g., R200 in kpc)
        nbins: Number of radial bins
        nR: Maximum radius in units of the scale radius (ignored if physical_bins=True)
        logbins: Whether to use logarithmic binning
        physical_bins: If True, use physical units (kpc) for binning instead of normalized
    
    Returns:
        bin_centers: Radial bin centers (units depend on physical_bins setting)
        surface_density: Surface density in each radial shell [mass/area]
    """
    # Map parameters
    map_size_mpc = 50/1024*128  # Total physical size of map in Mpc
    pixel_size_mpc = map_size_mpc / mass_map.shape[0]  # Size per pixel in Mpc
    pixel_size_kpc = pixel_size_mpc * 1000  # Convert to kpc
    
    # Calculate pixel distances from center
    center = (mass_map.shape[0]/2 - 0.5, mass_map.shape[1]/2 - 0.5)  # Center in pixel coordinates
    y, x = np.indices(mass_map.shape)
    radii_pixels = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    radii_physical = radii_pixels * pixel_size_kpc  # Physical distance in kpc
    radii_normalized = radii_physical / radius  # Normalized by scale radius
    
    # Radial binning - choose between physical and normalized
    if physical_bins:
        # Use physical units (kpc) as in your notebook
        min_radius = pixel_size_kpc * 4  # Minimum radius (4 pixels)
        max_radius = 2500  # 2.5 Mpc in kpc
        
        if not logbins:
            bin_edges = np.linspace(min_radius, max_radius, nbins + 1, endpoint=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        else:
            bin_edges = np.logspace(np.log10(min_radius), np.log10(max_radius), nbins + 1, endpoint=True)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean
        
        # Use physical radii for masking
        radii_for_binning = radii_physical
        
    else:
        # Use normalized units (original behavior)
        if not logbins:
            bin_edges = np.linspace(0.01, nR, nbins + 1, endpoint=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        else:
            bin_edges = np.logspace(np.log10(0.01), np.log10(nR), nbins + 1, endpoint=True)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean
        
        # Use normalized radii for masking
        radii_for_binning = radii_normalized
    
    # Calculate surface density Σ(R) in each annular shell
    surface_density = []
    for i in range(len(bin_edges) - 1):
        # Create mask for this radial shell
        mask = (radii_for_binning >= bin_edges[i]) & (radii_for_binning < bin_edges[i+1])
        
        # Total mass in this shell
        total_mass = np.sum(mass_map[mask])
        
        # Area of the annular shell in kpc^2
        if physical_bins:
            # Bin edges are already in kpc
            inner_radius_kpc = bin_edges[i]
            outer_radius_kpc = bin_edges[i+1]
        else:
            # Convert normalized bin edges to kpc
            inner_radius_kpc = bin_edges[i] * radius
            outer_radius_kpc = bin_edges[i+1] * radius
        
        shell_area_kpc2 = np.pi * (outer_radius_kpc**2 - inner_radius_kpc**2)
        
        # Surface density = mass / area
        if shell_area_kpc2 > 0:
            surf_density = total_mass / shell_area_kpc2
        else:
            surf_density = 0.0
            
        surface_density.append(surf_density)
    
    return np.array(bin_centers), np.array(surface_density)
def generate_density_profiles(hydro_sample, target, test_data, 
                             density_types=('dm', 'gas', 'star', 'all'), 
                             nbins=20, nR=2, logbins=True, physical_bins=False):
    """
    Generate density profiles for multiple fields.
    
    Args:
        hydro_sample: Sampled data array (N, batch_size, 3, H, W)
        target: True target data array (N, 3, H, W)
        test_data: DataHandler instance with metadata
        density_types: Tuple of fields to process ('dm', 'gas', 'star')
        nbins, nR, logbins: Parameters for radial binning
    
    Returns:
        (true_profiles, sampled_profiles) where each is a dictionary
        with density types as keys and lists of profiles as values
    """
    # Map density types to channel indices
    channel_map = {
        'dm': 0,
        'gas': 1,
        'star': 2,
        'all': [0,1,2],
    }
    
    # Validate input types
    invalid_types = [dt for dt in density_types if dt not in channel_map]
    if invalid_types:
        raise ValueError(f"Invalid density types: {invalid_types}. Valid options: {list(channel_map.keys())}")
    
    # Initialize profile storage
    true_profiles = {dt: [] for dt in density_types}
    sampled_profiles = {dt: [] for dt in density_types}

    # Helper function for density calculation
    def calculate_density(mass_map, radius):
        return get_density(
            mass_map=np.array(mass_map),
            radius=radius,
            nbins=nbins,
            nR=nR,
            logbins=logbins,
            physical_bins=physical_bins
        )[1]  # Return just the density values

    # Main processing loop
    for ii in range(len(hydro_sample)):
        radius = test_data.metadata.loc[:, 'R_hydro'].iloc[ii]
        
        for dt in density_types:
            channel = channel_map[dt]
            
            # Process true density
            true_map = test_data.unnormalize_target(target[ii])[channel]
            if dt =='all':
                true_map = np.sum(true_map, axis=0)
            true_profiles[dt].append(calculate_density(true_map, radius))
            
            # Process sampled densities
            sample_maps = [test_data.unnormalize_target(hydro_sample[ii, n])[channel] 
                          for n in range(len(hydro_sample[ii]))]
            if dt =='all':
                sample_maps = [np.sum(s, axis=0) for s in sample_maps]
            sampled_profiles[dt].append([
                calculate_density(smap, radius) for smap in sample_maps
            ])
    
    return true_profiles, sampled_profiles
def plot_combined_profiles(index, true_profiles, sampled_profiles, bin_centers):
    """
    Plot DM, gas, and star density profiles on the same panel.
    
    Args:
        index: Halo index to plot
        true_profiles: Dictionary of true density profiles
        sampled_profiles: Dictionary of sampled density profiles
        bin_centers: Radial bin centers from get_density()
    """
    plt.figure(figsize=(8, 6))
    
    # Define styling for each profile type
    styles = {
        'dm': {'color': 'blue', 'label': 'Dark Matter'},
        'gas': {'color': 'green', 'label': 'Gas'},
        'star': {'color': 'red', 'label': 'Stars'},
        'all':{'color':'black', 'label':'Total'}
    }
    
    for density_type in ['dm', 'gas', 'star', 'all']:
        # Extract data
        true = true_profiles[density_type][index]
        samples = np.array(sampled_profiles[density_type][index])
        
        # Calculate statistics
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        
        # Plotting
        plt.plot(bin_centers, true, 
                 color=styles[density_type]['color'], 
                 linestyle='-', 
                 label=f"True {styles[density_type]['label']}")
        
        plt.plot(bin_centers, mean,
                 color=styles[density_type]['color'],
                 linestyle='--',
                 label=f"Sampled {styles[density_type]['label']}")
        
        plt.fill_between(bin_centers, 
                         mean - std, 
                         mean + std,
                         color=styles[density_type]['color'], 
                         alpha=0.2)

    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Radius [R/R200]', fontsize=12)
    plt.ylabel(r'Density [M$_\odot$/kpc$^3$]', fontsize=12)
    plt.title(f'Density Profiles Comparison - Halo {index}', fontsize=14)
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_density_and_images(index, true_profiles, sampled_profiles, bin_centers, hydro_sample, target):
    """
    Creates a comprehensive visualization with density profiles on top and image comparisons below.
    
    Args:
        index: Index of the halo to visualize
        true_profiles: Dictionary of true density profiles by type
        sampled_profiles: Dictionary of sampled density profiles by type
        bin_centers: Radial bin centers
        hydro_sample: Sampled hydro data, shape (N, batch_size, 3, H, W)
        target: True target data, shape (N, 3, H, W)
    """
    fig = plt.figure(figsize=(15, 15))
    
    # Create grid for density plot (top) and image panels (bottom)
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], hspace=0.3)
    
    # Density plot on top spanning all columns
    ax_density = fig.add_subplot(gs[0, :])
    
    styles = {
        'dm': {'color': 'blue', 'label': 'Dark Matter'},
        'gas': {'color': 'green', 'label': 'Gas'},
        'star': {'color': 'red', 'label': 'Stars'},
        'all':{'color':'black', 'label':'Total'}
    }
    
    for density_type in ['dm', 'gas', 'star', 'all']:
        true = true_profiles[density_type][index]
        samples = np.array(sampled_profiles[density_type][index])
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        
        ax_density.plot(bin_centers, true, 
                        color=styles[density_type]['color'], 
                        linestyle='-', 
                        label=f"True {styles[density_type]['label']}")
        ax_density.plot(bin_centers, mean,
                        color=styles[density_type]['color'],
                        linestyle='--',
                        label=f"Sampled {styles[density_type]['label']}")
        ax_density.fill_between(bin_centers, 
                                mean - std, 
                                mean + std,
                                color=styles[density_type]['color'], 
                                alpha=0.2)

    # ax_density.set_xscale('log')
    ax_density.set_yscale('log')
    ax_density.set_xlabel('Radius [R/R200]', fontsize=12)
    ax_density.set_ylabel(r'Density [M$_\odot$/kpc$^3$]', fontsize=12)
    ax_density.set_title(f'Density Profiles Comparison - Halo {index}', fontsize=14)
    ax_density.legend(fontsize=10, frameon=False, ncols=3)
    ax_density.grid(True, which='both', linestyle='--', alpha=0.5)

    # Image panels below: 2 rows x 3 cols
    channel_names = ['dm_hydro', 'gas', 'star']
    for i in range(3):
        # Top row: generated images
        ax_gen = fig.add_subplot(gs[1, i])
        mean_hydro = hydro_sample[index, :, i].mean(axis=0)
        ax_gen.imshow(mean_hydro, cmap='viridis')
        ax_gen.set_title(f'Generated {channel_names[i]}')
        ax_gen.axis('off')

        # Bottom row: true images
        ax_true = fig.add_subplot(gs[2, i])
        ax_true.imshow(target[index, i], cmap='viridis')
        ax_true.set_title(f'True {channel_names[i]}')
        ax_true.axis('off')

    # plt.tight_layout()
    return fig

