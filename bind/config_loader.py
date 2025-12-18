"""ConfigLoader for loading and managing INI configuration files."""

import configparser
import glob
import os
import re

import numpy as np
import pandas as pd

from config import NORMALIZATION_STATS_DIR
from vdm.verbosity import VerbosityContext


def load_normalization_stats(base_path=None, verbosity=None):
    """
    Load normalization statistics from .npz files.
    
    Parameters
    ----------
    base_path : str, optional
        Directory containing the normalization .npz files.
        Defaults to NORMALIZATION_STATS_DIR from config.py
    verbosity : int or str or bool, optional
        Verbosity level. If None, uses global setting.
        If bool, True=DEBUG, False=SILENT.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - dm_mag_mean, dm_mag_std
        - gas_mag_mean, gas_mag_std
        - star_mag_mean, star_mag_std
    """
    if base_path is None:
        base_path = NORMALIZATION_STATS_DIR
    
    ctx = VerbosityContext(verbosity)
    stats = {}
    
    # Load DM stats
    dm_path = os.path.join(base_path, 'dark_matter_normalization_stats.npz')
    if os.path.exists(dm_path):
        dm_stats = np.load(dm_path)
        stats['dm_mag_mean'] = float(dm_stats['dm_mag_mean'])
        stats['dm_mag_std'] = float(dm_stats['dm_mag_std'])
        ctx.vprint_debug(f"✓ Loaded DM normalization: mean={stats['dm_mag_mean']:.6f}, std={stats['dm_mag_std']:.6f}")
    else:
        raise FileNotFoundError(f"DM normalization file not found: {dm_path}")
    
    # Load Gas stats
    gas_path = os.path.join(base_path, 'gas_normalization_stats.npz')
    if os.path.exists(gas_path):
        gas_stats = np.load(gas_path)
        stats['gas_mag_mean'] = float(gas_stats['gas_mag_mean'])
        stats['gas_mag_std'] = float(gas_stats['gas_mag_std'])
        ctx.vprint_debug(f"✓ Loaded Gas normalization: mean={stats['gas_mag_mean']:.6f}, std={stats['gas_mag_std']:.6f}")
    else:
        raise FileNotFoundError(f"Gas normalization file not found: {gas_path}")
    
    # Load Stellar stats
    star_path = os.path.join(base_path, 'stellar_normalization_stats.npz')
    if os.path.exists(star_path):
        star_stats = np.load(star_path)
        stats['star_mag_mean'] = float(star_stats['star_mag_mean'])
        stats['star_mag_std'] = float(star_stats['star_mag_std'])
        ctx.vprint_debug(f"✓ Loaded Stellar normalization: mean={stats['star_mag_mean']:.6f}, std={stats['star_mag_std']:.6f}")
    else:
        raise FileNotFoundError(f"Stellar normalization file not found: {star_path}")
    
    return stats


class ConfigLoader:
    """
    Load and manage configuration parameters from an INI file.
    
    Parameters
    ----------
    config_path : str
        Path to the INI configuration file.
    verbose : bool or int or str, optional
        Verbosity level. If bool, True=DEBUG, False=SILENT.
        If int/str, uses that level directly.
    train_samples : bool, optional
        Whether to load training samples (default: False).
    
    Attributes
    ----------
    best_ckpt : str or None
        Path to the best checkpoint found, or None if not found.
    tb_log_path : str
        Path to TensorBoard logs.
    conditioning_channels : int
        Number of conditioning channels (auto-detected from checkpoint).
    large_scale_channels : int
        Number of large-scale conditioning channels.
    """
    
    def __init__(self, config_path='config.ini', verbose=False, train_samples=False):
        self.config_path = config_path
        # Convert verbose to verbosity context
        self._verbosity = VerbosityContext(verbose)
        self.verbose = verbose  # Keep for backward compatibility
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
                      'ema_update_after_step', 'ema_update_every',
                      # DiT parameters
                      'patch_size', 'hidden_size', 'depth', 'num_heads', 'warmup_steps', 'max_epochs'}
        float_params = {'gamma_min', 'gamma_max', 'learning_rate', 'mass_conservation_weight',
                        'sparsity_threshold', 'sparse_loss_weight', 'focal_alpha', 'focal_gamma',
                        'param_prediction_weight', 'cross_attention_dropout',
                        # DDPM/score_models parameters
                        'beta_min', 'beta_max', 'sigma_min', 'sigma_max', 'ema_decay', 'dropout',
                        # DiT parameters
                        'mlp_ratio', 'weight_decay', 'gradient_clip_val'}
        bool_params = {'use_large_scale', 'use_fourier_features', 'fourier_legacy', 'legacy_fourier',
                       'add_attention', 'use_progressive_field_weighting', 'use_mass_conservation',
                       'use_sparsity_aware_loss', 'use_focal_loss', 'use_param_prediction', 
                       'use_auxiliary_mask', 'antithetic_time_sampling', 'use_cross_attention',
                       'use_chunked_cross_attention', 'downsample_cross_attn_cond',
                       # DDPM/score_models parameters
                       'use_param_conditioning', 'attention', 'enable_ema', 'enable_early_stopping',
                       'enable_gradient_monitoring',
                       # DiT parameters
                       'use_quantile_normalization', 'use_ema'}
        
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
            self._verbosity.vprint_debug(f"[ConfigLoader] Converted legacy_fourier={self.legacy_fourier} to fourier_legacy")
        
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
            self._verbosity.vprint_debug(f"[ConfigLoader] conditioning_channels not in config or None, using default: {self.conditioning_channels}")
        else:
            self._verbosity.vprint_debug(f"[ConfigLoader] Found conditioning_channels in config: {self.conditioning_channels}")
            self._verbosity.vprint_debug(f"[ConfigLoader] WARNING: This will be overridden by checkpoint auto-detection!")
        
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
        
        # ✅ HALO MASS CONDITIONING SUPPORT
        # If include_halo_mass is enabled, model expects log10(halo_mass) as additional parameter
        self.include_halo_mass = False
        if 'include_halo_mass' in params:
            self.include_halo_mass = params['include_halo_mass'].lower() in ('true', '1', 'yes')
        
        if self.include_halo_mass:
            # Get halo mass bounds (in log10 Msun)
            self.halo_mass_min = float(params.get('halo_mass_min', '13.0'))
            self.halo_mass_max = float(params.get('halo_mass_max', '15.0'))
            
            # Append halo mass bounds to param arrays
            if hasattr(self, 'min') and hasattr(self, 'max'):
                self.min = np.append(self.min, self.halo_mass_min)
                self.max = np.append(self.max, self.halo_mass_max)
                self.Nparams += 1
                self._verbosity.vprint_summary(f"[ConfigLoader] 📊 Halo mass conditioning enabled: log10 range [{self.halo_mass_min}, {self.halo_mass_max}]")
                self._verbosity.vprint_summary(f"[ConfigLoader]    → Total parameters: {self.Nparams}")
            else:
                # Create param arrays with just halo mass
                self.min = np.array([self.halo_mass_min])
                self.max = np.array([self.halo_mass_max])
                self.Nparams = 1
                self.use_param_conditioning = True
                self._verbosity.vprint_summary(f"[ConfigLoader] 📊 Halo mass conditioning enabled (halo mass only)")

        # ✅ QUANTILE NORMALIZATION SUPPORT
        # Load quantile_path if specified (for stellar channel normalization)
        if 'quantile_path' in params and params['quantile_path']:
            quantile_val = params['quantile_path'].strip()
            if quantile_val.lower() not in ['none', '']:
                self.quantile_path = quantile_val
                self._verbosity.vprint_debug(f"[ConfigLoader] 🌟 Quantile normalization enabled: {self.quantile_path}")
            else:
                self.quantile_path = None
        else:
            self.quantile_path = None

        self._verbosity.vprint_debug(f"[ConfigLoader] Loaded config from: {self.config_path}")
        if self._verbosity.is_debug():
            print(f"[ConfigLoader] Parameters:")
            for key in params:
                print(f"  {key}: {getattr(self, key, None)}")

    def _state_initialization(self):
        """
        Find the best checkpoint based on validation metrics from TensorBoard logs.
        
        For models that log val/elbo or val/loss (VDM, Triple, Interpolant, Consistency, DiT):
          - Reads TensorBoard events to find step with best validation metric
          - Selects checkpoint closest to that step
        
        For models without reliable validation metrics (DDPM, DSM, OT Flow):
          - Falls back to latest checkpoint by step number
        
        Checkpoint patterns supported:
        1. epoch-epoch=XXX-step=YYY.ckpt (epoch checkpoints)
        2. epoch=X-step=Y-val_*.ckpt (validation checkpoints)
        3. latest-epoch=X-step=Y.ckpt (step-based checkpoints)
        """
        self.tb_log_path = f"{self.tb_logs}/{self.model_name}/version_{self.version}/"
        
        # Collect all available checkpoints
        all_ckpts = []
        
        # Pattern 1: Epoch checkpoint files
        all_ckpts.extend(glob.glob(f'{self.tb_log_path}/checkpoints/epoch-epoch*.ckpt'))
        all_ckpts.extend(glob.glob(f'{self.tb_log_path}/**/checkpoints/epoch-epoch*.ckpt', recursive=True))
        
        # Pattern 2: Val checkpoint files
        all_ckpts.extend(glob.glob(f'{self.tb_log_path}/checkpoints/epoch=*-val_*.ckpt'))
        all_ckpts.extend(glob.glob(f'{self.tb_log_path}/**/checkpoints/epoch=*-val_*.ckpt', recursive=True))
        
        # Pattern 3: Latest checkpoint files
        all_ckpts.extend(glob.glob(f'{self.tb_log_path}/checkpoints/latest-*.ckpt'))
        all_ckpts.extend(glob.glob(f'{self.tb_log_path}/**/checkpoints/latest-*.ckpt', recursive=True))
        
        # Remove duplicates while preserving order
        all_ckpts = list(dict.fromkeys(all_ckpts))
        
        if not all_ckpts:
            self.best_ckpt = None
            self._verbosity.vprint_summary("[ConfigLoader] No checkpoint found.")
            return
        
        # Determine model type from model_name to decide checkpoint selection strategy
        model_lower = self.model_name.lower()
        # Models that should use latest checkpoint (validation metrics unreliable or not logged):
        # - DDPM, DSM, OT Flow: use denoising score matching loss, no val metric
        # - DiT: validation loss logging may be unreliable, use latest for now
        # - Triple VDM: validation metric logs val_elbo=0.0000, use latest instead
        use_latest = any(m in model_lower for m in ['ddpm', 'dsm', 'ot_flow', 'ot-flow', 'dit', 'triple'])
        
        if use_latest:
            # For DDPM, DSM, OT Flow, DiT: use latest checkpoint by step
            self._verbosity.vprint_debug(f"[ConfigLoader] Using latest checkpoint for {self.model_name}")
            self.best_ckpt = self._select_latest_checkpoint(all_ckpts)
        else:
            # For VDM, Triple, Interpolant, Consistency: use best validation metric
            self._verbosity.vprint_debug(f"[ConfigLoader] Looking for best validation checkpoint for {self.model_name}")
            self.best_ckpt = self._select_best_validation_checkpoint(all_ckpts)
        
        self._verbosity.vprint_debug(f"[ConfigLoader] TensorBoard log path: {self.tb_log_path}")
        if self.best_ckpt:
            self._verbosity.vprint_summary(f"[ConfigLoader] Found checkpoint: {self.best_ckpt}")
        else:
            self._verbosity.vprint_summary("[ConfigLoader] No checkpoint found.")
    
    def _select_latest_checkpoint(self, ckpts):
        """Select checkpoint with highest step number."""
        if not ckpts:
            return None
        ckpts.sort(key=self._natural_sort_key)
        return ckpts[-1]
    
    def _select_best_validation_checkpoint(self, ckpts):
        """
        Select checkpoint closest to the step with best validation metric.
        Falls back to latest checkpoint if TensorBoard logs unavailable.
        """
        if not ckpts:
            return None
        
        # Try to read validation metrics from TensorBoard
        best_step = self._get_best_validation_step()
        
        if best_step is None:
            # Fallback to latest checkpoint
            self._verbosity.vprint_debug("[ConfigLoader] Could not read validation metrics, using latest checkpoint")
            return self._select_latest_checkpoint(ckpts)
        
        self._verbosity.vprint_debug(f"[ConfigLoader] Best validation at step {best_step}")
        
        # Extract step number from each checkpoint and find closest to best_step
        ckpt_steps = []
        for ckpt in ckpts:
            step = self._extract_step_from_checkpoint(ckpt)
            if step is not None:
                ckpt_steps.append((ckpt, step))
        
        if not ckpt_steps:
            return self._select_latest_checkpoint(ckpts)
        
        # Find checkpoint with step closest to (but not exceeding) best_step
        # If no checkpoint at or before best_step, take the closest one after
        valid_ckpts = [(c, s) for c, s in ckpt_steps if s <= best_step]
        
        if valid_ckpts:
            # Take checkpoint closest to best_step (highest step <= best_step)
            best_ckpt = max(valid_ckpts, key=lambda x: x[1])[0]
        else:
            # All checkpoints are after best_step, take the earliest one
            best_ckpt = min(ckpt_steps, key=lambda x: x[1])[0]
        
        return best_ckpt
    
    def _get_best_validation_step(self):
        """
        Read TensorBoard logs to find the step with best validation metric.
        Returns None if logs cannot be read.
        """
        try:
            from tensorboard.backend.event_processing import event_accumulator
            
            ea = event_accumulator.EventAccumulator(self.tb_log_path)
            ea.Reload()
            
            scalars = ea.Tags().get('scalars', [])
            
            # Try different validation metric names
            val_metrics = ['val/elbo', 'val/loss', 'val_elbo', 'val_loss']
            
            for metric in val_metrics:
                if metric in scalars:
                    vals = ea.Scalars(metric)
                    if vals:
                        # Find step with minimum validation metric
                        best = min(vals, key=lambda x: x.value)
                        self._verbosity.vprint_debug(
                            f"[ConfigLoader] Found {metric}: best={best.value:.4f} at step {best.step}"
                        )
                        return best.step
            
            self._verbosity.vprint_debug(f"[ConfigLoader] No validation metrics found in TensorBoard logs")
            return None
            
        except ImportError:
            self._verbosity.vprint_debug("[ConfigLoader] tensorboard not installed, cannot read validation metrics")
            return None
        except Exception as e:
            self._verbosity.vprint_debug(f"[ConfigLoader] Error reading TensorBoard logs: {e}")
            return None
    
    def _extract_step_from_checkpoint(self, ckpt_path):
        """Extract step number from checkpoint filename."""
        filename = os.path.basename(ckpt_path)
        
        # Try different patterns
        # Pattern: step=12345 or step-12345
        match = re.search(r'step[=\-](\d+)', filename)
        if match:
            return int(match.group(1))
        
        # Pattern: epoch-epoch=XXX-step=YYY (older format with step at end)
        match = re.search(r'-(\d+)\.ckpt$', filename)
        if match:
            return int(match.group(1))
        
        return None

    @staticmethod
    def _natural_sort_key(s):
        """Generate key for natural sorting of strings containing numbers."""
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)
        ]
