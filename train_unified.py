"""
Unified Training Script for Generative Models.

This script consolidates all training scripts into a single unified entry point.
It supports the following model types:

- VDM (Variational Diffusion Model) - 3-channel mode
- Triple VDM - 3 independent single-channel VDMs
- DDPM (Denoising Diffusion Probabilistic Models) via score_models
- DSM (Denoising Score Matching) with custom UNet
- Interpolant (Flow Matching / Stochastic Interpolants)
- OT-Flow (Optimal Transport Flow Matching)
- Consistency (Consistency Models)

Usage:
    python train_unified.py --model vdm --config configs/clean_vdm_aggressive_stellar.ini
    python train_unified.py --model triple --config configs/clean_vdm_triple.ini
    python train_unified.py --model ddpm --config configs/ddpm.ini
    python train_unified.py --model dsm --config configs/dsm.ini
    python train_unified.py --model interpolant --config configs/interpolant.ini
    python train_unified.py --model ot_flow --config configs/ot_flow.ini
    python train_unified.py --model consistency --config configs/consistency.ini
    
    # CPU testing mode (any model)
    python train_unified.py --model vdm --config configs/clean_vdm.ini --cpu_only

Model Type Reference:
    vdm         - 3-channel VDM (LightCleanVDM)
    triple      - 3 independent VDMs (LightTripleVDM)
    ddpm        - DDPM/NCSNpp via score_models (LightScoreModel)
    dsm         - DSM with custom UNet (LightDSM)
    interpolant - Flow matching (LightInterpolant)
    ot_flow     - OT flow matching (LightOTFlow)
    consistency - Consistency models (LightConsistency)
"""

import os
import argparse
import configparser
import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from vdm.astro_dataset import get_astro_data
from vdm.networks_clean import UNetVDM
from vdm.callbacks import (
    FIDMonitorCallback,
    GradientMonitorCallback,
    CustomEarlyStopping,
    EMACallback,
)

torch.set_float32_matmul_precision("medium")

# =============================================================================
# Model Type Constants
# =============================================================================

MODEL_TYPES = ['vdm', 'triple', 'ddpm', 'dsm', 'interpolant', 'ot_flow', 'consistency']

DEFAULT_CONFIGS = {
    'vdm': 'configs/clean_vdm_aggressive_stellar.ini',
    'triple': 'configs/clean_vdm_triple.ini',
    'ddpm': 'configs/ddpm.ini',
    'dsm': 'configs/dsm.ini',
    'interpolant': 'configs/interpolant.ini',
    'ot_flow': 'configs/ot_flow.ini',
    'consistency': 'configs/consistency.ini',
}


# =============================================================================
# Config Parsing Utilities
# =============================================================================

class ConfigParser:
    """Helper class for parsing config files with type conversion."""
    
    def __init__(self, params):
        self.params = params
    
    def get_int(self, key, default):
        return int(self.params.get(key, default))
    
    def get_float(self, key, default):
        return float(self.params.get(key, default))
    
    def get_bool(self, key, default):
        val = self.params.get(key, str(default))
        if isinstance(val, bool):
            return val
        return val.lower() in ('true', '1', 'yes')
    
    def get_str(self, key, default):
        return self.params.get(key, default)
    
    def get_list_float(self, key, default='1.0,1.0,1.0'):
        val = self.params.get(key, default)
        return tuple(map(float, val.split(',')))
    
    def get_list_int(self, key, default='1,2,2,4'):
        val = self.params.get(key, default)
        return tuple(map(int, val.split(',')))
    
    def get_optional(self, key, default=None):
        """Get optional parameter, returns None if 'none' or empty."""
        val = self.params.get(key, default)
        if val is None:
            return None
        if isinstance(val, str) and val.lower() in ['none', '']:
            return None
        return val
    
    def get_optional_int(self, key, default=None):
        val = self.get_optional(key, default)
        return int(val) if val is not None else None


# =============================================================================
# Common Training Function
# =============================================================================

def train(
    model,
    datamodule,
    model_name,
    model_type,
    version=None,
    dataset='IllustrisTNG',
    boxsize=6.25,
    # Callbacks
    enable_early_stopping=True,
    early_stopping_patience=300,
    early_stopping_monitor='val/elbo',  # 'val/elbo' for VDM, 'val/loss' for others
    enable_fid_monitoring=False,
    fid_compute_every_n_epochs=1,
    fid_n_samples=100,
    enable_gradient_monitoring=True,
    gradient_log_frequency=100,
    # Training
    max_epochs=100,
    limit_train_batches=1.0,
    tb_logs='tb_logs',
    cpu_only=False,
    # EMA parameters
    enable_ema=False,
    ema_decay=0.9999,
    ema_update_after_step=0,
    ema_update_every=1,
    # Memory optimization
    accumulate_grad_batches=1,
    gradient_clip_val=1.0,
    # Speed optimizations
    precision="32",
    compile_model=False,
    # Resume
    resume_checkpoint=None,
    # DDP settings
    find_unused_parameters=False,
):
    """
    Unified training function for all model types.
    
    Args:
        model: Lightning model to train
        datamodule: PyTorch Lightning DataModule
        model_name: Name for logging and checkpoints
        model_type: Type of model ('vdm', 'triple', 'ddpm', etc.)
        version: TensorBoard version number
        dataset: Dataset name (for reference)
        boxsize: Physical size of box (for reference)
        enable_early_stopping: Enable early stopping callback
        early_stopping_patience: Epochs to wait before stopping
        early_stopping_monitor: Metric to monitor ('val/elbo' or 'val/loss')
        enable_fid_monitoring: Enable FID metric computation (VDM/Triple only)
        fid_compute_every_n_epochs: Epochs between FID computations
        fid_n_samples: Number of samples for FID computation
        enable_gradient_monitoring: Monitor gradient statistics
        gradient_log_frequency: Steps between gradient logging
        max_epochs: Maximum training epochs
        limit_train_batches: Fraction of training data per epoch
        tb_logs: TensorBoard logs directory
        cpu_only: Force CPU training (for testing only)
        enable_ema: Enable Exponential Moving Average
        ema_decay: EMA decay factor
        ema_update_after_step: Start EMA updates after this many steps
        ema_update_every: Update EMA every N steps
        accumulate_grad_batches: Accumulate gradients over N batches
        gradient_clip_val: Gradient clipping value (None to disable)
        precision: Training precision ("32", "16-mixed", "bf16-mixed")
        compile_model: Enable torch.compile optimization
        resume_checkpoint: Path to checkpoint to resume from
        find_unused_parameters: Enable find_unused_parameters in DDP
    """
    
    ckpt_path = resume_checkpoint
    
    # TensorBoard logger
    logger = TensorBoardLogger(tb_logs, name=model_name, version=version)

    # Checkpoint every time validation metric improves
    val_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{" + early_stopping_monitor.replace('/', '_') + ":.4f}",
        monitor=early_stopping_monitor,
        mode="min",
    )

    # Checkpoint at every 6000 steps
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
        every_n_train_steps=6000,
        save_top_k=10
    )
    
    # Checkpoint at every epoch
    epoch_checkpoint = ModelCheckpoint(
        filename="epoch-{epoch:03d}-{step}",
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,
    )
    
    # Build callbacks list
    callbacks_list = [
        LearningRateMonitor(),
        latest_checkpoint,
        val_checkpoint,
        epoch_checkpoint,
    ]
    
    # Add early stopping if enabled
    if enable_early_stopping:
        early_stop_callback = CustomEarlyStopping(
            monitor=early_stopping_monitor,
            patience=early_stopping_patience,
            verbose=True,
            mode='min',
            min_delta=0.0,
        )
        callbacks_list.append(early_stop_callback)
        print(f"✓ Early stopping enabled (patience={early_stopping_patience} epochs, monitor={early_stopping_monitor})")
    
    # Add FID monitoring if enabled (only for VDM-like models)
    if enable_fid_monitoring and model_type in ['vdm', 'triple']:
        fid_callback = FIDMonitorCallback(
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=datamodule.val_dataloader(),
            compute_every_n_epochs=fid_compute_every_n_epochs,
            n_samples=fid_n_samples,
            channel_names=['DM', 'Gas', 'Stars'],
            verbose=True
        )
        callbacks_list.append(fid_callback)
        print(f"✓ FID monitoring enabled (every {fid_compute_every_n_epochs} epochs, {fid_n_samples} samples)")
    
    # Add gradient monitoring if enabled
    if enable_gradient_monitoring:
        grad_callback = GradientMonitorCallback(
            log_every_n_steps=gradient_log_frequency,
            verbose=False
        )
        callbacks_list.append(grad_callback)
        print(f"✓ Gradient monitoring enabled (every {gradient_log_frequency} steps)")
    
    # Add EMA callback if enabled
    if enable_ema:
        ema_callback = EMACallback(
            decay=ema_decay,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every,
            use_ema_for_validation=True,
            save_ema_weights=True,
        )
        callbacks_list.append(ema_callback)
        print(f"✓ EMA enabled (decay={ema_decay}, warmup={ema_update_after_step} steps)")

    # Device configuration
    if cpu_only:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
        num_devices = 1
        print("="*80)
        print("⚠️  CPU-ONLY MODE (FOR TESTING)")
        print("="*80)
        print("Training will be MUCH slower than GPU.")
        print("Use this mode only for:")
        print("  - Testing code changes")
        print("  - Verifying configuration")
        print("  - Debugging issues")
        print("For production training, use GPU!")
        print("="*80)
    else:
        accelerator = "auto"
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_devices > 1:
            if find_unused_parameters:
                strategy = "ddp_find_unused_parameters_true"
            else:
                strategy = "ddp"
            devices = num_devices
            print(f"✓ Using DDP strategy with {num_devices} GPUs (find_unused_parameters={find_unused_parameters})")
            print(f"  Global batch size: {datamodule.batch_size}")
            print(f"  Per-GPU batch size: {datamodule.batch_size // num_devices}")
        else:
            strategy = "auto"
            devices = 1
            print(f"✓ Using single device (GPU or CPU auto-detect)")

    # Print speed optimization settings
    print(f"\n🚀 SPEED OPTIMIZATIONS:")
    print(f"  Precision: {precision}")
    print(f"  torch.compile: {'enabled' if compile_model else 'disabled'}")
    print(f"  Gradient clipping: {gradient_clip_val if gradient_clip_val else 'disabled'}")
    print(f"  Accumulate grad batches: {accumulate_grad_batches}")

    trainer = Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks_list,
        sync_batchnorm=True if num_devices > 1 else False,
        precision=precision,
        use_distributed_sampler=True if num_devices > 1 else False,
    )
    
    # Optional: torch.compile for additional speedup
    if compile_model and not cpu_only:
        print("🔧 Compiling model with torch.compile (first epoch will be slower)...")
        model = torch.compile(model)

    # Print training start banner
    print("\n" + "="*60)
    print(f"STARTING {model_type.upper()} TRAINING")
    print("="*60)
    print(f"  Model: {model_name}")
    print(f"  Type: {model_type}")
    print(f"  Dataset: {dataset}")
    print(f"  Box size: {boxsize} Mpc/h")
    print(f"  Max epochs: {max_epochs}")
    print(f"  TensorBoard logs: {tb_logs}/{model_name}")
    print("="*60 + "\n")

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    return trainer


# =============================================================================
# Parameter Loading Utilities
# =============================================================================

def load_param_normalization(param_norm_path):
    """Load parameter normalization bounds from CSV file."""
    if param_norm_path is None or (isinstance(param_norm_path, str) and param_norm_path.lower() in ['none', '']):
        return False, None, None, 0
    
    if not os.path.exists(param_norm_path):
        raise FileNotFoundError(f"Parameter norm path not found: {param_norm_path}")
    
    import pandas as pd
    minmax_df = pd.read_csv(param_norm_path)
    min_vals = np.array(minmax_df['MinVal'])
    max_vals = np.array(minmax_df['MaxVal'])
    
    if len(min_vals) != len(max_vals):
        raise ValueError("Mismatch in number of min and max values")
    
    n_params = len(min_vals)
    print(f"  Loaded {n_params} parameter bounds from {param_norm_path}")
    return True, min_vals, max_vals, n_params


# =============================================================================
# Model Creation Functions
# =============================================================================

def create_vdm_model(cfg, min_vals, max_vals, use_param_conditioning):
    """Create VDM (3-channel) model."""
    from vdm.vdm_model_clean import LightCleanVDM
    
    # Parse data noise (can be single value or per-channel)
    data_noise_str = cfg.get_str('data_noise', '1e-3')
    if ',' in data_noise_str:
        data_noise = cfg.get_list_float('data_noise', '1e-3,1e-3,1e-3')
    else:
        data_noise = cfg.get_float('data_noise', 1e-3)
    
    input_channels = 3  # [DM, Gas, Stars]
    
    score_model = UNetVDM(
        input_channels=input_channels,
        conditioning_channels=1,
        large_scale_channels=cfg.get_int('large_scale_channels', 3),
        gamma_min=cfg.get_float('gamma_min', -13.3),
        gamma_max=cfg.get_float('gamma_max', 5.0),
        embedding_dim=cfg.get_int('embedding_dim', 96),
        norm_groups=cfg.get_int('norm_groups', 8),
        n_blocks=cfg.get_int('n_blocks', 5),
        add_attention=cfg.get_bool('add_attention', True),
        n_attention_heads=cfg.get_int('n_attention_heads', 8),
        use_fourier_features=cfg.get_bool('use_fourier_features', True),
        legacy_fourier=cfg.get_bool('legacy_fourier', False),
        use_param_conditioning=use_param_conditioning,
        param_min=min_vals,
        param_max=max_vals,
        use_param_prediction=cfg.get_bool('use_param_prediction', False),
        use_auxiliary_mask=False,
        # Cross-attention parameters
        use_cross_attention=cfg.get_bool('use_cross_attention', False),
        cross_attention_location=cfg.get_str('cross_attention_location', 'bottleneck'),
        cross_attention_heads=cfg.get_int('cross_attention_heads', 8),
        cross_attention_dropout=cfg.get_float('cross_attention_dropout', 0.1),
        use_chunked_cross_attention=cfg.get_bool('use_chunked_cross_attention', True),
        cross_attention_chunk_size=cfg.get_int('cross_attention_chunk_size', 512),
        downsample_cross_attn_cond=cfg.get_bool('downsample_cross_attn_cond', True),
        cross_attn_cond_downsample_factor=cfg.get_int('cross_attn_cond_downsample_factor', 4),
        cross_attn_max_resolution=cfg.get_int('cross_attn_max_resolution', 128),
    )
    
    model = LightCleanVDM(
        score_model=score_model,
        learning_rate=cfg.get_float('learning_rate', 1e-4),
        lr_scheduler=cfg.get_str('lr_scheduler', 'onecycle'),
        gamma_min=cfg.get_float('gamma_min', -13.3),
        gamma_max=cfg.get_float('gamma_max', 5.0),
        image_shape=(input_channels, cfg.get_int('cropsize', 128), cfg.get_int('cropsize', 128)),
        noise_schedule=cfg.get_str('noise_schedule', 'learned'),
        data_noise=data_noise,
        antithetic_time_sampling=cfg.get_bool('antithetic_time_sampling', True),
        lambdas=cfg.get_list_float('lambdas', '1.0,1.0,1.0'),
        channel_weights=cfg.get_list_float('channel_weights', '1.0,1.0,1.0'),
        use_focal_loss=cfg.get_bool('use_focal_loss', False),
        focal_gamma=cfg.get_float('focal_gamma', 2.0),
        use_param_prediction=cfg.get_bool('use_param_prediction', False),
        param_prediction_weight=cfg.get_float('param_prediction_weight', 0.01),
    )
    
    return model, 'val/elbo'


def create_triple_model(cfg, min_vals, max_vals, use_param_conditioning):
    """Create Triple VDM (3 independent single-channel VDMs) model."""
    from vdm.vdm_model_triple import LightTripleVDM
    
    # Common UNet parameters
    unet_params = {
        'input_channels': 1,  # Single channel per model
        'conditioning_channels': 1,
        'large_scale_channels': cfg.get_int('large_scale_channels', 3),
        'gamma_min': cfg.get_float('gamma_min', -13.3),
        'gamma_max': cfg.get_float('gamma_max', 5.0),
        'embedding_dim': cfg.get_int('embedding_dim', 96),
        'norm_groups': cfg.get_int('norm_groups', 8),
        'n_blocks': cfg.get_int('n_blocks', 5),
        'add_attention': cfg.get_bool('add_attention', True),
        'n_attention_heads': cfg.get_int('n_attention_heads', 8),
        'use_fourier_features': cfg.get_bool('use_fourier_features', True),
        'legacy_fourier': cfg.get_bool('legacy_fourier', False),
        'use_param_conditioning': use_param_conditioning,
        'param_min': min_vals,
        'param_max': max_vals,
        'use_param_prediction': cfg.get_bool('use_param_prediction', False),
        'use_auxiliary_mask': False,
        # Cross-attention
        'use_cross_attention': cfg.get_bool('use_cross_attention', False),
        'cross_attention_location': cfg.get_str('cross_attention_location', 'bottleneck'),
        'cross_attention_heads': cfg.get_int('cross_attention_heads', 8),
        'cross_attention_dropout': cfg.get_float('cross_attention_dropout', 0.1),
        'use_chunked_cross_attention': cfg.get_bool('use_chunked_cross_attention', True),
        'cross_attention_chunk_size': cfg.get_int('cross_attention_chunk_size', 512),
        'downsample_cross_attn_cond': cfg.get_bool('downsample_cross_attn_cond', True),
        'cross_attn_cond_downsample_factor': cfg.get_int('cross_attn_cond_downsample_factor', 4),
        'cross_attn_max_resolution': cfg.get_int('cross_attn_max_resolution', 128),
    }
    
    # Create three separate UNet models
    print("\nCreating three separate UNet models...")
    hydro_dm_score_model = UNetVDM(**unet_params)
    gas_score_model = UNetVDM(**unet_params)
    stars_score_model = UNetVDM(**unet_params)
    print("✓ Models created")
    
    cropsize = cfg.get_int('cropsize', 128)
    
    model = LightTripleVDM(
        hydro_dm_score_model=hydro_dm_score_model,
        gas_score_model=gas_score_model,
        stars_score_model=stars_score_model,
        learning_rate=cfg.get_float('learning_rate', 1e-4),
        lr_scheduler=cfg.get_str('lr_scheduler', 'onecycle'),
        gamma_min=cfg.get_float('gamma_min', -13.3),
        gamma_max=cfg.get_float('gamma_max', 5.0),
        image_shape=(1, cropsize, cropsize),
        noise_schedule=cfg.get_str('noise_schedule', 'learned'),
        data_noise=cfg.get_float('data_noise', 1e-3),
        antithetic_time_sampling=cfg.get_bool('antithetic_time_sampling', True),
        lambdas=cfg.get_list_float('lambdas', '1.0,1.0,1.0'),
        channel_weights=cfg.get_list_float('channel_weights', '1.0,1.0,1.0'),
        use_focal_loss_hydro_dm=cfg.get_bool('use_focal_loss_hydro_dm', False),
        use_focal_loss_gas=cfg.get_bool('use_focal_loss_gas', False),
        use_focal_loss_stars=cfg.get_bool('use_focal_loss_stars', False),
        focal_gamma=cfg.get_float('focal_gamma', 2.0),
        use_param_prediction=cfg.get_bool('use_param_prediction', False),
        param_prediction_weight=cfg.get_float('param_prediction_weight', 0.01),
    )
    
    return model, 'val/elbo'


def create_ddpm_model(cfg, use_param_conditioning, n_params):
    """Create DDPM/NCSNpp model."""
    from vdm.ddpm_model import LightScoreModel, SCORE_MODELS_AVAILABLE
    
    if not SCORE_MODELS_AVAILABLE:
        raise ImportError(
            "score_models package not found!\n"
            "Install with: pip install score_models\n"
            "Or: pip install git+https://github.com/AlexandreAdam/score_models.git"
        )
    
    from score_models import NCSNpp, DDPM
    
    architecture = cfg.get_str('architecture', 'ncsnpp').lower()
    large_scale_channels = cfg.get_int('large_scale_channels', 3)
    conditioning_channels = 1 + large_scale_channels
    output_channels = 3
    
    nf = cfg.get_int('nf', 128)
    ch_mult = cfg.get_list_int('ch_mult', '1,2,2,4')
    attention = cfg.get_bool('attention', True)
    
    sde_type = cfg.get_str('sde', 'vp').lower()
    
    print(f"\nCreating {architecture.upper()} model...")
    
    if architecture == 'ncsnpp':
        condition_types = ["input"]
        condition_kwargs = {"condition_input_channels": conditioning_channels}
        
        if use_param_conditioning:
            condition_types.append("vector")
            condition_kwargs["condition_vector_channels"] = n_params
        
        net = NCSNpp(
            channels=output_channels,
            dimensions=2,
            nf=nf,
            ch_mult=ch_mult,
            attention=attention,
            condition=condition_types,
            **condition_kwargs,
        )
        
        model = LightScoreModel(
            model=net,
            sde=sde_type,
            beta_min=cfg.get_float('beta_min', 0.1),
            beta_max=cfg.get_float('beta_max', 20.0),
            sigma_min=cfg.get_float('sigma_min', 0.01),
            sigma_max=cfg.get_float('sigma_max', 50.0),
            learning_rate=cfg.get_float('learning_rate', 1e-4),
            lr_scheduler=cfg.get_str('lr_scheduler', 'cosine'),
            ema_decay=cfg.get_float('ema_decay', 0.9999),
            use_param_conditioning=use_param_conditioning,
        )
    
    elif architecture == 'ddpm':
        from vdm.ddpm_model import ConditionedDDPMWrapper
        
        num_res_blocks = cfg.get_int('num_res_blocks', 2)
        dropout = cfg.get_float('dropout', 0.1)
        
        net = DDPM(
            channels=output_channels + conditioning_channels,
            dimensions=2,
            nf=nf,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attention=attention,
            dropout=dropout,
        )
        
        wrapped_net = ConditionedDDPMWrapper(
            net=net,
            output_channels=output_channels,
            conditioning_channels=conditioning_channels,
        )
        
        model = LightScoreModel(
            model=wrapped_net,
            sde=sde_type,
            beta_min=cfg.get_float('beta_min', 0.1),
            beta_max=cfg.get_float('beta_max', 20.0),
            sigma_min=cfg.get_float('sigma_min', 0.01),
            sigma_max=cfg.get_float('sigma_max', 50.0),
            learning_rate=cfg.get_float('learning_rate', 1e-4),
            lr_scheduler=cfg.get_str('lr_scheduler', 'cosine'),
            ema_decay=cfg.get_float('ema_decay', 0.9999),
            use_param_conditioning=False,  # DDPM wrapper doesn't support vector conditioning
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'ncsnpp' or 'ddpm'.")
    
    return model, 'val/loss'


def create_dsm_model(cfg, min_vals, max_vals, use_param_conditioning):
    """Create DSM (Denoising Score Matching) model with custom UNet."""
    from vdm.dsm_model import LightDSM
    
    conditioning_channels = cfg.get_int('conditioning_channels', 1)
    large_scale_channels = cfg.get_int('large_scale_channels', 3)
    output_channels = 3
    
    print(f"\nCreating UNetVDM model for DSM...")
    
    unet = UNetVDM(
        input_channels=output_channels,
        conditioning_channels=conditioning_channels,
        large_scale_channels=large_scale_channels,
        embedding_dim=cfg.get_int('embedding_dim', 96),
        n_blocks=cfg.get_int('n_blocks', 5),
        norm_groups=cfg.get_int('norm_groups', 8),
        n_attention_heads=cfg.get_int('n_attention_heads', 8),
        use_fourier_features=cfg.get_bool('use_fourier_features', True),
        legacy_fourier=cfg.get_bool('fourier_legacy', False),
        add_attention=cfg.get_bool('add_attention', True),
        use_param_conditioning=use_param_conditioning,
        param_min=min_vals,
        param_max=max_vals,
    )
    
    model = LightDSM(
        score_model=unet,
        beta_min=cfg.get_float('beta_min', 0.1),
        beta_max=cfg.get_float('beta_max', 20.0),
        learning_rate=cfg.get_float('learning_rate', 1e-4),
        weight_decay=cfg.get_float('weight_decay', 1e-5),
        lr_scheduler=cfg.get_str('lr_scheduler', 'cosine'),
        n_sampling_steps=cfg.get_int('n_sampling_steps', 250),
        use_param_conditioning=use_param_conditioning,
        use_snr_weighting=cfg.get_bool('use_snr_weighting', True),
        channel_weights=cfg.get_list_float('channel_weights', '1.0,1.0,1.0'),
    )
    
    return model, 'val/loss'


def create_interpolant_model(cfg, min_vals, max_vals, use_param_conditioning):
    """Create Interpolant (Flow Matching) model."""
    from vdm.interpolant_model import LightInterpolant, VelocityNetWrapper
    
    conditioning_channels = cfg.get_int('conditioning_channels', 1)
    large_scale_channels = cfg.get_int('large_scale_channels', 3)
    total_conditioning_channels = conditioning_channels + large_scale_channels
    output_channels = 3
    
    print(f"\nCreating UNetVDM model for Interpolant...")
    
    unet = UNetVDM(
        input_channels=output_channels,
        conditioning_channels=conditioning_channels,
        large_scale_channels=large_scale_channels,
        embedding_dim=cfg.get_int('embedding_dim', 256),
        n_blocks=cfg.get_int('n_blocks', 32),
        norm_groups=cfg.get_int('norm_groups', 8),
        n_attention_heads=cfg.get_int('n_attention_heads', 8),
        use_fourier_features=cfg.get_bool('use_fourier_features', True),
        legacy_fourier=cfg.get_bool('fourier_legacy', False),
        add_attention=cfg.get_bool('add_attention', True),
        use_param_conditioning=use_param_conditioning,
        param_min=min_vals,
        param_max=max_vals,
    )
    
    velocity_model = VelocityNetWrapper(
        net=unet,
        output_channels=output_channels,
        conditioning_channels=total_conditioning_channels,
    )
    
    model = LightInterpolant(
        velocity_model=velocity_model,
        learning_rate=cfg.get_float('learning_rate', 1e-4),
        weight_decay=cfg.get_float('weight_decay', 1e-5),
        lr_scheduler=cfg.get_str('lr_scheduler', 'cosine'),
        n_sampling_steps=cfg.get_int('n_sampling_steps', 50),
        use_stochastic_interpolant=cfg.get_bool('use_stochastic_interpolant', False),
        sigma=cfg.get_float('sigma', 0.0),
        x0_mode=cfg.get_str('x0_mode', 'zeros'),
        use_param_conditioning=use_param_conditioning,
    )
    
    return model, 'val/loss'


def create_ot_flow_model(cfg, min_vals, max_vals, use_param_conditioning):
    """Create OT Flow Matching model."""
    from vdm.ot_flow_model import LightOTFlow, OTVelocityNetWrapper
    
    conditioning_channels = cfg.get_int('conditioning_channels', 1)
    large_scale_channels = cfg.get_int('large_scale_channels', 3)
    total_conditioning_channels = conditioning_channels + large_scale_channels
    output_channels = 3
    
    print(f"\nCreating UNetVDM model for OT Flow...")
    
    unet = UNetVDM(
        input_channels=output_channels,
        conditioning_channels=conditioning_channels,
        large_scale_channels=large_scale_channels,
        embedding_dim=cfg.get_int('embedding_dim', 256),
        n_blocks=cfg.get_int('n_blocks', 32),
        norm_groups=cfg.get_int('norm_groups', 8),
        n_attention_heads=cfg.get_int('n_attention_heads', 8),
        use_fourier_features=cfg.get_bool('use_fourier_features', True),
        legacy_fourier=cfg.get_bool('fourier_legacy', False),
        add_attention=cfg.get_bool('add_attention', True),
        use_param_conditioning=use_param_conditioning,
        param_min=min_vals,
        param_max=max_vals,
    )
    
    velocity_model = OTVelocityNetWrapper(
        net=unet,
        output_channels=output_channels,
        conditioning_channels=total_conditioning_channels,
    )
    
    model = LightOTFlow(
        velocity_model=velocity_model,
        learning_rate=cfg.get_float('learning_rate', 1e-4),
        weight_decay=cfg.get_float('weight_decay', 1e-5),
        lr_scheduler=cfg.get_str('lr_scheduler', 'cosine'),
        n_sampling_steps=cfg.get_int('n_sampling_steps', 50),
        ot_method=cfg.get_str('ot_method', 'exact'),
        ot_reg=cfg.get_float('ot_reg', 0.01),
        use_stochastic_interpolant=cfg.get_bool('use_stochastic_interpolant', False),
        sigma=cfg.get_float('sigma', 0.0),
        x0_mode=cfg.get_str('x0_mode', 'zeros'),
        use_param_conditioning=use_param_conditioning,
        use_ot_training=cfg.get_bool('use_ot_training', True),
        ot_warmup_epochs=cfg.get_int('ot_warmup_epochs', 0),
    )
    
    return model, 'val/loss'


def create_consistency_model(cfg, min_vals, max_vals, use_param_conditioning):
    """Create Consistency model."""
    from vdm.consistency_model import (
        LightConsistency,
        ConsistencyModel,
        ConsistencyFunction,
        ConsistencyNetWrapper,
        ConsistencyNoiseSchedule,
    )
    
    conditioning_channels = cfg.get_int('conditioning_channels', 1)
    large_scale_channels = cfg.get_int('large_scale_channels', 3)
    total_conditioning_channels = conditioning_channels + large_scale_channels
    output_channels = 3
    
    print(f"\nCreating UNetVDM model for Consistency...")
    
    unet = UNetVDM(
        input_channels=output_channels,
        conditioning_channels=conditioning_channels,
        large_scale_channels=large_scale_channels,
        embedding_dim=cfg.get_int('embedding_dim', 256),
        n_blocks=cfg.get_int('n_blocks', 32),
        norm_groups=cfg.get_int('norm_groups', 8),
        n_attention_heads=cfg.get_int('n_attention_heads', 8),
        use_fourier_features=cfg.get_bool('use_fourier_features', True),
        legacy_fourier=cfg.get_bool('fourier_legacy', False),
        add_attention=cfg.get_bool('add_attention', True),
        use_param_conditioning=use_param_conditioning,
        param_min=min_vals,
        param_max=max_vals,
    )
    
    net_wrapper = ConsistencyNetWrapper(
        net=unet,
        output_channels=output_channels,
        conditioning_channels=total_conditioning_channels,
    )
    
    sigma_min = cfg.get_float('sigma_min', 0.002)
    sigma_max = cfg.get_float('sigma_max', 80.0)
    sigma_data = cfg.get_float('sigma_data', 0.5)
    
    noise_schedule = ConsistencyNoiseSchedule(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    
    consistency_fn = ConsistencyFunction(
        net=net_wrapper,
        sigma_data=sigma_data,
        sigma_min=sigma_min,
    )
    
    consistency_model = ConsistencyModel(
        consistency_fn=consistency_fn,
        noise_schedule=noise_schedule,
        sigma_data=sigma_data,
    )
    
    model = LightConsistency(
        consistency_model=consistency_model,
        learning_rate=cfg.get_float('learning_rate', 1e-4),
        weight_decay=cfg.get_float('weight_decay', 1e-5),
        lr_scheduler=cfg.get_str('lr_scheduler', 'cosine'),
        n_sampling_steps=cfg.get_int('n_sampling_steps', 1),
        use_param_conditioning=use_param_conditioning,
        use_denoising_pretraining=cfg.get_bool('use_denoising_pretraining', True),
        denoising_warmup_epochs=cfg.get_int('denoising_warmup_epochs', 10),
        ct_n_steps=cfg.get_int('ct_n_steps', 18),
        ema_decay=cfg.get_float('ct_ema_decay', 0.9999),
    )
    
    return model, 'val/loss'


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for unified training."""
    parser = argparse.ArgumentParser(
        description='Unified training script for generative models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Types:
  vdm         - 3-channel VDM (Variational Diffusion Model)
  triple      - 3 independent single-channel VDMs
  ddpm        - DDPM/NCSNpp via score_models package
  dsm         - DSM (Denoising Score Matching) with custom UNet
  interpolant - Flow matching / stochastic interpolants
  ot_flow     - Optimal transport flow matching
  consistency - Consistency models (few-step sampling)

Examples:
  python train_unified.py --model vdm --config configs/clean_vdm.ini
  python train_unified.py --model interpolant --config configs/interpolant.ini --cpu_only
"""
    )
    
    parser.add_argument('--model', type=str, required=True, choices=MODEL_TYPES,
                        help='Model type to train')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (uses default if not specified)')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force training on CPU (for testing)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Use default config if not specified
    config_path = args.config if args.config else DEFAULT_CONFIGS.get(args.model)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if 'TRAINING' not in config:
        raise KeyError("TRAINING section not found in config file")
    
    params = config['TRAINING']
    cfg = ConfigParser(params)
    
    # Set seed
    seed = cfg.get_int('seed', 8)
    seed_everything(seed)
    
    # Print header
    print("\n" + "="*80)
    print(f"UNIFIED TRAINING: {args.model.upper()}")
    print("="*80)
    print(f"📁 Configuration: {config_path}")
    print(f"🎲 Seed: {seed}")
    
    # Common parameters
    dataset = cfg.get_str('dataset', 'IllustrisTNG')
    data_root = cfg.get_str('data_root', '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/')
    boxsize = cfg.get_float('boxsize', 6.25)
    batch_size = cfg.get_int('batch_size', 16)
    num_workers = cfg.get_int('num_workers', 8)
    
    print(f"📊 Dataset: {dataset}")
    print(f"📂 Data root: {data_root}")
    print(f"📦 Batch size: {batch_size}")
    
    # Check data directory exists
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Load parameter normalization
    param_norm_path = cfg.get_optional('param_norm_path')
    use_param_conditioning, min_vals, max_vals, n_params = load_param_normalization(param_norm_path)
    
    # Training parameters
    max_epochs = cfg.get_int('max_epochs', 100)
    limit_train_batches = cfg.get_float('limit_train_batches', 1.0)
    tb_logs = cfg.get_str('tb_logs', '/mnt/home/mlee1/ceph/tb_logs/')
    model_name = cfg.get_str('model_name', f'{args.model}_model')
    version = cfg.get_optional_int('version')
    
    # Callback parameters
    enable_early_stopping = cfg.get_bool('enable_early_stopping', True)
    early_stopping_patience = cfg.get_int('early_stopping_patience', 300)
    enable_fid_monitoring = cfg.get_bool('enable_fid_monitoring', False)
    fid_compute_every_n_epochs = cfg.get_int('fid_compute_every_n_epochs', 1)
    fid_n_samples = cfg.get_int('fid_n_samples', 100)
    enable_gradient_monitoring = cfg.get_bool('enable_gradient_monitoring', True)
    gradient_log_frequency = cfg.get_int('gradient_log_frequency', 100)
    
    # EMA parameters
    enable_ema = cfg.get_bool('enable_ema', args.model not in ['vdm', 'triple'])  # EMA default off for VDM
    ema_decay = cfg.get_float('ema_decay', 0.9999)
    ema_update_after_step = cfg.get_int('ema_update_after_step', 0)
    ema_update_every = cfg.get_int('ema_update_every', 1)
    
    # Memory/speed optimization
    accumulate_grad_batches = cfg.get_int('accumulate_grad_batches', 1)
    precision = cfg.get_str('precision', '32')
    compile_model = cfg.get_bool('compile_model', False)
    
    # Gradient clipping (model-specific defaults)
    if args.model == 'ddpm':
        gradient_clip_val = None  # DSM loss has naturally well-behaved gradients
    else:
        gradient_clip_val = cfg.get_float('gradient_clip_val', 1.0)
    
    # Fast ablation (optional sample limiting)
    limit_train_samples = cfg.get_optional_int('limit_train_samples')
    limit_val_samples = cfg.get_optional_int('limit_val_samples')
    
    # Stellar normalization
    quantile_path = cfg.get_optional('quantile_path')
    
    # Print fast ablation settings if enabled
    if limit_train_samples or limit_val_samples or limit_train_batches < 1.0:
        print(f"\n⚡ FAST ABLATION MODE:")
        if limit_train_samples:
            print(f"  Training samples: {limit_train_samples}")
        if limit_val_samples:
            print(f"  Validation samples: {limit_val_samples}")
        if limit_train_batches < 1.0:
            print(f"  Training batches: {limit_train_batches*100:.0f}%")
    
    # Load data
    print(f"\n📦 Loading data from {data_root}...")
    
    datamodule_kwargs = {
        'dataset': dataset,
        'data_root': data_root,
        'batch_size': batch_size,
        'num_workers': num_workers,
    }
    
    # Add optional parameters if get_astro_data supports them
    if limit_train_samples is not None:
        datamodule_kwargs['limit_train_samples'] = limit_train_samples
    if limit_val_samples is not None:
        datamodule_kwargs['limit_val_samples'] = limit_val_samples
    if quantile_path is not None:
        datamodule_kwargs['quantile_path'] = quantile_path
    
    # For VDM models, also add stellar_stats_path if applicable
    if args.model in ['vdm', 'triple'] and quantile_path is None:
        stellar_stats_path = cfg.get_optional('stellar_stats_path')
        if stellar_stats_path:
            datamodule_kwargs['stellar_stats_path'] = stellar_stats_path
    
    datamodule = get_astro_data(**datamodule_kwargs)
    
    # Create model based on type
    print(f"\n🔧 Creating {args.model.upper()} model...")
    
    # DDP settings - Triple VDM needs find_unused_parameters
    find_unused_parameters = args.model in ['triple', 'interpolant', 'ot_flow', 'consistency']
    
    if args.model == 'vdm':
        model, early_stopping_monitor = create_vdm_model(cfg, min_vals, max_vals, use_param_conditioning)
    elif args.model == 'triple':
        model, early_stopping_monitor = create_triple_model(cfg, min_vals, max_vals, use_param_conditioning)
    elif args.model == 'ddpm':
        model, early_stopping_monitor = create_ddpm_model(cfg, use_param_conditioning, n_params)
    elif args.model == 'dsm':
        model, early_stopping_monitor = create_dsm_model(cfg, min_vals, max_vals, use_param_conditioning)
    elif args.model == 'interpolant':
        model, early_stopping_monitor = create_interpolant_model(cfg, min_vals, max_vals, use_param_conditioning)
    elif args.model == 'ot_flow':
        model, early_stopping_monitor = create_ot_flow_model(cfg, min_vals, max_vals, use_param_conditioning)
    elif args.model == 'consistency':
        model, early_stopping_monitor = create_consistency_model(cfg, min_vals, max_vals, use_param_conditioning)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 MODEL PARAMETERS:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Train
    train(
        model=model,
        datamodule=datamodule,
        model_name=model_name,
        model_type=args.model,
        version=version,
        dataset=dataset,
        boxsize=boxsize,
        # Callbacks
        enable_early_stopping=enable_early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_monitor=early_stopping_monitor,
        enable_fid_monitoring=enable_fid_monitoring,
        fid_compute_every_n_epochs=fid_compute_every_n_epochs,
        fid_n_samples=fid_n_samples,
        enable_gradient_monitoring=enable_gradient_monitoring,
        gradient_log_frequency=gradient_log_frequency,
        # Training
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        tb_logs=tb_logs,
        cpu_only=args.cpu_only,
        # EMA
        enable_ema=enable_ema,
        ema_decay=ema_decay,
        ema_update_after_step=ema_update_after_step,
        ema_update_every=ema_update_every,
        # Memory/speed
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
        compile_model=compile_model,
        # Resume
        resume_checkpoint=args.resume,
        # DDP
        find_unused_parameters=find_unused_parameters,
    )


if __name__ == "__main__":
    main()
