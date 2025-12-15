"""
Training script for CleanVDM model.

This script trains the CleanVDM model on astrophysical data using PyTorch Lightning.
Based on train_model_enhanced.py but adapted for the simplified CleanVDM architecture.

Key differences from train_model_enhanced.py:
- Uses CleanVDM instead of LightEnhancedVDM
- Simplified loss structure (no progressive weighting, mass conservation, etc.)
- 3-channel mode: [DM, Gas, Stars] (all log-transformed and normalized)
- Optional focal loss for stellar channel
- Optional parameter prediction auxiliary task
"""

import os
import numpy as np
import configparser
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from vdm.astro_dataset import get_astro_data
from vdm.vdm_model_clean import LightCleanVDM  # Lightning wrapper
from vdm.networks_clean import UNetVDM  # Updated to use networks_clean
from vdm.callbacks import (
    FIDMonitorCallback, 
    GradientMonitorCallback, 
    CustomEarlyStopping,
    EMACallback,
)

torch.set_float32_matmul_precision("medium")


def train(
    model,
    datamodule,
    model_name,
    version=None,
    dataset='IllustrisTNG',
    boxsize=6.25,
    enable_early_stopping=True,
    early_stopping_patience=300,
    enable_fid_monitoring=True,
    fid_compute_every_n_epochs=1,
    fid_n_samples=100,
    enable_gradient_monitoring=True,
    gradient_log_frequency=100,
    max_epochs=100,
    limit_train_batches=1.0,
    tb_logs='tb_logs',
    cpu_only=False,
    # EMA parameters
    enable_ema=False,
    ema_decay=0.9999,
    ema_update_after_step=0,
    ema_update_every=1,
    # Speed optimizations
    precision="32",
    compile_model=False,
):
    """
    Train the CleanVDM model.
    
    Args:
        model: LightCleanVDM model
        datamodule: PyTorch Lightning DataModule
        model_name: Name for logging and checkpoints
        dataset: Dataset name (for reference)
        boxsize: Physical size of box (for reference)
        enable_early_stopping: Enable early stopping callback
        early_stopping_patience: Epochs to wait before stopping
        enable_fid_monitoring: Enable FID metric computation
        fid_compute_every_n_epochs: Epochs between FID computations
        fid_n_samples: Number of samples for FID computation
        enable_gradient_monitoring: Monitor gradient statistics
        gradient_log_frequency: Steps between gradient logging
        max_epochs: Maximum training epochs
        limit_train_batches: Fraction of training data per epoch
        tb_logs: TensorBoard logs directory
        cpu_only: Force CPU training (for testing only, much slower than GPU)
        enable_ema: Enable Exponential Moving Average of weights (recommended for diffusion)
        ema_decay: EMA decay factor (0.9999 is common for diffusion models)
        ema_update_after_step: Start EMA updates after this many steps (warmup)
        ema_update_every: Update EMA every N steps
    """
    
    ckpt_path = None
    
    # TensorBoard logger - use explicit version to avoid nested directories
    comet_logger = TensorBoardLogger(tb_logs, name=model_name, version=version)

    # Checkpoint every time val/elbo improves (keep only best 3)
    val_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-val_elbo={val/elbo:.3f}",
        monitor="val/elbo",
        mode="min",
        save_top_k=3,
        save_weights_only=True,  # Much faster - skip optimizer state
    )

    # Checkpoint at every 6000 steps (keep only latest 3)
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
        every_n_train_steps=6000,
        save_top_k=3,
        save_weights_only=True,
    )
    
    # Checkpoint every 5 epochs (not every epoch - too slow on Ceph)
    epoch_checkpoint = ModelCheckpoint(
        filename="epoch-{epoch:03d}-{step}",
        save_top_k=-1,  # Keep all epoch checkpoints (no monitor needed)
        every_n_epochs=5,
        save_weights_only=True,
    )
    
    # Build callbacks list (validation plots removed for faster training)
    callbacks_list = [
        LearningRateMonitor(),
        latest_checkpoint,
        val_checkpoint,
        epoch_checkpoint,
    ]
    
    # Add early stopping if enabled
    if enable_early_stopping:
        early_stop_callback = CustomEarlyStopping(
            monitor='val/elbo',
            patience=early_stopping_patience,
            verbose=True,
            mode='min',
            min_delta=0.0,
        )
        callbacks_list.append(early_stop_callback)
        print(f"✓ Early stopping enabled (patience={early_stopping_patience} epochs)")
    
    # Add FID monitoring if enabled
    if enable_fid_monitoring:
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
    
    # Print speed optimization settings
    print(f"\n🚀 SPEED OPTIMIZATIONS:")
    print(f"  Precision: {precision}")
    if compile_model:
        print(f"  torch.compile: enabled (first epoch will be slower)")
    else:
        print(f"  torch.compile: disabled")

    # Auto-detect number of available devices and set strategy accordingly
    # NOTE: cpu_only is passed as a global or through kwargs in train() function
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
            # strategy = DDPStrategy(
            #     find_unused_parameters=False,
            #     gradient_as_bucket_view=True,  # PERFORMANCE: More efficient gradient bucketing
            #     static_graph=False,  # Set to True if model architecture doesn't change
            # )
            strategy = "ddp"
            devices = 4
            print(f"✓ Using DDP strategy with {num_devices} GPUs")
            
            print(f"  Global batch size: {datamodule.batch_size}")
            print(f"  Per-GPU batch size: {datamodule.batch_size // num_devices}")
        else:
            strategy = "auto"
            devices = 1
            print(f"✓ Using single device (GPU or CPU auto-detect)")

    trainer = Trainer(
        logger=comet_logger,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        gradient_clip_val=1.0,
        callbacks=callbacks_list,
        sync_batchnorm=True if num_devices > 1 else False,  # Only sync for DDP
        # Performance optimizations
        precision=precision,  # "32", "16-mixed", or "bf16-mixed"
        use_distributed_sampler=True if num_devices > 1 else False,  # Only for DDP
    )
    
    # Optional: torch.compile for additional speedup (requires PyTorch 2.0+)
    if compile_model:
        print("🔧 Compiling model with torch.compile (first epoch will be slower)...")
        model = torch.compile(model)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train CleanVDM model')
    parser.add_argument('--config', type=str, default='configs/clean_vdm.ini',
                       help='Path to config file')
    parser.add_argument('--cpu_only', action='store_true',
                       help='Force training on CPU (for testing, much slower)')
    
    cli_args = parser.parse_args()
    
    config = configparser.ConfigParser()
    
    # Check if config file exists
    config_path = cli_args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config.read(config_path)
    
    if 'TRAINING' not in config:
        raise KeyError("TRAINING section not found in config file")
    
    params = config['TRAINING']

    # Load parameters
    try:
        seed = int(params['seed'])
        seed_everything(seed)
        
        cropsize = int(params['cropsize'])
        batch_size = int(params['batch_size'])
        num_workers = int(params['num_workers'])
        gamma_min = float(params['gamma_min'])
        gamma_max = float(params['gamma_max'])
        dataset = params['dataset']
        embedding_dim = int(params['embedding_dim'])
        norm_groups = int(params['norm_groups'])
        n_blocks = int(params['n_blocks'])
        n_attention_heads = int(params['n_attention_heads'])
        learning_rate = float(params['learning_rate'])
        lr_scheduler = params.get('lr_scheduler', fallback='onecycle')
        noise_schedule = params['noise_schedule']
        data_root = params['data_root']
        field = params['field']
        tb_logs = params['tb_logs']
        model_name = params['model_name']
        version = params.get('version', fallback=None)
        if version is not None and version.lower() not in ['none', '']:
            version = int(version)
        else:
            version = None
        param_norm_path = params.get('param_norm_path', fallback=None)
        
        # CleanVDM specific parameters
        # Support both single value and per-channel data_noise
        data_noise_str = params.get('data_noise', fallback='1e-3')
        if ',' in data_noise_str:
            # Per-channel: parse as tuple
            data_noise = tuple(map(float, data_noise_str.split(',')))
        else:
            # Single value: parse as float
            data_noise = float(data_noise_str)
        antithetic_time_sampling = params.getboolean('antithetic_time_sampling', fallback=True)
        
        # Loss weights
        lambdas_str = params.get('lambdas', fallback='1.0,1.0,1.0')
        lambdas = tuple(map(float, lambdas_str.split(',')))
        
        channel_weights_str = params.get('channel_weights', fallback='1.0,1.0,1.0')
        channel_weights = tuple(map(float, channel_weights_str.split(',')))
        
        # Focal loss parameters
        use_focal_loss = params.getboolean('use_focal_loss', fallback=False)
        focal_gamma = float(params.get('focal_gamma', fallback=2.0))
        
        # Parameter prediction
        use_param_prediction = params.getboolean('use_param_prediction', fallback=False)
        param_prediction_weight = float(params.get('param_prediction_weight', fallback=0.01))
        
        # Large-scale field channels
        large_scale_channels = int(params.get('large_scale_channels', fallback=1))
        
        # Fourier features
        use_fourier_features = params.getboolean('use_fourier_features', fallback=True)
        legacy_fourier = params.getboolean('legacy_fourier', fallback=False)
        
        # Attention
        add_attention = params.getboolean('add_attention', fallback=True)
        
        # Cross-Attention (Phase 1, 2, 3 with speed optimizations)
        use_cross_attention = params.getboolean('use_cross_attention', fallback=False)
        cross_attention_location = params.get('cross_attention_location', fallback='bottleneck')
        cross_attention_heads = int(params.get('cross_attention_heads', fallback=8))
        cross_attention_dropout = float(params.get('cross_attention_dropout', fallback=0.1))
        use_chunked_cross_attention = params.getboolean('use_chunked_cross_attention', fallback=True)
        cross_attention_chunk_size = int(params.get('cross_attention_chunk_size', fallback=512))
        
        # Speed optimizations
        downsample_cross_attn_cond = params.getboolean('downsample_cross_attn_cond', fallback=True)
        cross_attn_cond_downsample_factor = int(params.get('cross_attn_cond_downsample_factor', fallback=4))
        cross_attn_max_resolution = int(params.get('cross_attn_max_resolution', fallback=128))
        
        # Dataset info
        boxsize = float(params.get('boxsize', fallback=6.25))
        
        # Training limits (for fast ablation)
        limit_train_samples = params.get('limit_train_samples', fallback=None)
        if limit_train_samples is not None and limit_train_samples.lower() not in ['none', '']:
            limit_train_samples = int(limit_train_samples)
        else:
            limit_train_samples = None
            
        limit_val_samples = params.get('limit_val_samples', fallback=None)
        if limit_val_samples is not None and limit_val_samples.lower() not in ['none', '']:
            limit_val_samples = int(limit_val_samples)
        else:
            limit_val_samples = None
            
        limit_train_batches = float(params.get('limit_train_batches', fallback=1.0))
        max_epochs = int(params.get('max_epochs', fallback=100))
        
        # Callbacks
        enable_early_stopping = params.getboolean('enable_early_stopping', fallback=True)
        early_stopping_patience = int(params.get('early_stopping_patience', fallback=30))
        enable_fid_monitoring = params.getboolean('enable_fid_monitoring', fallback=True)
        fid_compute_every_n_epochs = int(params.get('fid_compute_every_n_epochs', fallback=1))
        fid_n_samples = int(params.get('fid_n_samples', fallback=100))
        enable_gradient_monitoring = params.getboolean('enable_gradient_monitoring', fallback=True)
        gradient_log_frequency = int(params.get('gradient_log_frequency', fallback=100))
        
        # EMA (Exponential Moving Average)
        enable_ema = params.getboolean('enable_ema', fallback=False)
        ema_decay = float(params.get('ema_decay', fallback=0.9999))
        ema_update_after_step = int(params.get('ema_update_after_step', fallback=0))
        ema_update_every = int(params.get('ema_update_every', fallback=1))
        
        # Stellar normalization: quantile OR Z-score
        quantile_path = params.get('quantile_path', fallback=None)
        if quantile_path is not None and quantile_path.lower() in ['none', '']:
            quantile_path = None
        
        # Speed optimizations
        precision = params.get('precision', fallback='32')
        compile_model = params.getboolean('compile_model', fallback=False)

    except KeyError as e:
        raise KeyError(f"Missing required parameter in config: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid parameter value in config: {e}")
    
    # Print fast ablation settings if enabled
    if limit_train_samples or limit_val_samples or limit_train_batches < 1.0:
        print(f"\n{'='*80}")
        print(f"⚡ FAST ABLATION MODE ENABLED")
        print(f"{'='*80}")
        if limit_train_samples:
            print(f"  Training samples limited to: {limit_train_samples}")
        if limit_val_samples:
            print(f"  Validation samples limited to: {limit_val_samples}")
        if limit_train_batches < 1.0:
            print(f"  Training batches per epoch: {limit_train_batches*100:.0f}%")
        print(f"  Max epochs: {max_epochs}")
        print(f"{'='*80}\n")
    
    # Check data directory exists
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")
        
    # Load parameter normalization if specified
    if param_norm_path is None or param_norm_path.lower() == 'none':
        use_param_conditioning = False
        min_vals = None
        max_vals = None
        Nparams = 0
    else:
        use_param_conditioning = True
        if not os.path.exists(param_norm_path):
            raise FileNotFoundError(f"Parameter norm path not found: {param_norm_path}")
        
        import pandas as pd
        minmax_df = pd.read_csv(param_norm_path)
        min_vals = np.array(minmax_df['MinVal'])
        max_vals = np.array(minmax_df['MaxVal'])
        
        if len(min_vals) != len(max_vals):
            raise ValueError("Mismatch in number of min and max values")
        Nparams = len(min_vals)

    print(f"\nTraining configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Data root: {data_root}")
    print(f"  Batch size: {batch_size}")
    print(f"  Crop size: {cropsize}")
    print(f"  Number of params: {Nparams}")
    print(f"  Large-scale channels: {large_scale_channels}")
    print(f"  Fourier features: {use_fourier_features}")
    print(f"  Attention mechanism: {add_attention}")
    print(f"  Channel weights: {channel_weights}")
    print(f"  Loss weights (diffusion, latent, recons): {lambdas}")
    print(f"  Data noise: {data_noise}")
    print(f"  Antithetic time sampling: {antithetic_time_sampling}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LR scheduler: {lr_scheduler}")
    
    if use_focal_loss:
        print(f"  Focal loss: enabled (gamma={focal_gamma})")
    else:
        print(f"  Focal loss: disabled")
    
    if use_param_prediction:
        print(f"  Parameter prediction: enabled (weight={param_prediction_weight})")
    else:
        print(f"  Parameter prediction: disabled")
    
    # Print cross-attention configuration
    if use_cross_attention:
        print(f"\n🔥 CROSS-ATTENTION ENABLED (Phase 3: {cross_attention_location.upper()})")
        print(f"  Location: {cross_attention_location}")
        print(f"  Heads: {cross_attention_heads}")
        print(f"  Dropout: {cross_attention_dropout}")
        print(f"  Chunked: {use_chunked_cross_attention} (chunk_size={cross_attention_chunk_size})")
        print(f"\n  🚀 SPEED OPTIMIZATIONS:")
        print(f"    - Downsample conditioning: {downsample_cross_attn_cond} ({cross_attn_cond_downsample_factor}x)")
        print(f"    - Max resolution: {cross_attn_max_resolution}")
        print(f"    - Flash Attention: Automatic (PyTorch 2.0+)")
        print(f"    - Expected speedup: ~25-30x vs unoptimized!")
    else:
        print(f"\n  Cross-attention: disabled")
    
    # Print stellar normalization method
    if quantile_path:
        print(f"\n🌟 STELLAR NORMALIZATION: Quantile transformation")
        print(f"   Quantile transformer: {quantile_path}")
    else:
        print(f"\n🌟 STELLAR NORMALIZATION: Z-score (standard)")
        print(f"   Stellar stats: /mnt/home/mlee1/vdm_BIND/data/stellar_normalization_stats.npz")
    
    # Load data (CleanVDM uses 3-channel mode)
    # Stellar normalization: quantile (if quantile_path provided) OR Z-score (default)
    dm = get_astro_data(
        dataset,
        data_root,
        num_workers=num_workers,
        batch_size=batch_size,
        limit_train_samples=limit_train_samples,
        limit_val_samples=limit_val_samples,
        stellar_stats_path='/mnt/home/mlee1/vdm_BIND/data/stellar_normalization_stats.npz',  # Z-score (default)
        quantile_path=quantile_path  # Quantile (optional, overrides Z-score)
    )
    
    input_channels = 3  # [DM, Gas, Stars]
    print(f"  Input channels: {input_channels} (3-channel mode: [DM, Gas, Stars])")
    
    # Create model
    vdm = LightCleanVDM(
        score_model=UNetVDM(
            input_channels=input_channels,
            conditioning_channels=1,
            large_scale_channels=large_scale_channels,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            embedding_dim=embedding_dim,
            norm_groups=norm_groups,
            n_blocks=n_blocks,
            add_attention=add_attention,
            n_attention_heads=n_attention_heads,
            use_fourier_features=use_fourier_features,
            legacy_fourier=legacy_fourier,
            use_param_conditioning=use_param_conditioning,
            param_min=min_vals,
            param_max=max_vals,
            use_param_prediction=use_param_prediction,
            use_auxiliary_mask=False,  # Not used in CleanVDM (3-channel mode)
            # Cross-attention parameters (Phase 1, 2, 3 with speed optimizations)
            use_cross_attention=use_cross_attention,
            cross_attention_location=cross_attention_location,
            cross_attention_heads=cross_attention_heads,
            cross_attention_dropout=cross_attention_dropout,
            use_chunked_cross_attention=use_chunked_cross_attention,
            cross_attention_chunk_size=cross_attention_chunk_size,
            # Speed optimizations
            downsample_cross_attn_cond=downsample_cross_attn_cond,
            cross_attn_cond_downsample_factor=cross_attn_cond_downsample_factor,
            cross_attn_max_resolution=cross_attn_max_resolution,
        ),
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        image_shape=(input_channels, cropsize, cropsize),
        noise_schedule=noise_schedule,
        data_noise=data_noise,
        antithetic_time_sampling=antithetic_time_sampling,
        lambdas=lambdas,
        channel_weights=channel_weights,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        use_param_prediction=use_param_prediction,
        param_prediction_weight=param_prediction_weight,
    )
    
    # Train
    train(
        model=vdm, 
        datamodule=dm, 
        model_name=model_name,
        version=version,
        dataset=dataset,
        boxsize=boxsize,
        enable_early_stopping=enable_early_stopping,
        early_stopping_patience=early_stopping_patience,
        enable_fid_monitoring=enable_fid_monitoring,
        fid_compute_every_n_epochs=fid_compute_every_n_epochs,
        fid_n_samples=fid_n_samples,
        enable_gradient_monitoring=enable_gradient_monitoring,
        gradient_log_frequency=gradient_log_frequency,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        tb_logs=tb_logs,
        cpu_only=cli_args.cpu_only,
        # EMA parameters
        enable_ema=enable_ema,
        ema_decay=ema_decay,
        ema_update_after_step=ema_update_after_step,
        ema_update_every=ema_update_every,
        # Speed optimizations
        precision=precision,
        compile_model=compile_model,
    )
