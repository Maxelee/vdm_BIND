"""
Training script for Triple CleanVDM model.

This script trains three separate single-channel CleanVDM models simultaneously:
- Hydro DM model (channel 0)
- Gas model (channel 1)
- Stars model (channel 2)

Each model operates independently but they are trained together for efficiency.
This approach is more memory-efficient than training one 3-channel model, and
allows each model to specialize on its specific channel.

Usage:
    python train_triple_model.py --config configs/clean_vdm_triple.ini
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
from vdm.vdm_model_triple import LightTripleVDM
from vdm.networks_clean import UNetVDM
from vdm.callbacks import (
    FIDMonitorCallback, 
    GradientMonitorCallback, 
    CustomEarlyStopping,
)

torch.set_float32_matmul_precision("medium")


def train(
    model,
    datamodule,
    model_name,
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
):
    """
    Train the Triple CleanVDM model.
    
    Args:
        model: LightTripleVDM model
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
    """
    
    ckpt_path = None
    
    # TensorBoard logger
    comet_logger = TensorBoardLogger(tb_logs, name=model_name)

    # Checkpoint every time val/elbo improves
    val_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val/elbo:.3f}",
        monitor="val/elbo",
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
    
    # Checkpoint at every epoch (consistent format with DDPM/interpolant)
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

    # Auto-detect number of available devices and set strategy accordingly
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
            # Use DDPStrategy with find_unused_parameters=True for independent models
            strategy = DDPStrategy(find_unused_parameters=True)
            devices = 4
            print(f"✓ Using DDP strategy with {num_devices} GPUs")
            print(f"  ⚠️  find_unused_parameters=True (required for independent model training)")
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
        callbacks=callbacks_list,
        sync_batchnorm=True if num_devices > 1 else False,
        precision="32",
        use_distributed_sampler=True if num_devices > 1 else False,
        # Note: gradient_clip_val not supported with manual optimization
        # Gradient clipping can be done manually in training_step if needed
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Triple CleanVDM model')
    parser.add_argument('--config', type=str, default='configs/clean_vdm_triple.ini',
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
        param_norm_path = params.get('param_norm_path', fallback=None)
        
        # Triple VDM specific parameters
        data_noise_str = params.get('data_noise', fallback='1e-3')
        data_noise = float(data_noise_str)
        antithetic_time_sampling = params.getboolean('antithetic_time_sampling', fallback=True)
        
        # Loss weights
        lambdas_str = params.get('lambdas', fallback='1.0,1.0,1.0')
        lambdas = tuple(map(float, lambdas_str.split(',')))
        
        # Channel weights (for combining losses)
        channel_weights_str = params.get('channel_weights', fallback='1.0,1.0,1.0')
        channel_weights = tuple(map(float, channel_weights_str.split(',')))
        
        # Focal loss parameters (per channel)
        use_focal_loss_hydro_dm = params.getboolean('use_focal_loss_hydro_dm', fallback=False)
        use_focal_loss_gas = params.getboolean('use_focal_loss_gas', fallback=False)
        use_focal_loss_stars = params.getboolean('use_focal_loss_stars', fallback=False)
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
        
        # Cross-Attention
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
        
        # Stellar normalization: quantile OR Z-score
        quantile_path = params.get('quantile_path', fallback=None)
        if quantile_path is not None and quantile_path.lower() in ['none', '']:
            quantile_path = None

    except KeyError as e:
        raise KeyError(f"Missing required parameter in config: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid parameter value in config: {e}")
    
    # Print configuration
    print(f"\n{'='*80}")
    print(f"TRIPLE VDM TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Training three separate single-channel models:")
    print(f"  Model 1: Hydro DM (dark matter)")
    print(f"  Model 2: Gas")
    print(f"  Model 3: Stars")
    print(f"\nDataset: {dataset}")
    print(f"Data root: {data_root}")
    print(f"Batch size: {batch_size}")
    print(f"Crop size: {cropsize}")
    print(f"Large-scale channels: {large_scale_channels}")
    print(f"Channel weights: {channel_weights}")
    print(f"Loss weights (diffusion, latent, recons): {lambdas}")
    print(f"Data noise: {data_noise}")
    print(f"Learning rate: {learning_rate}")
    print(f"LR scheduler: {lr_scheduler}")
    
    # Print focal loss settings
    if use_focal_loss_hydro_dm or use_focal_loss_gas or use_focal_loss_stars:
        print(f"\nFocal loss enabled:")
        print(f"  Hydro DM: {use_focal_loss_hydro_dm}")
        print(f"  Gas: {use_focal_loss_gas}")
        print(f"  Stars: {use_focal_loss_stars}")
        print(f"  Gamma: {focal_gamma}")
    else:
        print(f"\nFocal loss: disabled")
    
    if use_param_prediction:
        print(f"\nParameter prediction: enabled (weight={param_prediction_weight})")
    else:
        print(f"Parameter prediction: disabled")
    
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
    
    # Load data
    dm = get_astro_data(
        dataset,
        data_root,
        num_workers=num_workers,
        batch_size=batch_size,
        limit_train_samples=limit_train_samples,
        limit_val_samples=limit_val_samples,
        stellar_stats_path='/mnt/home/mlee1/vdm_BIND/data/stellar_normalization_stats.npz',
        quantile_path=quantile_path
    )
    
    # Common UNet parameters for all three models
    unet_params = {
        'input_channels': 1,  # Single channel per model
        'conditioning_channels': 1,
        'large_scale_channels': large_scale_channels,
        'gamma_min': gamma_min,
        'gamma_max': gamma_max,
        'embedding_dim': embedding_dim,
        'norm_groups': norm_groups,
        'n_blocks': n_blocks,
        'add_attention': add_attention,
        'n_attention_heads': n_attention_heads,
        'use_fourier_features': use_fourier_features,
        'legacy_fourier': legacy_fourier,
        'use_param_conditioning': use_param_conditioning,
        'param_min': min_vals,
        'param_max': max_vals,
        'use_param_prediction': use_param_prediction,
        'use_auxiliary_mask': False,
        # Cross-attention parameters
        'use_cross_attention': use_cross_attention,
        'cross_attention_location': cross_attention_location,
        'cross_attention_heads': cross_attention_heads,
        'cross_attention_dropout': cross_attention_dropout,
        'use_chunked_cross_attention': use_chunked_cross_attention,
        'cross_attention_chunk_size': cross_attention_chunk_size,
        # Speed optimizations
        'downsample_cross_attn_cond': downsample_cross_attn_cond,
        'cross_attn_cond_downsample_factor': cross_attn_cond_downsample_factor,
        'cross_attn_max_resolution': cross_attn_max_resolution,
    }
    
    # Create three separate UNet models
    print("\nCreating three separate UNet models...")
    hydro_dm_score_model = UNetVDM(**unet_params)
    gas_score_model = UNetVDM(**unet_params)
    stars_score_model = UNetVDM(**unet_params)
    print("✓ Models created")
    
    # Create triple VDM model
    vdm = LightTripleVDM(
        hydro_dm_score_model=hydro_dm_score_model,
        gas_score_model=gas_score_model,
        stars_score_model=stars_score_model,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        image_shape=(1, cropsize, cropsize),  # Single channel per model
        noise_schedule=noise_schedule,
        data_noise=data_noise,
        antithetic_time_sampling=antithetic_time_sampling,
        lambdas=lambdas,
        channel_weights=channel_weights,
        use_focal_loss_hydro_dm=use_focal_loss_hydro_dm,
        use_focal_loss_gas=use_focal_loss_gas,
        use_focal_loss_stars=use_focal_loss_stars,
        focal_gamma=focal_gamma,
        use_param_prediction=use_param_prediction,
        param_prediction_weight=param_prediction_weight,
    )
    
    # Train
    train(
        model=vdm, 
        datamodule=dm, 
        model_name=model_name,
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
    )
