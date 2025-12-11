"""
Training script for DSM (Denoising Score Matching) with custom UNet.

This script trains the DSM model using the same UNet architecture as VDM/Interpolant,
allowing fair comparison between different loss formulations:
- VDM: ELBO loss with learned gamma schedule
- DSM: Simple MSE on noise prediction with VP-SDE schedule
- Interpolant: MSE on velocity prediction with linear interpolation

Usage:
    python train_dsm.py --config configs/dsm.ini
    python train_dsm.py --config configs/dsm.ini --cpu_only  # For testing

Key features:
- Same UNet architecture as VDM (Fourier features, cross-attention, FiLM)
- VP-SDE noise schedule (beta_min, beta_max)
- DSM loss: || epsilon_hat - epsilon ||^2
- Optional SNR weighting to match VDM loss
"""

import os
import numpy as np
import configparser
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from vdm.astro_dataset import get_astro_data
from vdm.dsm_model import LightDSM
from vdm.networks_clean import UNetVDM
from vdm.callbacks import (
    GradientMonitorCallback, 
    CustomEarlyStopping,
    EMACallback,
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
    enable_gradient_monitoring=True,
    gradient_log_frequency=100,
    max_epochs=100,
    limit_train_batches=1.0,
    tb_logs='tb_logs',
    cpu_only=False,
    # EMA parameters
    enable_ema=True,
    ema_decay=0.9999,
    ema_update_after_step=0,
    ema_update_every=1,
    # Memory optimization
    accumulate_grad_batches=1,
    # Resume from checkpoint
    resume_checkpoint=None,
    # Version
    version=None,
):
    """
    Train the DSM Model.
    """
    
    ckpt_path = resume_checkpoint
    
    # TensorBoard logger
    logger = TensorBoardLogger(tb_logs, name=model_name, version=version)

    # Checkpoint on val loss improvement
    val_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val/loss:.4f}",
        monitor="val/loss",
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
        save_top_k=-1,
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
            monitor='val/loss',
            patience=early_stopping_patience,
            verbose=True,
            mode='min',
            min_delta=0.0,
        )
        callbacks_list.append(early_stop_callback)
        print(f"✓ Early stopping enabled (patience={early_stopping_patience} epochs)")
    
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
    else:
        accelerator = "auto"
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_devices > 1:
            # Use find_unused_parameters=True because UNet may have optional components
            # (e.g., param conditioning) that aren't used in every forward pass
            strategy = "ddp_find_unused_parameters_true"
            devices = num_devices
            print(f"✓ Using DDP strategy with {num_devices} GPUs (find_unused_parameters=True)")
        else:
            strategy = "auto"
            devices = 1
            print(f"✓ Using single device")

    trainer = Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        gradient_clip_val=1.0,  # Clip gradients for stability
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks_list,
        sync_batchnorm=True if num_devices > 1 else False,
        precision="32",
        use_distributed_sampler=True if num_devices > 1 else False,
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train DSM with custom UNet')
    parser.add_argument('--config', type=str, default='configs/dsm.ini',
                       help='Path to config file')
    parser.add_argument('--cpu_only', action='store_true',
                       help='Force training on CPU (for testing)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
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
        
        # Data parameters
        cropsize = int(params['cropsize'])
        batch_size = int(params['batch_size'])
        num_workers = int(params['num_workers'])
        dataset = params['dataset']
        data_root = params['data_root']
        boxsize = float(params.get('boxsize', fallback=6.25))
        
        # Architecture parameters (same as VDM)
        embedding_dim = int(params.get('embedding_dim', fallback=96))
        n_blocks = int(params.get('n_blocks', fallback=5))
        norm_groups = int(params.get('norm_groups', fallback=8))
        n_attention_heads = int(params.get('n_attention_heads', fallback=8))
        
        # Conditioning channels
        conditioning_channels = int(params.get('conditioning_channels', fallback=1))
        large_scale_channels = int(params.get('large_scale_channels', fallback=3))
        total_conditioning_channels = conditioning_channels + large_scale_channels
        
        # Fourier features
        use_fourier_features = params.getboolean('use_fourier_features', fallback=True)
        fourier_legacy = params.getboolean('fourier_legacy', fallback=False)
        
        # Attention
        add_attention = params.getboolean('add_attention', fallback=True)
        
        # Parameter conditioning
        use_param_conditioning = params.getboolean('use_param_conditioning', fallback=True)
        param_norm_path = params.get('param_norm_path', fallback=None)
        
        # Load parameter normalization
        if param_norm_path is None or param_norm_path.lower() in ['none', '']:
            use_param_conditioning = False
            param_min = None
            param_max = None
        else:
            import pandas as pd
            if not os.path.exists(param_norm_path):
                raise FileNotFoundError(f"Parameter norm path not found: {param_norm_path}")
            minmax_df = pd.read_csv(param_norm_path)
            param_min = list(minmax_df['MinVal'])
            param_max = list(minmax_df['MaxVal'])
            print(f"  Loaded {len(param_min)} parameter bounds from {param_norm_path}")
        
        # Noise schedule (VP-SDE)
        beta_min = float(params.get('beta_min', fallback=0.1))
        beta_max = float(params.get('beta_max', fallback=20.0))
        
        # Training parameters
        learning_rate = float(params['learning_rate'])
        weight_decay = float(params.get('weight_decay', fallback=1e-5))
        lr_scheduler = params.get('lr_scheduler', fallback='cosine')
        max_epochs = int(params.get('max_epochs', fallback=250))
        accumulate_grad_batches = int(params.get('accumulate_grad_batches', fallback=1))
        
        # Loss weighting
        use_snr_weighting = params.getboolean('use_snr_weighting', fallback=True)
        channel_weights_str = params.get('channel_weights', fallback='1.0,1.0,1.0')
        channel_weights = tuple(map(float, channel_weights_str.split(',')))
        
        # Sampling
        n_sampling_steps = int(params.get('n_sampling_steps', fallback=250))
        
        # Output channels (always 3: DM, Gas, Stars)
        output_channels = 3
        
        # Logging
        tb_logs = params['tb_logs']
        model_name = params['model_name']
        version = params.get('version', fallback=None)
        if version is not None and version.lower() not in ['none', '']:
            version = int(version)
        else:
            version = None
        
        # Callbacks
        enable_early_stopping = params.getboolean('enable_early_stopping', fallback=True)
        early_stopping_patience = int(params.get('early_stopping_patience', fallback=300))
        enable_gradient_monitoring = params.getboolean('enable_gradient_monitoring', fallback=True)
        gradient_log_frequency = int(params.get('gradient_log_frequency', fallback=100))
        
        # EMA
        enable_ema = params.getboolean('enable_ema', fallback=True)
        ema_decay = float(params.get('ema_decay', fallback=0.9999))
        ema_update_after_step = int(params.get('ema_update_after_step', fallback=0))
        ema_update_every = int(params.get('ema_update_every', fallback=1))
        
        # Fast ablation
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
        
        # Stellar normalization (for data loading)
        quantile_path = params.get('quantile_path', fallback=None)
        if quantile_path is not None and quantile_path.lower() in ['none', '']:
            quantile_path = None

    except KeyError as e:
        raise KeyError(f"Missing required parameter in config: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid parameter value in config: {e}")
    
    # Print config summary
    print(f"\n{'='*80}")
    print(f"DSM Model Training Configuration (Custom UNet)")
    print(f"{'='*80}")
    print(f"  Architecture: UNetVDM (same as VDM/Interpolant)")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  N blocks: {n_blocks}")
    print(f"  Fourier features: {use_fourier_features}")
    print(f"  Attention: {add_attention}")
    print(f"  Param conditioning: {use_param_conditioning}")
    print(f"  Conditioning channels: {total_conditioning_channels} (DM={conditioning_channels} + LS={large_scale_channels})")
    print(f"")
    print(f"  Noise schedule: VP-SDE")
    print(f"  Beta range: [{beta_min}, {beta_max}]")
    print(f"  SNR weighting: {use_snr_weighting}")
    print(f"  Channel weights: {channel_weights}")
    print(f"")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Accumulate grad batches: {accumulate_grad_batches}")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch = batch_size * accumulate_grad_batches * num_gpus
    print(f"  Effective batch size: {effective_batch} (batch={batch_size} × accum={accumulate_grad_batches} × gpus={num_gpus})")
    print(f"  Sampling steps: {n_sampling_steps}")
    print(f"{'='*80}\n")
    
    # Check data directory exists
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Load data
    dm = get_astro_data(
        dataset=dataset,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        quantile_path=quantile_path,
        limit_train_samples=limit_train_samples,
        limit_val_samples=limit_val_samples,
    )
    
    # Create UNet (same as VDM)
    print(f"\nCreating UNetVDM model...")
    
    unet = UNetVDM(
        input_channels=output_channels,  # 3 output channels
        conditioning_channels=conditioning_channels,  # DM condition
        large_scale_channels=large_scale_channels,  # Large-scale context
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
    
    # Create DSM model
    model = LightDSM(
        score_model=unet,
        beta_min=beta_min,
        beta_max=beta_max,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        n_sampling_steps=n_sampling_steps,
        use_param_conditioning=use_param_conditioning,
        use_snr_weighting=use_snr_weighting,
        channel_weights=channel_weights,
    )
    
    # Count parameters
    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_model_params:,}")
    
    # Train
    train(
        model=model, 
        datamodule=dm, 
        model_name=model_name,
        dataset=dataset,
        boxsize=boxsize,
        enable_early_stopping=enable_early_stopping,
        early_stopping_patience=early_stopping_patience,
        enable_gradient_monitoring=enable_gradient_monitoring,
        gradient_log_frequency=gradient_log_frequency,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        tb_logs=tb_logs,
        cpu_only=cli_args.cpu_only,
        enable_ema=enable_ema,
        ema_decay=ema_decay,
        ema_update_after_step=ema_update_after_step,
        ema_update_every=ema_update_every,
        accumulate_grad_batches=accumulate_grad_batches,
        resume_checkpoint=cli_args.resume,
        version=version,
    )
