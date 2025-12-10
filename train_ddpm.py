"""
Training script for DDPM/Score Model.

This script trains score-based diffusion models using the score_models package
while maintaining compatibility with the existing VDM-BIND data pipeline.

Usage:
    python train_ddpm.py --config configs/ddpm.ini
    python train_ddpm.py --config configs/ddpm.ini --cpu_only  # For testing

Key differences from VDM training:
- Uses Denoising Score Matching (DSM) loss instead of VDM ELBO
- Supports VP-SDE (like DDPM) or VE-SDE (like NCSN) noise schedules
- Uses score_models architectures (DDPM, NCSNpp) instead of custom UNet
"""

import os
import numpy as np
import configparser
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from vdm.astro_dataset import get_astro_data
from vdm.ddpm_model import LightScoreModel, create_ncsnpp_model, SCORE_MODELS_AVAILABLE
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
):
    """
    Train the Score Model.
    
    Args:
        model: LightScoreModel model
        datamodule: PyTorch Lightning DataModule
        model_name: Name for logging and checkpoints
        dataset: Dataset name (for reference)
        boxsize: Physical size of box (for reference)
        enable_early_stopping: Enable early stopping callback
        early_stopping_patience: Epochs to wait before stopping
        enable_gradient_monitoring: Monitor gradient statistics
        gradient_log_frequency: Steps between gradient logging
        max_epochs: Maximum training epochs
        limit_train_batches: Fraction of training data per epoch
        tb_logs: TensorBoard logs directory
        cpu_only: Force CPU training
        enable_ema: Enable EMA (recommended for diffusion)
        ema_decay: EMA decay factor
        ema_update_after_step: Start EMA after this many steps
        ema_update_every: Update EMA every N steps
    """
    
    ckpt_path = None
    
    # TensorBoard logger
    logger = TensorBoardLogger(tb_logs, name=model_name)

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
    
    # Build callbacks list
    callbacks_list = [
        LearningRateMonitor(),
        latest_checkpoint,
        val_checkpoint,
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
            strategy = "ddp"
            devices = num_devices
            print(f"✓ Using DDP strategy with {num_devices} GPUs")
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
        gradient_clip_val=1.0,
        callbacks=callbacks_list,
        sync_batchnorm=True if num_devices > 1 else False,
        precision="32",
        use_distributed_sampler=True if num_devices > 1 else False,
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import argparse
    
    # Check score_models is available
    if not SCORE_MODELS_AVAILABLE:
        raise ImportError(
            "score_models package not found!\n"
            "Install with: pip install score_models\n"
            "Or: pip install git+https://github.com/AlexandreAdam/score_models.git"
        )
    
    # Import score_models architectures
    from score_models import NCSNpp, DDPM
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train DDPM/Score Model')
    parser.add_argument('--config', type=str, default='configs/ddpm.ini',
                       help='Path to config file')
    parser.add_argument('--cpu_only', action='store_true',
                       help='Force training on CPU (for testing)')
    
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
        
        # Architecture parameters
        architecture = params.get('architecture', fallback='ncsnpp').lower()
        nf = int(params.get('nf', fallback=128))
        ch_mult_str = params.get('ch_mult', fallback='1,2,2,4')
        ch_mult = tuple(map(int, ch_mult_str.split(',')))
        num_res_blocks = int(params.get('num_res_blocks', fallback=2))
        attention = params.getboolean('attention', fallback=True)
        dropout = float(params.get('dropout', fallback=0.1))
        
        # SDE parameters
        sde_type = params.get('sde', fallback='vp').lower()
        beta_min = float(params.get('beta_min', fallback=0.1))
        beta_max = float(params.get('beta_max', fallback=20.0))
        sigma_min = float(params.get('sigma_min', fallback=0.01))
        sigma_max = float(params.get('sigma_max', fallback=50.0))
        
        # Training parameters
        learning_rate = float(params['learning_rate'])
        lr_scheduler = params.get('lr_scheduler', fallback='cosine')
        max_epochs = int(params.get('max_epochs', fallback=100))
        
        # Large-scale conditioning
        large_scale_channels = int(params.get('large_scale_channels', fallback=3))
        conditioning_channels = 1 + large_scale_channels  # DM + large-scale
        
        # Parameter conditioning (astro params)
        use_param_conditioning = params.getboolean('use_param_conditioning', fallback=False)
        n_params = int(params.get('n_params', fallback=35))
        
        # Output channels (always 3: DM, Gas, Stars)
        output_channels = 3
        
        # Logging
        tb_logs = params['tb_logs']
        model_name = params['model_name']
        
        # Callbacks
        enable_early_stopping = params.getboolean('enable_early_stopping', fallback=True)
        early_stopping_patience = int(params.get('early_stopping_patience', fallback=30))
        enable_gradient_monitoring = params.getboolean('enable_gradient_monitoring', fallback=True)
        gradient_log_frequency = int(params.get('gradient_log_frequency', fallback=100))
        
        # EMA
        enable_ema = params.getboolean('enable_ema', fallback=True)
        ema_decay = float(params.get('ema_decay', fallback=0.9999))
        ema_update_after_step = int(params.get('ema_update_after_step', fallback=1000))
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
        
        # Stellar normalization
        quantile_path = params.get('quantile_path', fallback=None)
        if quantile_path is not None and quantile_path.lower() in ['none', '']:
            quantile_path = None

    except KeyError as e:
        raise KeyError(f"Missing required parameter in config: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid parameter value in config: {e}")
    
    # Print config summary
    print(f"\n{'='*80}")
    print(f"DDPM/Score Model Training Configuration")
    print(f"{'='*80}")
    print(f"  Architecture: {architecture.upper()}")
    print(f"  SDE type: {sde_type.upper()}")
    if sde_type == 'vp':
        print(f"  Beta range: [{beta_min}, {beta_max}]")
    else:
        print(f"  Sigma range: [{sigma_min}, {sigma_max}]")
    print(f"  Base features: {nf}")
    print(f"  Channel mult: {ch_mult}")
    print(f"  Conditioning channels: {conditioning_channels}")
    print(f"  Param conditioning: {use_param_conditioning} (n_params={n_params})")
    print(f"  Output channels: {output_channels}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
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
    
    # Create model
    print(f"\nCreating {architecture.upper()} model...")
    
    if architecture == 'ncsnpp':
        # NCSNpp with native input conditioning (and optional vector conditioning)
        condition_types = ["input"]
        condition_kwargs = {
            "condition_input_channels": conditioning_channels,
        }
        
        # Add vector conditioning for astro params if enabled
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
            beta_min=beta_min,
            beta_max=beta_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            ema_decay=ema_decay,
            use_param_conditioning=use_param_conditioning,
        )
        
    elif architecture == 'ddpm':
        # DDPM with manual input conditioning
        from vdm.ddpm_model import ConditionedDDPMWrapper
        
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
            beta_min=beta_min,
            beta_max=beta_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            ema_decay=ema_decay,
            use_param_conditioning=False,  # DDPM wrapper doesn't support vector conditioning yet
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'ncsnpp' or 'ddpm'.")
    
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
    )
