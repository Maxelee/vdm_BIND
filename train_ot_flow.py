"""
Training script for Optimal Transport Flow Matching.

This script trains OT flow matching models using optimal transport coupling
for pairing source and target samples, following Lipman et al. (2022).

Key features:
- Optimal transport coupling for straighter interpolation paths
- Better sample quality for structured data (like astronomical fields)
- Compatible with existing VDM-BIND data pipeline

Usage:
    python train_ot_flow.py --config configs/ot_flow.ini
    python train_ot_flow.py --config configs/ot_flow.ini --cpu_only

Reference:
    Lipman et al. (2022) "Flow Matching for Generative Modeling"
    https://arxiv.org/abs/2210.02747
"""

import os
import numpy as np
import configparser
import argparse
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from vdm.astro_dataset import get_astro_data
from vdm.ot_flow_model import (
    LightOTFlow,
    OTInterpolant,
    OTVelocityNetWrapper,
)
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
    version=None,
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
    # Speed optimizations
    precision="32",
    compile_model=False,
):
    """
    Train the OT Flow Matching Model.
    """
    
    ckpt_path = None
    
    # TensorBoard logger - use explicit version to avoid nested directories
    logger = TensorBoardLogger(tb_logs, name=model_name, version=version)
    
    # Checkpoints
    val_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
    )
    
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
        every_n_train_steps=6000,
        save_top_k=10
    )
    
    epoch_checkpoint = ModelCheckpoint(
        filename="epoch-{epoch:03d}-{step}",
        save_top_k=-1,
        every_n_epochs=1,
    )
    
    callbacks_list = [
        LearningRateMonitor(),
        latest_checkpoint,
        val_checkpoint,
        epoch_checkpoint,
    ]
    
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
    
    if enable_gradient_monitoring:
        grad_callback = GradientMonitorCallback(
            log_every_n_steps=gradient_log_frequency,
            verbose=False
        )
        callbacks_list.append(grad_callback)
        print(f"✓ Gradient monitoring enabled (every {gradient_log_frequency} steps)")
    
    if enable_ema:
        ema_callback = EMACallback(
            decay=ema_decay,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every,
            use_ema_for_validation=True,
            save_ema_weights=True,
        )
        callbacks_list.append(ema_callback)
        print(f"✓ EMA enabled (decay={ema_decay})")
    
    # Configure trainer
    if cpu_only:
        print("\n⚠️  CPU-ONLY MODE: Training on CPU for testing")
        trainer = Trainer(
            accelerator='cpu',
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks_list,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_train_batches=limit_train_batches,
            enable_progress_bar=True,
        )
    else:
        trainer = Trainer(
            accelerator='auto',
            devices='auto',
            strategy='ddp_find_unused_parameters_true',
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks_list,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_train_batches=limit_train_batches,
            precision=precision,
        )
    
    # Optional: torch.compile for additional speedup
    if compile_model and not cpu_only:
        print("🔧 Compiling model with torch.compile...")
        model = torch.compile(model)
    
    print(f"\n🚀 SPEED OPTIMIZATIONS: precision={precision}, compile={compile_model}")
    
    print("\n" + "="*60)
    print("STARTING OT FLOW MATCHING TRAINING")
    print("="*60)
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Box size: {boxsize} Mpc/h")
    print(f"  Max epochs: {max_epochs}")
    print(f"  TensorBoard logs: {tb_logs}/{model_name}")
    print("="*60 + "\n")
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    return trainer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train OT Flow Matching model')
    parser.add_argument('--config', type=str, default='configs/ot_flow.ini',
                        help='Path to configuration file')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Train on CPU only (for testing)')
    args = parser.parse_args()
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(args.config)
    params = config['TRAINING']
    
    # Parse helpers
    def get_int(key, default):
        return int(params.get(key, default))
    
    def get_float(key, default):
        return float(params.get(key, default))
    
    def get_bool(key, default):
        val = params.get(key, str(default))
        return val.lower() in ('true', '1', 'yes')
    
    def get_str(key, default):
        return params.get(key, default)
    
    # Extract parameters
    seed = get_int('seed', 8)
    dataset = get_str('dataset', 'IllustrisTNG')
    data_root = get_str('data_root', '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/')
    boxsize = get_float('boxsize', 6.25)
    
    # Training hyperparameters
    batch_size = get_int('batch_size', 16)
    accumulate_grad_batches = get_int('accumulate_grad_batches', 1)
    num_workers = get_int('num_workers', 8)
    cropsize = get_int('cropsize', 128)
    max_epochs = get_int('max_epochs', 250)
    
    # Learning rate
    learning_rate = get_float('learning_rate', 1e-4)
    lr_scheduler = get_str('lr_scheduler', 'cosine')
    weight_decay = get_float('weight_decay', 1e-5)
    
    # Architecture
    embedding_dim = get_int('embedding_dim', 256)
    n_blocks = get_int('n_blocks', 32)
    norm_groups = get_int('norm_groups', 8)
    n_attention_heads = get_int('n_attention_heads', 8)
    conditioning_channels = get_int('conditioning_channels', 1)
    large_scale_channels = get_int('large_scale_channels', 3)
    use_fourier_features = get_bool('use_fourier_features', True)
    fourier_legacy = get_bool('fourier_legacy', False)
    add_attention = get_bool('add_attention', True)
    
    # OT-specific parameters
    n_sampling_steps = get_int('n_sampling_steps', 50)
    x0_mode = get_str('x0_mode', 'zeros')
    ot_method = get_str('ot_method', 'exact')
    ot_reg = get_float('ot_reg', 0.01)
    use_stochastic_interpolant = get_bool('use_stochastic_interpolant', False)
    sigma = get_float('sigma', 0.0)
    use_ot_training = get_bool('use_ot_training', True)
    ot_warmup_epochs = get_int('ot_warmup_epochs', 0)
    
    # Parameter conditioning
    param_norm_path = get_str('param_norm_path', '/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv')
    
    if param_norm_path is None or param_norm_path.lower() == 'none':
        use_param_conditioning = False
        min_vals = None
        max_vals = None
        Nparams = 0
    else:
        use_param_conditioning = True
        import pandas as pd
        if not os.path.exists(param_norm_path):
            raise FileNotFoundError(f"Parameter norm path not found: {param_norm_path}")
        minmax_df = pd.read_csv(param_norm_path)
        min_vals = np.array(minmax_df['MinVal'])
        max_vals = np.array(minmax_df['MaxVal'])
        Nparams = len(min_vals)
        print(f"  Loaded {Nparams} parameter bounds from {param_norm_path}")
    
    # EMA
    enable_ema = get_bool('enable_ema', True)
    ema_decay = get_float('ema_decay', 0.9999)
    ema_update_after_step = get_int('ema_update_after_step', 0)
    ema_update_every = get_int('ema_update_every', 1)
    
    # Early stopping
    enable_early_stopping = get_bool('enable_early_stopping', True)
    early_stopping_patience = get_int('early_stopping_patience', 300)
    
    # Gradient monitoring
    enable_gradient_monitoring = get_bool('enable_gradient_monitoring', True)
    gradient_log_frequency = get_int('gradient_log_frequency', 100)
    
    # Logging
    tb_logs = get_str('tb_logs', '/mnt/home/mlee1/ceph/tb_logs/')
    model_name = get_str('model_name', 'ot_flow_3ch')
    version = get_int('version', 0)
    
    # Speed optimizations
    precision = get_str('precision', '32')
    compile_model = get_bool('compile_model', False)
    
    # Set seed
    seed_everything(seed)
    
    print("\n" + "="*80)
    print("OT FLOW MATCHING TRAINING")
    print("="*80)
    print(f"\n📁 Configuration: {args.config}")
    print(f"🎲 Seed: {seed}")
    print(f"📊 Dataset: {dataset}")
    print(f"📂 Data root: {data_root}")
    
    total_conditioning_channels = conditioning_channels + large_scale_channels
    
    print(f"\n🔧 ARCHITECTURE:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  N blocks: {n_blocks}")
    print(f"  Attention heads: {n_attention_heads}")
    print(f"  Conditioning channels: {total_conditioning_channels}")
    print(f"  Fourier features: {use_fourier_features}")
    
    print(f"\n🔄 OT FLOW MATCHING:")
    print(f"  OT method: {ot_method}")
    if ot_method == 'sinkhorn':
        print(f"  OT regularization: {ot_reg}")
    print(f"  x0 mode: {x0_mode}")
    print(f"  Stochastic: {use_stochastic_interpolant} (sigma={sigma})")
    print(f"  Sampling steps: {n_sampling_steps}")
    print(f"  Use OT training: {use_ot_training}")
    if ot_warmup_epochs > 0:
        print(f"  OT warmup epochs: {ot_warmup_epochs}")
    
    print(f"\n📈 TRAINING:")
    print(f"  Batch size: {batch_size}")
    print(f"  Accumulate grad batches: {accumulate_grad_batches}")
    print(f"  Effective batch size: {batch_size * accumulate_grad_batches}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  EMA: {enable_ema}")
    
    # Import UNetVDM
    from vdm.networks_clean import UNetVDM
    
    # Create UNet
    output_channels = 3
    
    unet = UNetVDM(
        input_channels=output_channels,
        conditioning_channels=conditioning_channels,
        large_scale_channels=large_scale_channels,
        embedding_dim=embedding_dim,
        n_blocks=n_blocks,
        norm_groups=norm_groups,
        n_attention_heads=n_attention_heads,
        use_fourier_features=use_fourier_features,
        legacy_fourier=fourier_legacy,
        add_attention=add_attention,
        use_param_conditioning=use_param_conditioning,
        param_min=min_vals,
        param_max=max_vals,
    )
    
    # Wrap UNet
    velocity_model = OTVelocityNetWrapper(
        net=unet,
        output_channels=output_channels,
        conditioning_channels=total_conditioning_channels,
    )
    
    # Create model
    model = LightOTFlow(
        velocity_model=velocity_model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        n_sampling_steps=n_sampling_steps,
        ot_method=ot_method,
        ot_reg=ot_reg,
        use_stochastic_interpolant=use_stochastic_interpolant,
        sigma=sigma,
        x0_mode=x0_mode,
        use_param_conditioning=use_param_conditioning,
        use_ot_training=use_ot_training,
        ot_warmup_epochs=ot_warmup_epochs,
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 MODEL PARAMETERS:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Load data
    print(f"\n📦 Loading data from {data_root}...")
    datamodule = get_astro_data(
        dataset=dataset,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Train
    train(
        model=model,
        datamodule=datamodule,
        model_name=model_name,
        version=version,
        dataset=dataset,
        boxsize=boxsize,
        enable_early_stopping=enable_early_stopping,
        early_stopping_patience=early_stopping_patience,
        enable_gradient_monitoring=enable_gradient_monitoring,
        gradient_log_frequency=gradient_log_frequency,
        max_epochs=max_epochs,
        tb_logs=tb_logs,
        cpu_only=args.cpu_only,
        enable_ema=enable_ema,
        ema_decay=ema_decay,
        ema_update_after_step=ema_update_after_step,
        ema_update_every=ema_update_every,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        compile_model=compile_model,
    )


if __name__ == "__main__":
    main()
