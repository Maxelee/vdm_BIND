"""
Generative Zoo Training Script.

This script provides a clean interface for training generative models using the
Method + Backbone + Data abstraction pattern.

Usage:
    # Train VDM with UNet backbone
    python train_zoo.py --method vdm --backbone unet-b --config configs/clean_vdm_aggressive_stellar.ini
    
    # Train Flow Matching with DiT backbone
    python train_zoo.py --method flow --backbone dit-s --config configs/interpolant.ini
    
    # Train Consistency with FNO backbone
    python train_zoo.py --method consistency --backbone fno-b --config configs/consistency.ini

Architecture:
    Method (vdm, flow, consistency)  <- Training/sampling paradigm
        +
    Backbone (unet, dit, fno)        <- Network architecture  
        +
    Dataset (astro_dataset)          <- Data source
        =
    Experiment

The key insight is that methods and backbones are orthogonal choices:
- VDM can use UNet, DiT, or FNO
- Flow Matching can use UNet, DiT, or FNO
- Consistency can use UNet, DiT, or FNO

This enables systematic comparison of architectures and training paradigms.
"""

import os
import argparse
import configparser
import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from vdm.astro_dataset import get_astro_data
from vdm.methods import MethodRegistry, list_methods, create_method
from vdm.backbones import list_backbones
from vdm.callbacks import CustomEarlyStopping, EMACallback, GradientMonitorCallback

torch.set_float32_matmul_precision("medium")


# =============================================================================
# Available Methods and Backbones
# =============================================================================

METHODS = list_methods()
BACKBONES = list_backbones()

DEFAULT_CONFIGS = {
    'vdm': 'configs/clean_vdm_aggressive_stellar.ini',
    'flow': 'configs/interpolant.ini',
    'consistency': 'configs/consistency.ini',
}


# =============================================================================
# Config Parsing
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
    
    def get_optional(self, key, default=None):
        val = self.params.get(key, default)
        if val is None or (isinstance(val, str) and val.lower() in ['none', '']):
            return None
        return val
    
    def get_optional_int(self, key, default=None):
        val = self.get_optional(key, default)
        return int(val) if val is not None else None


# =============================================================================
# Parameter Normalization Loading
# =============================================================================

def load_param_normalization(param_norm_path, param_min_inline=None, param_max_inline=None, n_params_inline=None):
    """Load parameter normalization bounds from various sources."""
    
    # Check for inline specification first
    if param_min_inline is not None and param_max_inline is not None:
        min_vals = np.array([float(x.strip()) for x in param_min_inline.split(',')])
        max_vals = np.array([float(x.strip()) for x in param_max_inline.split(',')])
        n_params = len(min_vals)
        print(f"✓ Parameter conditioning: {n_params} parameters (inline specification)")
        return True, min_vals, max_vals, n_params
    
    # Check for explicit n_params=0 (unconditional)
    if n_params_inline is not None and int(n_params_inline) == 0:
        print("✓ Parameter conditioning: UNCONDITIONAL (n_params=0)")
        return False, None, None, 0
    
    # Check for 'none' or empty path (unconditional)
    if param_norm_path is None or (isinstance(param_norm_path, str) and param_norm_path.lower() in ['none', '']):
        print("✓ Parameter conditioning: UNCONDITIONAL (no param_norm_path)")
        return False, None, None, 0
    
    if not os.path.exists(param_norm_path):
        raise FileNotFoundError(f"Parameter norm path not found: {param_norm_path}")
    
    # Detect file format (CSV or JSON)
    if param_norm_path.endswith('.json'):
        import json
        with open(param_norm_path, 'r') as f:
            param_config = json.load(f)
        min_vals = np.array(param_config['param_min'])
        max_vals = np.array(param_config['param_max'])
    else:
        import pandas as pd
        minmax_df = pd.read_csv(param_norm_path)
        if 'MinVal' in minmax_df.columns:
            min_vals = np.array(minmax_df['MinVal'])
            max_vals = np.array(minmax_df['MaxVal'])
        else:
            min_vals = np.array(minmax_df['min'])
            max_vals = np.array(minmax_df['max'])
    
    n_params = len(min_vals)
    print(f"✓ Parameter conditioning: {n_params} parameters from {param_norm_path}")
    return True, min_vals, max_vals, n_params


# =============================================================================
# Training Function
# =============================================================================

def train(
    method,
    datamodule,
    model_name,
    method_type,
    backbone_type,
    early_stopping_metric='val/loss',
    max_epochs=100,
    tb_logs='tb_logs',
    cpu_only=False,
    enable_ema=False,
    ema_decay=0.9999,
    gradient_clip_val=1.0,
    resume_checkpoint=None,
):
    """Train a method using PyTorch Lightning."""
    
    # Logger
    logger = TensorBoardLogger(tb_logs, name=model_name)
    
    # Checkpoints
    val_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{" + early_stopping_metric.replace('/', '_') + ":.4f}",
        monitor=early_stopping_metric,
        mode="min",
        save_top_k=3,
        save_weights_only=True,
    )
    
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
        every_n_train_steps=6000,
        save_top_k=3,
        save_weights_only=True,
    )
    
    epoch_checkpoint = ModelCheckpoint(
        filename="epoch-{epoch:03d}-{step}",
        save_top_k=-1,
        every_n_epochs=5,
        save_weights_only=True,
    )
    
    callbacks = [
        LearningRateMonitor(),
        latest_checkpoint,
        val_checkpoint,
        epoch_checkpoint,
        CustomEarlyStopping(
            monitor=early_stopping_metric,
            patience=300,
            verbose=True,
            mode='min',
        ),
        GradientMonitorCallback(log_every_n_steps=100),
    ]
    
    if enable_ema:
        callbacks.append(EMACallback(decay=ema_decay))
        print(f"✓ EMA enabled (decay={ema_decay})")
    
    # Device config
    if cpu_only:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
        print("⚠️  CPU-ONLY MODE (for testing)")
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
    
    trainer = Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        precision="32",
    )
    
    # Print training banner
    print("\n" + "="*60)
    print(f"GENERATIVE ZOO TRAINING")
    print("="*60)
    print(f"  Method: {method_type}")
    print(f"  Backbone: {backbone_type}")
    print(f"  Monitor: {early_stopping_metric}")
    print(f"  Max epochs: {max_epochs}")
    print("="*60 + "\n")
    
    trainer.fit(model=method, datamodule=datamodule, ckpt_path=resume_checkpoint)
    
    return trainer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generative Zoo Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Methods: {', '.join(METHODS)}
Backbones: {', '.join(BACKBONES)}

Examples:
  python train_zoo.py --method vdm --backbone unet-b
  python train_zoo.py --method flow --backbone dit-s
  python train_zoo.py --method consistency --backbone fno-b --n_sampling_steps 1
"""
    )
    
    parser.add_argument('--method', type=str, required=True, choices=METHODS,
                        help='Training method (vdm, flow, consistency)')
    parser.add_argument('--backbone', type=str, required=True, choices=BACKBONES,
                        help='Backbone architecture (unet, dit, fno, or variants)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (uses default if not specified)')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU training (for testing)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Override arguments
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--n_sampling_steps', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    
    args = parser.parse_args()
    
    # Use default config if not specified
    config_path = args.config or DEFAULT_CONFIGS.get(args.method, 'configs/clean_vdm_aggressive_stellar.ini')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
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
    print(f"GENERATIVE ZOO: {args.method.upper()} + {args.backbone.upper()}")
    print("="*80)
    print(f"📁 Configuration: {config_path}")
    print(f"🎲 Seed: {seed}")
    
    # Data parameters
    data_root = cfg.get_str('data_root', '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/')
    batch_size = args.batch_size or cfg.get_int('batch_size', 16)
    num_workers = cfg.get_int('num_workers', 8)
    img_size = cfg.get_int('cropsize', 128)
    
    print(f"📂 Data root: {data_root}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🖼️ Image size: {img_size}")
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Load parameter normalization
    param_norm_path = cfg.get_optional('param_norm_path')
    param_min_inline = cfg.get_optional('param_min')
    param_max_inline = cfg.get_optional('param_max')
    n_params_inline = cfg.get_optional('n_params')
    
    use_param_conditioning, min_vals, max_vals, n_params = load_param_normalization(
        param_norm_path, param_min_inline, param_max_inline, n_params_inline
    )
    
    # Load data
    print(f"\n📦 Loading data...")
    datamodule = get_astro_data(
        dataset=cfg.get_str('dataset', 'IllustrisTNG'),
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Create method with backbone
    print(f"\n🔧 Creating {args.method.upper()} method with {args.backbone.upper()} backbone...")
    
    # Build method-specific parameters
    method_kwargs = {
        'backbone_type': args.backbone,
        'img_size': img_size,
        'image_shape': (3, img_size, img_size),
        'conditioning_channels': cfg.get_int('conditioning_channels', 1),
        'large_scale_channels': cfg.get_int('large_scale_channels', 3),
        'param_dim': n_params if use_param_conditioning else 0,
        'use_param_conditioning': use_param_conditioning,
        'learning_rate': args.learning_rate or cfg.get_float('learning_rate', 1e-4),
        'lr_scheduler': cfg.get_str('lr_scheduler', 'cosine'),
        'n_sampling_steps': args.n_sampling_steps or cfg.get_int('n_sampling_steps', 100),
    }
    
    if min_vals is not None:
        method_kwargs['param_min'] = list(min_vals)
        method_kwargs['param_max'] = list(max_vals)
    
    # Method-specific parameters
    if args.method == 'vdm':
        method_kwargs.update({
            'gamma_min': cfg.get_float('gamma_min', -13.3),
            'gamma_max': cfg.get_float('gamma_max', 5.0),
            'noise_schedule': cfg.get_str('noise_schedule', 'fixed_linear'),
            'channel_weights': cfg.get_list_float('channel_weights', '1.0,1.0,1.0'),
            'use_focal_loss': cfg.get_bool('use_focal_loss', False),
        })
    elif args.method == 'flow':
        method_kwargs.update({
            'use_stochastic_interpolant': cfg.get_bool('use_stochastic_interpolant', False),
            'sigma': cfg.get_float('sigma', 0.0),
            'x0_mode': cfg.get_str('x0_mode', 'zeros'),
        })
    elif args.method == 'consistency':
        method_kwargs.update({
            'sigma_min': cfg.get_float('sigma_min', 0.002),
            'sigma_max': cfg.get_float('sigma_max', 80.0),
            'use_denoising_pretraining': cfg.get_bool('use_denoising_pretraining', True),
            'ct_n_steps': cfg.get_int('ct_n_steps', 18),
        })
    
    method = create_method(args.method, **method_kwargs)
    
    # Print model summary
    total_params = sum(p.numel() for p in method.parameters())
    trainable_params = sum(p.numel() for p in method.parameters() if p.requires_grad)
    print(f"\n📊 MODEL PARAMETERS:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Get early stopping metric
    early_stopping_metric = MethodRegistry.get_early_stopping_metric(args.method)
    
    # Train
    train(
        method=method,
        datamodule=datamodule,
        model_name=f"{args.method}_{args.backbone}",
        method_type=args.method,
        backbone_type=args.backbone,
        early_stopping_metric=early_stopping_metric,
        max_epochs=args.max_epochs or cfg.get_int('max_epochs', 100),
        tb_logs=cfg.get_str('tb_logs', '/mnt/home/mlee1/ceph/tb_logs/'),
        cpu_only=args.cpu_only,
        enable_ema=cfg.get_bool('enable_ema', False),
        ema_decay=cfg.get_float('ema_decay', 0.9999),
        gradient_clip_val=cfg.get_float('gradient_clip_val', 1.0),
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
