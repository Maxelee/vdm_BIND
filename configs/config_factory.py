"""
Config Factory for VDM-BIND Generative Zoo.

This module provides a streamlined way to generate training configurations
by selecting method, backbone, and data source.

Usage:
    # Command line - generate config file
    python configs/config_factory.py --method vdm --backbone unet --output my_config.ini
    
    # Command line - print to stdout
    python configs/config_factory.py --method flow --backbone dit --print
    
    # Python API
    from configs.config_factory import ConfigFactory
    
    config = ConfigFactory.create(
        method="vdm",
        backbone="unet",
        data_root="/path/to/data",
    )
    config.save("my_config.ini")
    
    # Or get as dictionary
    config_dict = ConfigFactory.create_dict(method="flow", backbone="dit")

The factory uses sensible defaults for the IllustrisTNG dataset at Flatiron.
All paths can be overridden.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from configparser import ConfigParser
import argparse


# =============================================================================
# Default Paths (Flatiron-specific, can be overridden)
# =============================================================================

DEFAULT_PATHS = {
    "data_root": "/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/",
    "tb_logs": "/mnt/home/mlee1/ceph/tb_logs3",
    "param_norm_path": "/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv",
    "quantile_path": "/mnt/home/mlee1/vdm_BIND/data/quantile_normalizer_stellar.pkl",
}


# =============================================================================
# Method-specific defaults
# =============================================================================

METHOD_DEFAULTS = {
    "vdm": {
        "noise_schedule": "learned_nn",
        "gamma_min": -13.3,
        "gamma_max": 13.0,
        "antithetic_time_sampling": True,
        "data_noise": "5e-4, 5e-4, 5e-4",
        "lambdas": "1.0, 1.0, 1.0",
        "channel_weights": "1.0, 1.0, 3.0",
        "use_focal_loss": False,
        "focal_gamma": 3.0,
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine_warmup",
        "n_sampling_steps": 250,
        "early_stopping_metric": "val/elbo",
    },
    "flow": {
        "use_stochastic_interpolant": False,
        "sigma": 0.0,
        "x0_mode": "zeros",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler": "cosine_warmup",
        "n_sampling_steps": 50,
        "early_stopping_metric": "val/loss",
    },
    "consistency": {
        "sigma_min": 0.002,
        "sigma_max": 80.0,
        "sigma_data": 0.5,
        "rho": 7.0,
        "use_denoising_pretraining": True,
        "denoising_warmup_epochs": 10,
        "ct_n_steps": 18,
        "ema_decay": 0.9999,
        "learning_rate": 5e-5,
        "weight_decay": 1e-5,
        "lr_scheduler": "cosine_warmup",
        "n_sampling_steps": 1,
        "early_stopping_metric": "val/loss",
    },
    "dsm": {
        "beta_min": 0.1,
        "beta_max": 20.0,
        "use_snr_weighting": True,
        "channel_weights": "1.0, 1.0, 3.0",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler": "cosine_warmup",
        "n_sampling_steps": 250,
        "early_stopping_metric": "val/loss",
    },
    "ot_flow": {
        "ot_method": "exact",  # 'exact' or 'sinkhorn'
        "ot_reg": 0.01,  # Sinkhorn regularization
        "x0_mode": "zeros",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler": "cosine_warmup",
        "n_sampling_steps": 50,
        "early_stopping_metric": "val/loss",
    },
}


# =============================================================================
# Backbone-specific defaults
# =============================================================================

BACKBONE_DEFAULTS = {
    "unet": {
        "backbone_type": "unet",
        "embedding_dim": 96,
        "n_blocks": 5,
        "norm_groups": 8,
        "n_attention_heads": 8,
        "use_fourier_features": True,
        "fourier_legacy": False,
        "add_attention": True,
        "use_cross_attention": False,
        "batch_size": 128,
    },
    "unet-s": {
        "backbone_type": "unet-s",
        "embedding_dim": 64,
        "n_blocks": 4,
        "norm_groups": 8,
        "n_attention_heads": 8,
        "use_fourier_features": True,
        "fourier_legacy": False,
        "add_attention": True,
        "use_cross_attention": False,
        "batch_size": 128,
    },
    "unet-l": {
        "backbone_type": "unet-l",
        "embedding_dim": 128,
        "n_blocks": 6,
        "norm_groups": 8,
        "n_attention_heads": 8,
        "use_fourier_features": True,
        "fourier_legacy": False,
        "add_attention": True,
        "use_cross_attention": False,
        "batch_size": 64,
    },
    "dit": {
        "backbone_type": "dit",
        "dit_variant": "DiT-B/4",
        "patch_size": 4,
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "batch_size": 16,
        "accumulate_grad_batches": 2,
    },
    "dit-s": {
        "backbone_type": "dit-s",
        "dit_variant": "DiT-S/4",
        "patch_size": 4,
        "hidden_size": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "batch_size": 32,
        "accumulate_grad_batches": 1,
    },
    "dit-l": {
        "backbone_type": "dit-l",
        "dit_variant": "DiT-L/4",
        "patch_size": 4,
        "hidden_size": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "batch_size": 8,
        "accumulate_grad_batches": 4,
    },
    "dit-xl": {
        "backbone_type": "dit-xl",
        "dit_variant": "DiT-XL/4",
        "patch_size": 4,
        "hidden_size": 1152,
        "depth": 28,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "batch_size": 4,
        "accumulate_grad_batches": 8,
    },
    "fno": {
        "backbone_type": "fno",
        "fno_hidden_channels": 64,
        "fno_n_layers": 4,
        "fno_modes": 32,
        "use_film": True,
        "use_residual": True,
        "batch_size": 64,
        "accumulate_grad_batches": 2,
    },
    "fno-s": {
        "backbone_type": "fno-s",
        "fno_hidden_channels": 32,
        "fno_n_layers": 3,
        "fno_modes": 24,
        "use_film": True,
        "use_residual": True,
        "batch_size": 128,
        "accumulate_grad_batches": 1,
    },
    "fno-l": {
        "backbone_type": "fno-l",
        "fno_hidden_channels": 96,
        "fno_n_layers": 6,
        "fno_modes": 48,
        "use_film": True,
        "use_residual": True,
        "batch_size": 32,
        "accumulate_grad_batches": 4,
    },
}


# =============================================================================
# Base config (shared across all methods/backbones)
# =============================================================================

BASE_CONFIG = {
    # Dataset
    "seed": 8,
    "dataset": "IllustrisTNG",
    "field": "gas",
    "boxsize": 6.25,
    "cropsize": 128,
    "num_workers": 20,
    "max_epochs": 200,
    
    # Conditioning
    "conditioning_channels": 1,
    "large_scale_channels": 3,
    "use_param_conditioning": True,
    "use_quantile_normalization": True,
    
    # Monitoring
    "validation_plot_frequency": 500,
    "enable_early_stopping": False,
    "enable_fid_monitoring": False,
    "enable_gradient_monitoring": True,
    "gradient_log_frequency": 50,
    
    # EMA
    "enable_ema": True,
    "ema_decay": 0.9999,
    "ema_update_after_step": 1000,
    "ema_update_every": 1,
    
    # Training data limits
    "limit_train_samples": None,
    "limit_val_samples": None,
    "limit_train_batches": 1.0,
    
    # Speed optimizations
    "precision": 32,
    "compile_model": False,
    "gradient_clip_val": 1.0,
    
    # Accumulation default (can be overridden by backbone)
    "accumulate_grad_batches": 1,
}


# =============================================================================
# Config Factory
# =============================================================================

@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Core selections
    method: str = "vdm"
    backbone: str = "unet"
    
    # Paths
    data_root: str = DEFAULT_PATHS["data_root"]
    tb_logs: str = DEFAULT_PATHS["tb_logs"]
    param_norm_path: str = DEFAULT_PATHS["param_norm_path"]
    quantile_path: str = DEFAULT_PATHS["quantile_path"]
    
    # Model name (auto-generated if None)
    model_name: Optional[str] = None
    version: int = 0
    
    # All other parameters stored as dict
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate model name if not provided."""
        if self.model_name is None:
            self.model_name = f"{self.method}_{self.backbone}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for INI file."""
        result = {
            "method": self.method,
            "backbone_type": self.backbone,
            "data_root": self.data_root,
            "tb_logs": self.tb_logs,
            "param_norm_path": self.param_norm_path,
            "quantile_path": self.quantile_path,
            "model_name": self.model_name,
            "version": self.version,
        }
        result.update(self.params)
        return result
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to INI file."""
        filepath = Path(filepath)
        
        config = ConfigParser()
        config["TRAINING"] = {}
        
        # Add header comment
        all_params = self.to_dict()
        
        for key, value in all_params.items():
            if value is not None:
                config["TRAINING"][key] = str(value)
        
        with open(filepath, 'w') as f:
            # Write header
            f.write(f"# Auto-generated config: {self.method} + {self.backbone}\n")
            f.write(f"# Generated by configs/config_factory.py\n")
            f.write("#\n")
            f.write(f"# Method: {self.method}\n")
            f.write(f"# Backbone: {self.backbone}\n")
            f.write("#\n\n")
            config.write(f)
        
        print(f"✓ Config saved to: {filepath}")
    
    def to_ini_string(self) -> str:
        """Generate INI file content as string."""
        lines = [
            f"# Auto-generated config: {self.method} + {self.backbone}",
            "# Generated by configs/config_factory.py",
            "#",
            f"# Method: {self.method}",
            f"# Backbone: {self.backbone}",
            "#",
            "",
            "[TRAINING]",
        ]
        
        all_params = self.to_dict()
        
        # Group parameters by category
        categories = {
            "Core": ["method", "backbone_type", "seed", "dataset", "field", "boxsize"],
            "Paths": ["data_root", "tb_logs", "param_norm_path", "quantile_path"],
            "Training": ["batch_size", "num_workers", "cropsize", "max_epochs", 
                        "accumulate_grad_batches", "precision", "compile_model"],
            "Learning Rate": ["learning_rate", "weight_decay", "lr_scheduler", 
                             "gradient_clip_val"],
            "Architecture": ["embedding_dim", "n_blocks", "norm_groups", "n_attention_heads",
                            "use_fourier_features", "fourier_legacy", "add_attention",
                            "use_cross_attention", "dit_variant", "patch_size", "hidden_size",
                            "depth", "num_heads", "mlp_ratio", "fno_hidden_channels",
                            "fno_n_layers", "fno_modes", "use_film", "use_residual"],
            "Conditioning": ["conditioning_channels", "large_scale_channels",
                            "use_param_conditioning", "use_quantile_normalization"],
            "Method-Specific": ["noise_schedule", "gamma_min", "gamma_max", 
                               "antithetic_time_sampling", "data_noise", "lambdas",
                               "channel_weights", "use_focal_loss", "focal_gamma",
                               "use_stochastic_interpolant", "sigma", "x0_mode",
                               "sigma_min", "sigma_max", "sigma_data", "rho",
                               "use_denoising_pretraining", "denoising_warmup_epochs",
                               "ct_n_steps", "n_sampling_steps"],
            "EMA": ["enable_ema", "ema_decay", "ema_update_after_step", "ema_update_every"],
            "Monitoring": ["validation_plot_frequency", "enable_early_stopping",
                          "enable_fid_monitoring", "enable_gradient_monitoring",
                          "gradient_log_frequency"],
            "Logging": ["model_name", "version"],
        }
        
        written_keys = set()
        
        for category, keys in categories.items():
            category_params = [(k, all_params[k]) for k in keys 
                              if k in all_params and all_params[k] is not None]
            if category_params:
                lines.append(f"\n# {category}")
                for key, value in category_params:
                    lines.append(f"{key} = {value}")
                    written_keys.add(key)
        
        # Write any remaining parameters
        remaining = [(k, v) for k, v in all_params.items() 
                    if k not in written_keys and v is not None]
        if remaining:
            lines.append("\n# Other")
            for key, value in remaining:
                lines.append(f"{key} = {value}")
        
        return "\n".join(lines)


class ConfigFactory:
    """Factory for creating training configurations."""
    
    @classmethod
    def list_methods(cls) -> list:
        """List available methods."""
        return list(METHOD_DEFAULTS.keys())
    
    @classmethod
    def list_backbones(cls) -> list:
        """List available backbones."""
        return list(BACKBONE_DEFAULTS.keys())
    
    @classmethod
    def create(
        cls,
        method: str = "vdm",
        backbone: str = "unet",
        data_root: Optional[str] = None,
        tb_logs: Optional[str] = None,
        param_norm_path: Optional[str] = None,
        quantile_path: Optional[str] = None,
        model_name: Optional[str] = None,
        version: int = 0,
        **overrides
    ) -> TrainingConfig:
        """
        Create a training configuration.
        
        Args:
            method: Training method ('vdm', 'flow', 'consistency')
            backbone: Network backbone ('unet', 'dit', 'fno' + variants)
            data_root: Path to training data
            tb_logs: Path for TensorBoard logs
            param_norm_path: Path to parameter normalization file
            quantile_path: Path to quantile normalizer
            model_name: Name for the model (auto-generated if None)
            version: Version number for logging
            **overrides: Any parameter overrides
        
        Returns:
            TrainingConfig instance
        """
        # Validate inputs
        method = method.lower()
        backbone = backbone.lower()
        
        if method not in METHOD_DEFAULTS:
            raise ValueError(f"Unknown method: {method}. Available: {cls.list_methods()}")
        
        if backbone not in BACKBONE_DEFAULTS:
            raise ValueError(f"Unknown backbone: {backbone}. Available: {cls.list_backbones()}")
        
        # Build parameters: base -> method -> backbone -> overrides
        params = dict(BASE_CONFIG)
        params.update(METHOD_DEFAULTS[method])
        params.update(BACKBONE_DEFAULTS[backbone])
        params.update(overrides)
        
        return TrainingConfig(
            method=method,
            backbone=backbone,
            data_root=data_root or DEFAULT_PATHS["data_root"],
            tb_logs=tb_logs or DEFAULT_PATHS["tb_logs"],
            param_norm_path=param_norm_path or DEFAULT_PATHS["param_norm_path"],
            quantile_path=quantile_path or DEFAULT_PATHS["quantile_path"],
            model_name=model_name,
            version=version,
            params=params,
        )
    
    @classmethod
    def create_dict(
        cls,
        method: str = "vdm",
        backbone: str = "unet",
        **kwargs
    ) -> Dict[str, Any]:
        """Create configuration as dictionary."""
        config = cls.create(method=method, backbone=backbone, **kwargs)
        return config.to_dict()
    
    @classmethod
    def create_ini(
        cls,
        method: str = "vdm",
        backbone: str = "unet",
        **kwargs
    ) -> str:
        """Create configuration as INI string."""
        config = cls.create(method=method, backbone=backbone, **kwargs)
        return config.to_ini_string()
    
    @classmethod
    def print_options(cls):
        """Print available methods and backbones."""
        print("\n" + "=" * 60)
        print("CONFIG FACTORY OPTIONS")
        print("=" * 60)
        
        print("\nAvailable Methods:")
        for method in cls.list_methods():
            defaults = METHOD_DEFAULTS[method]
            metric = defaults.get("early_stopping_metric", "val/loss")
            steps = defaults.get("n_sampling_steps", "?")
            print(f"  {method:15s} - {steps:>4} sampling steps, monitor: {metric}")
        
        print("\nAvailable Backbones:")
        for backbone in cls.list_backbones():
            defaults = BACKBONE_DEFAULTS[backbone]
            batch = defaults.get("batch_size", "?")
            print(f"  {backbone:15s} - batch_size: {batch}")
        
        print("\n" + "=" * 60)


def interactive_create():
    """Interactive config creation."""
    print("\n" + "=" * 60)
    print("INTERACTIVE CONFIG GENERATOR")
    print("=" * 60)
    
    # Method selection
    print("\nAvailable methods:")
    methods = ConfigFactory.list_methods()
    for i, m in enumerate(methods, 1):
        print(f"  {i}. {m}")
    
    while True:
        try:
            choice = input("\nSelect method (1-3) [1]: ").strip() or "1"
            method = methods[int(choice) - 1]
            break
        except (ValueError, IndexError):
            print("Invalid choice, try again.")
    
    # Backbone selection
    print("\nAvailable backbones:")
    backbones = ["unet", "dit", "fno", "unet-s", "unet-l", "dit-s", "dit-l", "fno-s", "fno-l"]
    for i, b in enumerate(backbones, 1):
        print(f"  {i}. {b}")
    
    while True:
        try:
            choice = input("\nSelect backbone (1-9) [1]: ").strip() or "1"
            backbone = backbones[int(choice) - 1]
            break
        except (ValueError, IndexError):
            print("Invalid choice, try again.")
    
    # Model name
    default_name = f"{method}_{backbone}"
    model_name = input(f"\nModel name [{default_name}]: ").strip() or default_name
    
    # Output path
    default_output = f"configs/{model_name}.ini"
    output = input(f"\nOutput path [{default_output}]: ").strip() or default_output
    
    # Create and save
    config = ConfigFactory.create(
        method=method,
        backbone=backbone,
        model_name=model_name,
    )
    config.save(output)
    
    print(f"\n✓ Config created: {output}")
    print(f"  Method: {method}")
    print(f"  Backbone: {backbone}")
    print(f"\nTo train, run:")
    print(f"  python train_zoo.py --config {output}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate training configs for VDM-BIND Generative Zoo"
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=ConfigFactory.list_methods(),
        default="vdm",
        help="Training method"
    )
    parser.add_argument(
        "--backbone", "-b",
        type=str,
        choices=ConfigFactory.list_backbones(),
        default="unet",
        help="Network backbone"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: configs/{method}_{backbone}.ini)"
    )
    parser.add_argument(
        "--print", "-p",
        action="store_true",
        help="Print config to stdout instead of saving"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available methods and backbones"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom model name"
    )
    
    args = parser.parse_args()
    
    if args.list:
        ConfigFactory.print_options()
        return
    
    if args.interactive:
        interactive_create()
        return
    
    # Create config
    config = ConfigFactory.create(
        method=args.method,
        backbone=args.backbone,
        model_name=args.model_name,
    )
    
    if args.print:
        print(config.to_ini_string())
    else:
        output = args.output or f"configs/{config.model_name}.ini"
        config.save(output)


if __name__ == "__main__":
    main()
