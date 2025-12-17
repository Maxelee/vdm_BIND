"""
Backbone Abstraction Layer for VDM-BIND Generative Zoo.

This module provides a unified interface for different neural network backbones
(UNet, DiT, FNO) that can be used with various generative methods (VDM, DDPM,
Flow Matching, Consistency Models, etc.).

Standard Interface:
    All backbones follow the signature:
    forward(x_t, t, conditioning=None, param_conditioning=None) -> prediction
    
    Args:
        x_t: (B, C, H, W) - Noisy input at timestep t
        t: (B,) - Timestep values in [0, 1]
        conditioning: (B, C_cond, H, W) - Spatial conditioning (e.g., DM field)
        param_conditioning: (B, N_params) - Cosmological/physical parameters
    
    Returns:
        prediction: (B, C, H, W) - Predicted noise, score, or velocity

This enables the "Generative Zoo" design:
    Method + Backbone + Dataset = Experiment
    
Example:
    # Create backbone from registry
    backbone = BackboneRegistry.create("unet", config)
    
    # Use with any method
    method = VDMMethod(backbone, ...)
    method = FlowMatchingMethod(backbone, ...)
    method = ConsistencyMethod(backbone, ...)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Type, Callable, Tuple
import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# Base Classes
# =============================================================================

class BackboneBase(ABC, nn.Module):
    """
    Abstract base class for all neural network backbones.
    
    Backbones are architecture-specific neural networks that take noisy data
    and return predictions (noise, score, or velocity). They handle:
    - Time embedding (how timestep t is encoded)
    - Spatial conditioning (how DM/large-scale maps are incorporated)
    - Parameter conditioning (how cosmological params are embedded)
    
    The standard interface allows any backbone to be used with any method.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        param_dim: int = 0,
        img_size: int = 128,
        **kwargs
    ):
        """
        Initialize backbone with standard parameters.
        
        Args:
            input_channels: Number of channels in target (e.g., 3 for [DM, Gas, Stars])
            output_channels: Number of output channels (usually same as input)
            conditioning_channels: Channels for DM conditioning (typically 1)
            large_scale_channels: Channels for large-scale context (typically 3)
            param_dim: Dimension of parameter conditioning (0 = no params)
            img_size: Input/output spatial resolution
            **kwargs: Backbone-specific arguments
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conditioning_channels = conditioning_channels
        self.large_scale_channels = large_scale_channels
        self.param_dim = param_dim
        self.img_size = img_size
    
    @abstractmethod
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with standard interface.
        
        Args:
            x_t: (B, C, H, W) - Noisy input at timestep t
            t: (B,) - Timestep values in [0, 1], where:
                - t=0: Clean data (high SNR)
                - t=1: Pure noise (low SNR)
            conditioning: (B, C_cond, H, W) - Spatial conditioning
            param_conditioning: (B, N_params) - Parameter conditioning
        
        Returns:
            (B, output_channels, H, W) - Network prediction
        """
        pass
    
    @property
    def total_conditioning_channels(self) -> int:
        """Total channels for spatial conditioning."""
        return self.conditioning_channels + self.large_scale_channels
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        return {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "conditioning_channels": self.conditioning_channels,
            "large_scale_channels": self.large_scale_channels,
            "param_dim": self.param_dim,
            "img_size": self.img_size,
        }


# =============================================================================
# Backbone Registry
# =============================================================================

class BackboneRegistry:
    """
    Registry for backbone architectures.
    
    Allows registration and instantiation of backbones by name.
    
    Usage:
        # Register a backbone
        @BackboneRegistry.register("my_unet")
        class MyUNet(BackboneBase):
            ...
        
        # Or manually
        BackboneRegistry.register_backbone("my_unet", MyUNet)
        
        # Create instance
        backbone = BackboneRegistry.create("my_unet", config)
        
        # List available
        print(BackboneRegistry.available())
    """
    
    _registry: Dict[str, Type[BackboneBase]] = {}
    _configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a backbone class."""
        def decorator(backbone_cls: Type[BackboneBase]) -> Type[BackboneBase]:
            cls._registry[name.lower()] = backbone_cls
            return backbone_cls
        return decorator
    
    @classmethod
    def register_backbone(cls, name: str, backbone_cls: Type[BackboneBase]) -> None:
        """Manually register a backbone class."""
        cls._registry[name.lower()] = backbone_cls
    
    @classmethod
    def register_config(cls, name: str, config: Dict[str, Any]) -> None:
        """Register a preset configuration for a backbone variant."""
        cls._configs[name.lower()] = config
    
    @classmethod
    def create(
        cls,
        name: str,
        **kwargs
    ) -> BackboneBase:
        """
        Create a backbone instance by name.
        
        Args:
            name: Backbone name (e.g., "unet", "dit-b", "fno-l")
            **kwargs: Arguments passed to backbone constructor
        
        Returns:
            Instantiated backbone
        
        Raises:
            KeyError: If backbone name not found
        """
        name_lower = name.lower()
        
        # Check for preset config
        if name_lower in cls._configs:
            preset = cls._configs[name_lower]
            # Preset config is overridden by explicit kwargs
            merged = {**preset, **kwargs}
            backbone_name = preset.get("_backbone_class", name_lower.split("-")[0])
        else:
            merged = kwargs
            backbone_name = name_lower.split("-")[0]  # "dit-b" -> "dit"
        
        if backbone_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Unknown backbone: {name}. Available: {available}"
            )
        
        backbone_cls = cls._registry[backbone_name]
        
        # Remove internal keys
        merged.pop("_backbone_class", None)
        
        return backbone_cls(**merged)
    
    @classmethod
    def available(cls) -> list:
        """List available backbone names."""
        # Include both base names and preset configs
        base_names = list(cls._registry.keys())
        preset_names = list(cls._configs.keys())
        return sorted(set(base_names + preset_names))
    
    @classmethod
    def get_class(cls, name: str) -> Type[BackboneBase]:
        """Get the backbone class (not instance) by name."""
        backbone_name = name.lower().split("-")[0]
        if backbone_name not in cls._registry:
            raise KeyError(f"Unknown backbone: {name}")
        return cls._registry[backbone_name]


# =============================================================================
# UNet Backbone Wrapper
# =============================================================================

@BackboneRegistry.register("unet")
class UNetBackbone(BackboneBase):
    """
    UNet backbone wrapper for VDM-style networks.
    
    This wraps the existing UNetVDM architecture from networks_clean.py,
    converting from the standard t ∈ [0,1] interface to gamma (log-SNR)
    used internally by UNetVDM.
    
    Time mapping:
        t=0 (clean) -> gamma_max (high SNR)
        t=1 (noise) -> gamma_min (low SNR)
        gamma = gamma_max - (gamma_max - gamma_min) * t
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        param_dim: int = 0,
        img_size: int = 128,
        embedding_dim: int = 128,
        n_blocks: int = 4,
        norm_groups: int = 8,
        n_attention_heads: int = 4,
        dropout_prob: float = 0.0,
        attention_everywhere: bool = True,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        use_fourier_features: bool = True,
        legacy_fourier: bool = True,
        use_cross_attention: bool = False,
        param_min: list = None,
        param_max: list = None,
        **kwargs
    ):
        """
        Initialize UNet backbone.
        
        Args:
            input_channels: Target channels (e.g., 3 for DM/Gas/Stars)
            output_channels: Output channels (usually same as input)
            conditioning_channels: DM conditioning channels (typically 1)
            large_scale_channels: Large-scale context channels (typically 3)
            param_dim: Parameter conditioning dimension (0 = no params)
            img_size: Spatial resolution
            embedding_dim: Base embedding dimension
            n_blocks: Number of up/down blocks
            norm_groups: Groups for GroupNorm
            n_attention_heads: Attention heads
            dropout_prob: Dropout probability
            attention_everywhere: Use attention at all resolutions
            gamma_min: Minimum gamma (noise end)
            gamma_max: Maximum gamma (clean end)
            use_fourier_features: Enable Fourier features
            legacy_fourier: Use legacy Fourier mode
            use_cross_attention: Enable cross-attention conditioning
            param_min: Min values for parameter normalization
            param_max: Max values for parameter normalization
            **kwargs: Additional UNetVDM arguments
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            conditioning_channels=conditioning_channels,
            large_scale_channels=large_scale_channels,
            param_dim=param_dim,
            img_size=img_size,
        )
        
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
        # Import here to avoid circular imports
        from vdm.networks_clean import UNetVDM
        
        # Determine if we need param conditioning
        use_param_conditioning = param_dim > 0
        
        # Auto-generate param_min/param_max if not provided but param_dim > 0
        if use_param_conditioning and (param_min is None or param_max is None):
            param_min = [0.0] * param_dim
            param_max = [1.0] * param_dim
        
        # Build UNetVDM with appropriate settings
        self.net = UNetVDM(
            input_channels=input_channels,
            conditioning_channels=conditioning_channels,
            large_scale_channels=large_scale_channels,
            embedding_dim=embedding_dim,
            n_blocks=n_blocks,
            norm_groups=norm_groups,
            n_attention_heads=n_attention_heads,
            dropout_prob=dropout_prob,
            attention_everywhere=attention_everywhere,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            use_fourier_features=use_fourier_features,
            legacy_fourier=legacy_fourier,
            use_cross_attention=use_cross_attention,
            use_param_conditioning=use_param_conditioning,
            param_min=param_min,
            param_max=param_max,
            **kwargs
        )
    
    def t_to_gamma(self, t: Tensor) -> Tensor:
        """
        Convert timestep t to gamma (log-SNR).
        
        Args:
            t: (B,) timestep in [0, 1]
               t=0: clean (high SNR)
               t=1: noisy (low SNR)
        
        Returns:
            gamma: (B,) log-SNR values
                   t=0 -> gamma_max
                   t=1 -> gamma_min
        """
        return self.gamma_max - (self.gamma_max - self.gamma_min) * t
    
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with standard interface.
        
        Converts t to gamma internally for UNetVDM.
        """
        # Convert t to gamma
        gamma = self.t_to_gamma(t)
        
        # Call UNetVDM
        # UNetVDM may return tuple (prediction, params) or (prediction, params, mask)
        # We only need the prediction
        output = self.net(x_t, gamma, conditioning, param_conditioning)
        
        if isinstance(output, tuple):
            return output[0]  # Just the prediction
        return output
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        config = super().get_config()
        config.update({
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "backbone_type": "unet",
        })
        return config


# =============================================================================
# DiT Backbone Wrapper
# =============================================================================

@BackboneRegistry.register("dit")
class DiTBackbone(BackboneBase):
    """
    DiT (Diffusion Transformer) backbone wrapper.
    
    Wraps the DiT architecture from dit.py. DiT natively uses timestep t,
    so no conversion is needed.
    
    Note: DiT concatenates x_t with conditioning internally, so we pass
    in_channels as the target channels only.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        param_dim: int = 0,
        img_size: int = 128,
        patch_size: int = 4,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs
    ):
        """
        Initialize DiT backbone.
        
        Args:
            input_channels: Target channels
            output_channels: Output channels
            conditioning_channels: DM conditioning channels
            large_scale_channels: Large-scale context channels
            param_dim: Parameter conditioning dimension
            img_size: Spatial resolution
            patch_size: Transformer patch size
            hidden_size: Transformer hidden dimension
            depth: Number of transformer blocks
            num_heads: Attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            **kwargs: Additional DiT arguments
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            conditioning_channels=conditioning_channels,
            large_scale_channels=large_scale_channels,
            param_dim=param_dim,
            img_size=img_size,
        )
        
        # Import here to avoid circular imports
        from vdm.dit import DiT
        
        # DiT computes total_in_channels internally as:
        # total_in_channels = in_channels + conditioning_channels + large_scale_channels
        # So we pass in_channels = target channels only (input_channels)
        
        self.net = DiT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=input_channels,  # Target channels only
            out_channels=output_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            n_params=param_dim,
            conditioning_channels=conditioning_channels,
            large_scale_channels=large_scale_channels,
        )
    
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with standard interface.
        
        DiT uses t directly and handles concatenation internally.
        """
        # DiT's forward() concatenates x_t and conditioning internally,
        # so we pass them separately (not pre-concatenated)
        return self.net(x_t, t, conditioning, param_conditioning)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        config = super().get_config()
        config.update({
            "backbone_type": "dit",
        })
        return config


# =============================================================================
# FNO Backbone Wrapper
# =============================================================================

@BackboneRegistry.register("fno")
class FNOBackbone(BackboneBase):
    """
    FNO (Fourier Neural Operator) backbone wrapper.
    
    Wraps the FNO architecture from fno.py. FNO natively uses timestep t,
    so no conversion is needed.
    
    Note: FNO concatenates x_t with conditioning internally.
    FNO's forward signature is: forward(t, x_t, conditioning, param_conditioning)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        param_dim: int = 0,
        img_size: int = 128,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes: int = 32,
        use_film: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
        param_min: list = None,
        param_max: list = None,
        **kwargs
    ):
        """
        Initialize FNO backbone.
        
        Args:
            input_channels: Target channels
            output_channels: Output channels
            conditioning_channels: DM conditioning channels
            large_scale_channels: Large-scale context channels
            param_dim: Parameter conditioning dimension
            img_size: Spatial resolution
            hidden_channels: FNO hidden dimension
            n_layers: Number of FNO layers
            modes: Fourier modes kept (used for both dimensions)
            use_film: Enable FiLM conditioning
            use_residual: Enable residual connections
            dropout: Dropout probability
            param_min: Min values for parameter normalization
            param_max: Max values for parameter normalization
            **kwargs: Additional FNO arguments
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            conditioning_channels=conditioning_channels,
            large_scale_channels=large_scale_channels,
            param_dim=param_dim,
            img_size=img_size,
        )
        
        # Import here to avoid circular imports
        from vdm.fno import FNO2d
        
        # FNO computes total_input = in_channels + conditioning_channels + large_scale_channels
        # internally, so we pass in_channels as target only
        
        # Auto-adjust modes for small images (modes must be <= img_size // 2)
        actual_modes = min(modes, img_size // 2)
        
        self.net = FNO2d(
            in_channels=input_channels,  # Target channels only
            out_channels=output_channels,
            conditioning_channels=conditioning_channels,
            large_scale_channels=large_scale_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            modes1=actual_modes,
            modes2=actual_modes,
            n_params=param_dim,
            param_min=param_min,
            param_max=param_max,
            dropout=dropout,
            use_param_conditioning=param_dim > 0,
        )
    
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with standard interface.
        
        FNO uses t directly. Note that FNO's native signature is
        forward(t, x_t, conditioning, ...) so we reorder arguments.
        """
        # FNO expects conditioning as a separate argument, not pre-concatenated
        # If no conditioning provided, create zeros
        if conditioning is None:
            B, C, H, W = x_t.shape
            conditioning = torch.zeros(
                B, self.conditioning_channels + self.large_scale_channels, H, W,
                device=x_t.device, dtype=x_t.dtype
            )
        
        # FNO forward signature: (t, x_t, conditioning, param_conditioning)
        return self.net(t, x_t, conditioning, param_conditioning)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        config = super().get_config()
        config.update({
            "backbone_type": "fno",
        })
        return config


# =============================================================================
# Register Preset Configurations
# =============================================================================

# DiT variants (following original DiT paper)
BackboneRegistry.register_config("dit-s", {
    "_backbone_class": "dit",
    "hidden_size": 384,
    "depth": 12,
    "num_heads": 6,
    "patch_size": 4,
})

BackboneRegistry.register_config("dit-b", {
    "_backbone_class": "dit",
    "hidden_size": 768,
    "depth": 12,
    "num_heads": 12,
    "patch_size": 4,
})

BackboneRegistry.register_config("dit-l", {
    "_backbone_class": "dit",
    "hidden_size": 1024,
    "depth": 24,
    "num_heads": 16,
    "patch_size": 4,
})

BackboneRegistry.register_config("dit-xl", {
    "_backbone_class": "dit",
    "hidden_size": 1152,
    "depth": 28,
    "num_heads": 16,
    "patch_size": 4,
})

# FNO variants
BackboneRegistry.register_config("fno-s", {
    "_backbone_class": "fno",
    "hidden_channels": 32,
    "n_layers": 4,
    "modes": 16,
})

BackboneRegistry.register_config("fno-b", {
    "_backbone_class": "fno",
    "hidden_channels": 64,
    "n_layers": 4,
    "modes": 32,
})

BackboneRegistry.register_config("fno-l", {
    "_backbone_class": "fno",
    "hidden_channels": 128,
    "n_layers": 6,
    "modes": 48,
})

BackboneRegistry.register_config("fno-xl", {
    "_backbone_class": "fno",
    "hidden_channels": 256,
    "n_layers": 8,
    "modes": 64,
})

# UNet variants
BackboneRegistry.register_config("unet-s", {
    "_backbone_class": "unet",
    "embedding_dim": 64,
    "n_blocks": 3,
})

BackboneRegistry.register_config("unet-b", {
    "_backbone_class": "unet",
    "embedding_dim": 128,
    "n_blocks": 4,
})

BackboneRegistry.register_config("unet-l", {
    "_backbone_class": "unet",
    "embedding_dim": 192,
    "n_blocks": 5,
})


# =============================================================================
# Utility Functions
# =============================================================================

def create_backbone(
    backbone_type: str,
    **kwargs
) -> BackboneBase:
    """
    Convenience function to create a backbone.
    
    Args:
        backbone_type: Backbone name (e.g., "unet", "dit-b", "fno-l")
        **kwargs: Backbone configuration
    
    Returns:
        Instantiated backbone
    
    Example:
        backbone = create_backbone("dit-b", img_size=128, param_dim=6)
    """
    return BackboneRegistry.create(backbone_type, **kwargs)


def list_backbones() -> list:
    """List all available backbone types."""
    return BackboneRegistry.available()


# =============================================================================
# Backbone Info
# =============================================================================

def print_backbone_info():
    """Print information about available backbones."""
    print("\n" + "=" * 60)
    print("AVAILABLE BACKBONES")
    print("=" * 60)
    
    print("\nBase architectures:")
    print("  - unet    : UNet with attention and Fourier features")
    print("  - dit     : Diffusion Transformer (Vision Transformer)")
    print("  - fno     : Fourier Neural Operator (spectral convolutions)")
    
    print("\nPreset configurations:")
    
    dit_presets = [n for n in BackboneRegistry._configs if n.startswith("dit")]
    fno_presets = [n for n in BackboneRegistry._configs if n.startswith("fno")]
    unet_presets = [n for n in BackboneRegistry._configs if n.startswith("unet")]
    
    print("\n  DiT variants:")
    for name in sorted(dit_presets):
        cfg = BackboneRegistry._configs[name]
        print(f"    {name:8s}: hidden={cfg['hidden_size']}, depth={cfg['depth']}, heads={cfg['num_heads']}")
    
    print("\n  FNO variants:")
    for name in sorted(fno_presets):
        cfg = BackboneRegistry._configs[name]
        print(f"    {name:8s}: hidden={cfg['hidden_channels']}, layers={cfg['num_layers']}, modes={cfg['modes']}")
    
    print("\n  UNet variants:")
    for name in sorted(unet_presets):
        cfg = BackboneRegistry._configs[name]
        print(f"    {name:8s}: embed_dim={cfg['embedding_dim']}, blocks={cfg['n_blocks']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Demo usage
    print_backbone_info()
    
    # Test backbone creation
    print("\nTesting backbone creation...")
    
    # Test UNet
    unet = create_backbone("unet-b", img_size=64)
    print(f"UNet-B created: {sum(p.numel() for p in unet.parameters()):,} params")
    
    # Test DiT (only if dit.py exists)
    try:
        dit = create_backbone("dit-s", img_size=64)
        print(f"DiT-S created: {sum(p.numel() for p in dit.parameters()):,} params")
    except ImportError as e:
        print(f"DiT not available: {e}")
    
    # Test FNO (only if fno.py exists)
    try:
        fno = create_backbone("fno-b", img_size=64)
        print(f"FNO-B created: {sum(p.numel() for p in fno.parameters()):,} params")
    except ImportError as e:
        print(f"FNO not available: {e}")
    
    print("\nDone!")
