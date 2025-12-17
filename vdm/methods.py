"""
Method Abstraction Layer for VDM-BIND Generative Zoo.

This module provides a unified interface for different generative methods
(VDM, Flow Matching, Consistency Models, etc.) that can use any backbone.

Standard Interface:
    All methods follow the same patterns:
    
    Training:
        loss, metrics = method.compute_loss(batch)
    
    Sampling:
        samples = method.sample(conditioning, n_samples, n_steps)
    
This enables the "Generative Zoo" design:
    Method + Backbone + Dataset = Experiment
    
Example:
    # Create backbone
    backbone = BackboneRegistry.create("dit-b", img_size=128, param_dim=6)
    
    # Create method with backbone
    method = MethodRegistry.create("vdm", backbone=backbone, learning_rate=1e-4)
    
    # Or use preset
    method = MethodRegistry.create("vdm", backbone_type="unet-b", ...)

Architecture:
    BaseMethod (LightningModule)
    ├── VDMMethod           - Variational Diffusion Model
    ├── FlowMatchingMethod  - Flow Matching / Stochastic Interpolants
    ├── OTFlowMethod        - Optimal Transport Flow Matching
    ├── ConsistencyMethod   - Consistency Models
    └── DSMMethod           - Denoising Score Matching
    
Note: DDPM is available via the legacy ddpm_model.py using the score_models package.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Type, Callable, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from lightning.pytorch import LightningModule

from vdm.backbones import BackboneBase, BackboneRegistry, create_backbone

# =============================================================================
# Global verbosity setting
# =============================================================================
_VERBOSE = True

def set_verbose(verbose: bool):
    """Set global verbosity for method/backbone initialization."""
    global _VERBOSE
    _VERBOSE = verbose

def get_verbose() -> bool:
    """Get current verbosity setting."""
    return _VERBOSE


# =============================================================================
# Base Method Class
# =============================================================================

class BaseMethod(LightningModule, ABC):
    """
    Abstract base class for all generative methods.
    
    Methods define the training paradigm (how to learn) while backbones
    define the architecture (what network to use).
    
    All methods must implement:
        - compute_loss(): Compute training loss from a batch
        - sample(): Generate samples given conditioning
        - training_step(): PyTorch Lightning training step
        - validation_step(): PyTorch Lightning validation step
    
    Methods receive:
        - backbone: A BackboneBase instance with standard interface
        - Training hyperparameters (lr, scheduler, etc.)
        - Method-specific parameters (noise schedule, etc.)
    """
    
    # Method name for registration
    method_name: str = "base"
    
    # Default early stopping metric
    early_stopping_metric: str = "val/loss"
    
    def __init__(
        self,
        backbone: BackboneBase,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        lr_scheduler: str = "cosine",
        warmup_steps: int = 0,
        # Sampling parameters
        n_sampling_steps: int = 100,
        # Conditioning
        use_param_conditioning: bool = False,
        # Data shape
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        **kwargs
    ):
        """
        Initialize base method.
        
        Args:
            backbone: Network backbone (UNet, DiT, FNO, etc.)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            lr_scheduler: Scheduler type ('cosine', 'onecycle', 'constant')
            warmup_steps: Linear warmup steps
            n_sampling_steps: Number of steps for sampling
            use_param_conditioning: Whether to use parameter conditioning
            image_shape: Shape of output (C, H, W)
            **kwargs: Method-specific arguments
        """
        super().__init__()
        
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.warmup_steps = warmup_steps
        self.n_sampling_steps = n_sampling_steps
        self.use_param_conditioning = use_param_conditioning
        self.image_shape = image_shape
        
        # Save hyperparameters (exclude backbone to avoid serialization issues)
        self.save_hyperparameters(ignore=['backbone'])
    
    @abstractmethod
    def compute_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute training loss.
        
        Args:
            x: Target data (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            param_conditioning: Parameter conditioning (B, N_params) or None
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics to log
        """
        pass
    
    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        n_samples: int,
        n_steps: Optional[int] = None,
        param_conditioning: Optional[Tensor] = None,
        device: str = 'cuda',
        return_trajectory: bool = False,
        verbose: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Generate samples.
        
        Args:
            conditioning: Spatial conditioning (B, C_cond, H, W)
            n_samples: Number of samples (if conditioning.shape[0] != n_samples, 
                       conditioning will be broadcasted)
            n_steps: Number of sampling steps (uses default if None)
            param_conditioning: Parameter conditioning (B, N_params) or None
            device: Device to generate on
            return_trajectory: Whether to return intermediate states
            verbose: Show progress bar
        
        Returns:
            samples: Generated samples (B, C, H, W)
            trajectory: List of intermediate states if return_trajectory=True
        """
        pass
    
    def _unpack_batch(self, batch: Tuple) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Unpack batch from AstroDataset.
        
        AstroDataset returns: (m_dm, large_scale, m_target, conditions)
        
        Returns:
            x: Target data (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            params: Parameter conditioning or None
        """
        m_dm, large_scale, m_target, conditions = batch
        
        # Target is what we want to generate
        x = m_target  # (B, C, H, W)
        
        # Spatial conditioning: concatenate DM and large-scale context
        conditioning = torch.cat([m_dm, large_scale], dim=1)  # (B, 1+N, H, W)
        
        # Parameter conditioning
        params = conditions if self.use_param_conditioning else None
        
        return x, conditioning, params
    
    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Standard training step."""
        x, conditioning, params = self._unpack_batch(batch)
        loss, metrics = self.compute_loss(x, conditioning, params)
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f"train/{key}", value, prog_bar=(key == "loss"), sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Standard validation step."""
        x, conditioning, params = self._unpack_batch(batch)
        loss, metrics = self.compute_loss(x, conditioning, params)
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=(key == "loss"), sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        if self.lr_scheduler_type == "constant":
            return optimizer
        
        elif self.lr_scheduler_type == "cosine":
            # Cosine annealing with warmup
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            if self.warmup_steps > 0:
                warmup = LinearLR(
                    optimizer, 
                    start_factor=0.01, 
                    end_factor=1.0, 
                    total_iters=self.warmup_steps
                )
                cosine = CosineAnnealingLR(
                    optimizer, 
                    T_max=self.trainer.estimated_stepping_batches - self.warmup_steps
                )
                scheduler = SequentialLR(
                    optimizer, 
                    schedulers=[warmup, cosine], 
                    milestones=[self.warmup_steps]
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer, 
                    T_max=self.trainer.estimated_stepping_batches
                )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        
        elif self.lr_scheduler_type == "onecycle":
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_type}")
    
    # =========================================================================
    # BIND Interface - for inference pipeline compatibility
    # =========================================================================
    
    def draw_samples(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: Optional[int] = None,
        conditional_params: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        BIND-compatible sampling interface.
        
        This method provides compatibility with the BIND inference pipeline.
        
        Args:
            conditioning: Spatial conditioning (B, C_cond, H, W)
            batch_size: Number of samples to generate
            n_sampling_steps: Override default sampling steps
            conditional_params: Parameter conditioning (B, N_params)
            **kwargs: Additional sampling arguments
        
        Returns:
            samples: Generated samples (B, C, H, W)
        """
        n_steps = n_sampling_steps or self.n_sampling_steps
        device = next(self.parameters()).device
        
        return self.sample(
            conditioning=conditioning,
            n_samples=batch_size,
            n_steps=n_steps,
            param_conditioning=conditional_params,
            device=device,
            **kwargs
        )


# =============================================================================
# Method Registry
# =============================================================================

class MethodRegistry:
    """
    Registry for generative methods.
    
    Allows registration and instantiation of methods by name.
    
    Usage:
        # Register a method
        @MethodRegistry.register("my_method")
        class MyMethod(BaseMethod):
            ...
        
        # Create instance
        method = MethodRegistry.create("my_method", backbone=backbone, **config)
        
        # Or with automatic backbone creation
        method = MethodRegistry.create(
            "my_method", 
            backbone_type="unet-b", 
            img_size=128,
            **config
        )
    """
    
    _registry: Dict[str, Type[BaseMethod]] = {}
    _configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a method class."""
        def decorator(method_cls: Type[BaseMethod]) -> Type[BaseMethod]:
            cls._registry[name.lower()] = method_cls
            method_cls.method_name = name.lower()
            return method_cls
        return decorator
    
    @classmethod
    def register_method(cls, name: str, method_cls: Type[BaseMethod]) -> None:
        """Manually register a method class."""
        cls._registry[name.lower()] = method_cls
    
    @classmethod
    def register_config(cls, name: str, config: Dict[str, Any]) -> None:
        """Register default configuration for a method."""
        cls._configs[name.lower()] = config
    
    @classmethod
    def create(
        cls,
        name: str,
        backbone: Optional[BackboneBase] = None,
        backbone_type: Optional[str] = None,
        **kwargs
    ) -> BaseMethod:
        """
        Create a method instance by name.
        
        Args:
            name: Method name (e.g., "vdm", "flow", "consistency")
            backbone: Pre-created backbone instance
            backbone_type: Backbone type to create (e.g., "unet-b", "dit-s")
            **kwargs: Method configuration
        
        Returns:
            Instantiated method
        
        Note:
            Either `backbone` or `backbone_type` must be provided.
            If both are provided, `backbone` takes precedence.
        """
        name_lower = name.lower()
        
        if name_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(f"Unknown method: {name}. Available: {available}")
        
        # Create backbone if needed
        if backbone is None:
            if backbone_type is None:
                raise ValueError("Either 'backbone' or 'backbone_type' must be provided")
            
            # Get img_size (needed for backbone and image_shape)
            img_size = kwargs.pop('img_size', 128)
            
            # Extract backbone-relevant kwargs
            backbone_kwargs = {
                'img_size': img_size,
                'input_channels': kwargs.get('image_shape', (3, img_size, img_size))[0] if 'image_shape' in kwargs else 3,
                'output_channels': kwargs.get('image_shape', (3, img_size, img_size))[0] if 'image_shape' in kwargs else 3,
                'param_dim': kwargs.pop('param_dim', 0),
                'conditioning_channels': kwargs.pop('conditioning_channels', 1),
                'large_scale_channels': kwargs.pop('large_scale_channels', 3),
            }
            
            # Add normalization bounds if provided
            if 'param_min' in kwargs:
                backbone_kwargs['param_min'] = kwargs.pop('param_min')
            if 'param_max' in kwargs:
                backbone_kwargs['param_max'] = kwargs.pop('param_max')
            
            backbone = create_backbone(backbone_type, **backbone_kwargs)
            
            # Ensure image_shape is set correctly if not explicitly provided
            if 'image_shape' not in kwargs:
                n_channels = backbone_kwargs['output_channels']
                kwargs['image_shape'] = (n_channels, img_size, img_size)
        
        # Get default config if registered
        if name_lower in cls._configs:
            merged = {**cls._configs[name_lower], **kwargs}
        else:
            merged = kwargs
        
        method_cls = cls._registry[name_lower]
        return method_cls(backbone=backbone, **merged)
    
    @classmethod
    def available(cls) -> List[str]:
        """List available method names."""
        return sorted(cls._registry.keys())
    
    @classmethod
    def get_class(cls, name: str) -> Type[BaseMethod]:
        """Get method class by name."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise KeyError(f"Unknown method: {name}")
        return cls._registry[name_lower]
    
    @classmethod
    def get_early_stopping_metric(cls, name: str) -> str:
        """Get the recommended early stopping metric for a method."""
        method_cls = cls.get_class(name)
        return method_cls.early_stopping_metric


# =============================================================================
# Utility Functions
# =============================================================================

def create_method(
    method_type: str,
    backbone_type: str = "unet-b",
    **kwargs
) -> BaseMethod:
    """
    Convenience function to create a method with backbone.
    
    Args:
        method_type: Method name (e.g., "vdm", "flow")
        backbone_type: Backbone name (e.g., "unet-b", "dit-s")
        **kwargs: Method and backbone configuration
    
    Returns:
        Instantiated method
    
    Example:
        method = create_method(
            "vdm",
            backbone_type="dit-b",
            img_size=128,
            learning_rate=1e-4,
        )
    """
    return MethodRegistry.create(method_type, backbone_type=backbone_type, **kwargs)


def list_methods() -> List[str]:
    """List all available method types."""
    return MethodRegistry.available()


def print_method_info():
    """Print information about available methods."""
    print("\n" + "=" * 60)
    print("AVAILABLE METHODS")
    print("=" * 60)
    
    for name in sorted(MethodRegistry.available()):
        method_cls = MethodRegistry.get_class(name)
        metric = method_cls.early_stopping_metric
        print(f"  {name:15s} - monitor: {metric}")
    
    print("\n" + "=" * 60)


# =============================================================================
# Register default method configs
# =============================================================================

MethodRegistry.register_config("vdm", {
    "n_sampling_steps": 250,
    "lr_scheduler": "onecycle",
})

MethodRegistry.register_config("flow", {
    "n_sampling_steps": 50,
    "lr_scheduler": "cosine",
})

MethodRegistry.register_config("consistency", {
    "n_sampling_steps": 1,
    "lr_scheduler": "cosine",
})


# =============================================================================
# VDM Method Implementation
# =============================================================================

@MethodRegistry.register("vdm")
class VDMMethod(BaseMethod):
    """
    Variational Diffusion Model method.
    
    Implements the VDM training paradigm:
    - Forward: Add noise to data using variance-preserving diffusion
    - Training: Predict the noise that was added
    - Sampling: Iteratively denoise from pure noise
    
    Loss: ELBO = diffusion_loss + latent_loss + reconstruction_loss
    
    Reference: arxiv:2107.00630
    """
    
    method_name = "vdm"
    early_stopping_metric = "val/elbo"
    
    def __init__(
        self,
        backbone: BackboneBase,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        lr_scheduler: str = "onecycle",
        warmup_steps: int = 0,
        # VDM-specific parameters
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        noise_schedule: str = "fixed_linear",
        antithetic_time_sampling: bool = True,
        # Loss weights
        lambdas: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        channel_weights: Tuple[float, ...] = (1.0, 1.0, 1.0),
        data_noise: Union[float, Tuple[float, ...]] = 1e-3,
        # Focal loss
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        # Sampling
        n_sampling_steps: int = 250,
        # Conditioning
        use_param_conditioning: bool = False,
        # Data shape
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            n_sampling_steps=n_sampling_steps,
            use_param_conditioning=use_param_conditioning,
            image_shape=image_shape,
        )
        
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.antithetic_time_sampling = antithetic_time_sampling
        self.lambdas = lambdas
        
        # Per-channel data noise
        if isinstance(data_noise, (tuple, list)):
            self.register_buffer('data_noise', torch.tensor(data_noise, dtype=torch.float32))
            self.use_per_channel_data_noise = True
        else:
            self.data_noise = data_noise
            self.use_per_channel_data_noise = False
        
        # Channel weights
        self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))
        
        # Focal loss
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        # Noise schedule
        from vdm.utils import FixedLinearSchedule, LearnedLinearSchedule, NNSchedule
        
        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(gamma_min=gamma_min, gamma_max=gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(gamma_min=gamma_min, gamma_max=gamma_max)
        elif noise_schedule == "learned_nn":
            self.gamma = NNSchedule(gamma_min=gamma_min, gamma_max=gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        
        # Print config
        if get_verbose():
            print(f"\n{'='*60}")
            print(f"VDM METHOD INITIALIZED")
            print(f"{'='*60}")
            print(f"  Backbone: {type(backbone).__name__}")
            print(f"  Gamma range: [{gamma_min}, {gamma_max}]")
            print(f"  Noise schedule: {noise_schedule}")
            print(f"  Sampling steps: {n_sampling_steps}")
            print(f"  Channel weights: {channel_weights}")
            print(f"  Focal loss: {use_focal_loss}")
            print(f"{'='*60}\n")
    
    def alpha(self, gamma: Tensor) -> Tensor:
        """Signal coefficient: sqrt(sigmoid(-gamma))"""
        return torch.sqrt(torch.sigmoid(-gamma))
    
    def sigma(self, gamma: Tensor) -> Tensor:
        """Noise coefficient: sqrt(sigmoid(gamma))"""
        return torch.sqrt(torch.sigmoid(gamma))
    
    def variance_preserving_map(
        self,
        x: Tensor,
        times: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Add noise to data: z_t = alpha_t * x + sigma_t * noise"""
        with torch.enable_grad():
            times_reshaped = times.view((times.shape[0],) + (1,) * (x.ndim - 1))
            gamma_t = self.gamma(times_reshaped)
        
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma(gamma_t)
        
        if noise is None:
            noise = torch.randn_like(x)
        
        z_t = alpha_t * x + sigma_t * noise
        return z_t, gamma_t
    
    def sample_times(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample diffusion times."""
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=device)
        else:
            times = torch.rand(batch_size, device=device)
        return times
    
    def compute_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute VDM ELBO loss."""
        batch_size = x.shape[0]
        device = x.device
        
        # BPD factor for loss scaling
        bpd_factor = 1.0 / (np.prod(self.image_shape) * np.log(2))
        
        # Sample times and noise
        times = self.sample_times(batch_size, device)
        times.requires_grad_(True)
        noise = torch.randn_like(x)
        
        # Add noise to data
        z_t, gamma_t = self.variance_preserving_map(x, times, noise)
        
        # Convert time to t ∈ [0,1] for backbone
        # gamma = gamma_max - (gamma_max - gamma_min) * t
        # So t = (gamma_max - gamma) / (gamma_max - gamma_min)
        # Use view() instead of squeeze() to preserve batch dimension
        gamma_t_flat = gamma_t.view(batch_size)
        t_normalized = (self.gamma_max - gamma_t_flat) / (self.gamma_max - self.gamma_min)
        
        # Predict noise using backbone
        pred_noise = self.backbone(
            z_t,
            t_normalized,
            conditioning=conditioning,
            param_conditioning=param_conditioning,
        )
        
        # Compute diffusion loss with autograd for gamma gradient
        from torch import autograd
        gamma_grad = autograd.grad(
            gamma_t,
            times,
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Per-channel MSE with weights
        n_channels = pred_noise.shape[1]
        channel_losses = []
        
        for c in range(n_channels):
            pred_c = pred_noise[:, c:c+1]
            true_c = noise[:, c:c+1]
            mse_c = ((pred_c - true_c) ** 2).flatten(start_dim=1).sum(dim=-1)
            
            # Apply focal loss for stellar channel if enabled
            if self.use_focal_loss and c == 2 and n_channels == 3:
                p_t = torch.exp(-mse_c)
                focal_weight = (1 - p_t) ** self.focal_gamma
                weighted_mse = self.channel_weights[c] * focal_weight * mse_c
            else:
                weighted_mse = self.channel_weights[c] * mse_c
            
            channel_losses.append(weighted_mse)
        
        total_mse = sum(channel_losses)
        diffusion_loss = (bpd_factor * 0.5 * total_mse * gamma_grad).mean()
        
        # Latent loss (simplified)
        gamma_1 = self.gamma(torch.ones(1, device=device))
        sigma_1_sq = self.sigma(gamma_1) ** 2
        latent_loss = bpd_factor * 0.5 * np.prod(self.image_shape) * (
            1.0 + np.log(2 * np.pi) + torch.log(sigma_1_sq).item()
        )
        
        # Reconstruction loss (simplified)
        gamma_0 = self.gamma(torch.zeros(1, device=device))
        sigma_0_sq = self.sigma(gamma_0).squeeze() ** 2
        if self.use_per_channel_data_noise:
            recons_loss = bpd_factor * 0.5 * sum([
                self.image_shape[1] * self.image_shape[2] * (
                    np.log(2 * np.pi) + 2 * np.log(self.data_noise[c].item()) +
                    sigma_0_sq.item() / (self.data_noise[c].item() ** 2)
                )
                for c in range(n_channels)
            ])
        else:
            recons_loss = bpd_factor * 0.5 * np.prod(self.image_shape) * (
                np.log(2 * np.pi) + 2 * np.log(self.data_noise) +
                sigma_0_sq.item() / (self.data_noise ** 2)
            )
        
        # Total ELBO
        total_loss = (
            self.lambdas[0] * diffusion_loss +
            self.lambdas[1] * latent_loss +
            self.lambdas[2] * recons_loss
        )
        
        metrics = {
            "elbo": total_loss.item(),
            "loss": total_loss.item(),
            "diffusion_loss": diffusion_loss.item(),
            "latent_loss": latent_loss.item() if isinstance(latent_loss, Tensor) else latent_loss,
            "reconstruction_loss": recons_loss.item() if isinstance(recons_loss, Tensor) else recons_loss,
        }
        
        return total_loss, metrics
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        n_samples: int,
        n_steps: Optional[int] = None,
        param_conditioning: Optional[Tensor] = None,
        device: str = 'cuda',
        return_trajectory: bool = False,
        verbose: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """Generate samples via iterative denoising."""
        from torch.special import expm1
        from tqdm import trange
        
        self.backbone.eval()
        n_steps = n_steps or self.n_sampling_steps
        
        # Initialize from noise
        z = torch.randn(n_samples, *self.image_shape, device=device)
        
        # Move conditioning to device
        conditioning = conditioning.to(device)
        
        # Broadcast conditioning if needed
        if conditioning.shape[0] != n_samples:
            conditioning = conditioning.expand(n_samples, -1, -1, -1)
        if param_conditioning is not None:
            param_conditioning = param_conditioning.to(device)
            if param_conditioning.shape[0] != n_samples:
                param_conditioning = param_conditioning.expand(n_samples, -1)
        
        trajectory = [z.clone()] if return_trajectory else None
        
        # Iterate from t=1 to t=0
        iterator = trange(n_steps) if verbose else range(n_steps)
        for i in iterator:
            t = 1.0 - i / n_steps
            s = 1.0 - (i + 1) / n_steps
            
            t_tensor = torch.full((n_samples,), t, device=device)
            s_tensor = torch.full((n_samples,), s, device=device)
            
            gamma_t = self.gamma(t_tensor.view(-1, 1, 1, 1))
            gamma_s = self.gamma(s_tensor.view(-1, 1, 1, 1))
            
            c = -expm1(gamma_s - gamma_t)
            alpha_t = self.alpha(gamma_t)
            alpha_s = self.alpha(gamma_s)
            sigma_t = self.sigma(gamma_t)
            sigma_s = self.sigma(gamma_s)
            
            # Convert to normalized time for backbone (use view to preserve batch dim)
            t_normalized = (self.gamma_max - gamma_t.view(n_samples)) / (self.gamma_max - self.gamma_min)
            
            # Predict noise
            pred_noise = self.backbone(
                z, t_normalized, conditioning=conditioning, param_conditioning=param_conditioning
            )
            
            # Compute p(z_s | z_t)
            mean = alpha_s / alpha_t * (z - c * sigma_t * pred_noise)
            scale = sigma_s * torch.sqrt(c)
            
            z = mean + scale * torch.randn_like(z) if i < n_steps - 1 else mean
            
            if return_trajectory:
                trajectory.append(z.clone())
        
        if return_trajectory:
            return z, trajectory
        return z


# =============================================================================
# Flow Matching Method Implementation
# =============================================================================

@MethodRegistry.register("flow")
class FlowMatchingMethod(BaseMethod):
    """
    Flow Matching / Stochastic Interpolant method.
    
    Learns a velocity field that transports from a source (noise/zeros) to target.
    
    Training:
        - Interpolate: x_t = t * x_1 + (1-t) * x_0
        - Velocity target: v = x_1 - x_0
        - Loss: MSE between predicted and true velocity
    
    Sampling:
        - Integrate ODE: dx/dt = v(t, x) from t=0 to t=1
    
    Reference: BaryonBridge (Sadr et al.)
    """
    
    method_name = "flow"
    early_stopping_metric = "val/loss"
    
    def __init__(
        self,
        backbone: BackboneBase,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        warmup_steps: int = 0,
        # Flow-specific parameters
        use_stochastic_interpolant: bool = False,
        sigma: float = 0.0,
        x0_mode: str = "zeros",  # 'zeros', 'noise', 'dm_copy'
        # Sampling
        n_sampling_steps: int = 50,
        # Conditioning
        use_param_conditioning: bool = False,
        # Data shape
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            n_sampling_steps=n_sampling_steps,
            use_param_conditioning=use_param_conditioning,
            image_shape=image_shape,
        )
        
        self.use_stochastic_interpolant = use_stochastic_interpolant
        self.sigma = sigma
        self.x0_mode = x0_mode
        
        if get_verbose():
            print(f"\n{'='*60}")
            print(f"FLOW MATCHING METHOD INITIALIZED")
            print(f"{'='*60}")
            print(f"  Backbone: {type(backbone).__name__}")
            print(f"  Sampling steps: {n_sampling_steps}")
            print(f"  Stochastic: {use_stochastic_interpolant} (sigma={sigma})")
            print(f"  x0 mode: {x0_mode}")
            print(f"{'='*60}\n")
    
    def _get_x0(self, x1: Tensor, dm_condition: Optional[Tensor] = None) -> Tensor:
        """Initialize source distribution."""
        B, C, H, W = x1.shape
        
        if self.x0_mode == "zeros":
            return torch.zeros_like(x1)
        elif self.x0_mode == "noise":
            return torch.randn_like(x1)
        elif self.x0_mode == "dm_copy":
            if dm_condition is not None:
                return dm_condition.expand(-1, C, -1, -1).clone()
            return torch.zeros_like(x1)
        else:
            raise ValueError(f"Unknown x0_mode: {self.x0_mode}")
    
    def compute_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute flow matching loss."""
        x1 = x  # Target
        dm_condition = conditioning[:, :1]  # First channel is DM
        
        # Initialize x0
        x0 = self._get_x0(x1, dm_condition)
        
        # Sample time uniformly
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        
        # Linear interpolation: x_t = t * x_1 + (1-t) * x_0
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = t_expanded * x1 + (1 - t_expanded) * x0
        
        # True velocity: v = x_1 - x_0
        v_true = x1 - x0
        
        # Add noise for stochastic interpolant
        if self.use_stochastic_interpolant and self.sigma > 0:
            sigma_t = self.sigma * torch.sqrt(2 * t_expanded * (1 - t_expanded))
            epsilon = torch.randn_like(x_t)
            x_t = x_t + sigma_t * epsilon
            
            # Correct velocity for noise term
            t_clamped = torch.clamp(t_expanded, min=1e-5, max=1 - 1e-5)
            gamma_dot = self.sigma * (1 - 2 * t_expanded) / torch.sqrt(2 * t_clamped * (1 - t_clamped))
            v_true = v_true + gamma_dot * epsilon
        
        # Predict velocity
        v_pred = self.backbone(x_t, t, conditioning=conditioning, param_conditioning=param_conditioning)
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(v_pred, v_true)
        
        metrics = {
            "loss": loss.item(),
            "mse": loss.item(),
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        n_samples: int,
        n_steps: Optional[int] = None,
        param_conditioning: Optional[Tensor] = None,
        device: str = 'cuda',
        return_trajectory: bool = False,
        verbose: bool = False,
        stochastic: bool = False,
        sde_noise_scale: Optional[float] = None,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """Generate samples via ODE/SDE integration."""
        import math
        from tqdm import trange
        
        self.backbone.eval()
        n_steps = n_steps or self.n_sampling_steps
        
        # Move conditioning to device
        conditioning = conditioning.to(device)
        
        # Initialize x0
        if conditioning.shape[0] != n_samples:
            conditioning = conditioning.expand(n_samples, -1, -1, -1)
        
        dm_condition = conditioning[:, :1]
        x = self._get_x0(
            torch.zeros(n_samples, *self.image_shape, device=device),
            dm_condition
        )
        
        if param_conditioning is not None:
            param_conditioning = param_conditioning.to(device)
            if param_conditioning.shape[0] != n_samples:
                param_conditioning = param_conditioning.expand(n_samples, -1)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        # Integrate from t=0 to t=1
        dt = 1.0 / n_steps
        noise_scale = sde_noise_scale if sde_noise_scale is not None else (self.sigma if self.sigma > 0 else 0.1)
        
        iterator = trange(n_steps) if verbose else range(n_steps)
        for i in iterator:
            t = torch.full((n_samples,), i * dt, device=device, dtype=x.dtype)
            
            # Predict velocity
            v = self.backbone(x, t, conditioning=conditioning, param_conditioning=param_conditioning)
            
            # Euler step
            x = x + v * dt
            
            # Add noise for SDE sampling
            if stochastic and i < n_steps - 1:
                scale = noise_scale * math.sqrt(dt) * (1.0 - (i + 1) * dt)
                x = x + scale * torch.randn_like(x)
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return x, trajectory
        return x


# =============================================================================
# Consistency Method Implementation
# =============================================================================

@MethodRegistry.register("consistency")
class ConsistencyMethod(BaseMethod):
    """
    Consistency Model method.
    
    Learns to map any noisy point directly to clean data.
    Enables single-step or few-step sampling.
    
    f_θ(x_t, t) = x_0 for all t on the same trajectory.
    
    Reference: Song et al. (2023) "Consistency Models"
    """
    
    method_name = "consistency"
    early_stopping_metric = "val/loss"
    
    def __init__(
        self,
        backbone: BackboneBase,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        warmup_steps: int = 0,
        # Consistency-specific parameters
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        # Training mode
        use_denoising_pretraining: bool = True,
        denoising_warmup_epochs: int = 10,
        ct_n_steps: int = 18,  # Number of discretization steps for CT
        ema_decay: float = 0.9999,
        # Sampling
        n_sampling_steps: int = 1,
        # Conditioning
        use_param_conditioning: bool = False,
        # Data shape
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            n_sampling_steps=n_sampling_steps,
            use_param_conditioning=use_param_conditioning,
            image_shape=image_shape,
        )
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.use_denoising_pretraining = use_denoising_pretraining
        self.denoising_warmup_epochs = denoising_warmup_epochs
        self.ct_n_steps = ct_n_steps
        self.ema_decay = ema_decay
        
        # Track training phase
        self._in_denoising_phase = use_denoising_pretraining
        
        if get_verbose():
            print(f"\n{'='*60}")
            print(f"CONSISTENCY METHOD INITIALIZED")
            print(f"{'='*60}")
            print(f"  Backbone: {type(backbone).__name__}")
            print(f"  Sigma range: [{sigma_min}, {sigma_max}]")
            print(f"  Sampling steps: {n_sampling_steps}")
            print(f"  Denoising pretraining: {use_denoising_pretraining}")
            print(f"{'='*60}\n")
    
    def get_sigma(self, t: Tensor) -> Tensor:
        """Get noise level from time t ∈ [0, 1]."""
        sigma_min_inv_rho = self.sigma_min ** (1 / self.rho)
        sigma_max_inv_rho = self.sigma_max ** (1 / self.rho)
        return (sigma_min_inv_rho + t * (sigma_max_inv_rho - sigma_min_inv_rho)) ** self.rho
    
    def get_scalings(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Get c_skip, c_out, c_in scalings."""
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        
        sigma_data_sq = self.sigma_data ** 2
        c_skip = sigma_data_sq / (sigma ** 2 + sigma_data_sq)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + sigma_data_sq)
        c_in = 1 / torch.sqrt(sigma ** 2 + sigma_data_sq)
        
        return c_skip, c_out, c_in
    
    def consistency_function(
        self,
        x: Tensor,
        sigma: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply consistency function with skip connection."""
        c_skip, c_out, c_in = self.get_scalings(sigma)
        
        # Convert sigma to time for backbone
        t = (sigma ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)) / (
            self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)
        )
        if t.dim() > 1:
            t = t.squeeze()
        
        # Network prediction
        F_x = self.backbone(c_in * x, t, conditioning=conditioning, param_conditioning=param_conditioning)
        
        # Skip connection
        return c_skip * x + c_out * F_x
    
    def compute_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute consistency or denoising loss."""
        batch_size = x.shape[0]
        device = x.device
        
        # Check if in denoising phase
        if self._in_denoising_phase:
            return self._compute_denoising_loss(x, conditioning, param_conditioning)
        else:
            return self._compute_consistency_loss(x, conditioning, param_conditioning)
    
    def _compute_denoising_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Simple denoising loss for pretraining."""
        batch_size = x.shape[0]
        device = x.device
        
        # Sample noise level
        t = torch.rand(batch_size, device=device)
        sigma = self.get_sigma(t)
        
        # Add noise
        noise = torch.randn_like(x)
        x_noisy = x + sigma.view(-1, 1, 1, 1) * noise
        
        # Predict clean data
        x_pred = self.consistency_function(x_noisy, sigma, conditioning, param_conditioning)
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(x_pred, x)
        
        return loss, {"loss": loss.item(), "denoising_loss": loss.item()}
    
    def _compute_consistency_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Consistency training loss."""
        batch_size = x.shape[0]
        device = x.device
        
        # Get discretized sigma levels
        sigma_levels = self._get_discrete_sigmas(device)
        n_levels = len(sigma_levels) - 1
        
        # Sample adjacent pairs
        idx = torch.randint(0, n_levels, (batch_size,), device=device)
        sigma_n = sigma_levels[idx]
        sigma_n_plus_1 = sigma_levels[idx + 1]
        
        # Add noise at sigma_n+1
        noise = torch.randn_like(x)
        x_n_plus_1 = x + sigma_n_plus_1.view(-1, 1, 1, 1) * noise
        
        # Apply consistency function at both levels
        f_n_plus_1 = self.consistency_function(x_n_plus_1, sigma_n_plus_1, conditioning, param_conditioning)
        
        # For x_n, we use the ODE update (simplified Euler step)
        x_n = x + sigma_n.view(-1, 1, 1, 1) * noise
        with torch.no_grad():
            f_n = self.consistency_function(x_n, sigma_n, conditioning, param_conditioning)
        
        # Consistency loss
        loss = torch.nn.functional.mse_loss(f_n_plus_1, f_n)
        
        return loss, {"loss": loss.item(), "consistency_loss": loss.item()}
    
    def _get_discrete_sigmas(self, device: torch.device) -> Tensor:
        """Get discretized sigma schedule."""
        t_values = torch.linspace(0, 1, self.ct_n_steps + 1, device=device)
        return self.get_sigma(t_values)
    
    def on_train_epoch_start(self):
        """Check if denoising phase should end."""
        if self._in_denoising_phase and self.current_epoch >= self.denoising_warmup_epochs:
            self._in_denoising_phase = False
            print(f"\n✓ Switching from denoising pretraining to consistency training at epoch {self.current_epoch}")
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        n_samples: int,
        n_steps: Optional[int] = None,
        param_conditioning: Optional[Tensor] = None,
        device: str = 'cuda',
        return_trajectory: bool = False,
        verbose: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """Generate samples (can be single-step!)."""
        from tqdm import trange
        
        self.backbone.eval()
        n_steps = n_steps or self.n_sampling_steps
        
        # Move conditioning to device
        conditioning = conditioning.to(device)
        
        # Broadcast conditioning
        if conditioning.shape[0] != n_samples:
            conditioning = conditioning.expand(n_samples, -1, -1, -1)
        if param_conditioning is not None:
            param_conditioning = param_conditioning.to(device)
            if param_conditioning.shape[0] != n_samples:
                param_conditioning = param_conditioning.expand(n_samples, -1)
        
        # Initialize from noise at sigma_max
        x = self.sigma_max * torch.randn(n_samples, *self.image_shape, device=device)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        if n_steps == 1:
            # Single-step sampling!
            sigma = torch.full((n_samples,), self.sigma_max, device=device)
            x = self.consistency_function(x, sigma, conditioning, param_conditioning)
        else:
            # Multi-step refinement
            sigma_levels = self._get_discrete_sigmas(device)
            step_indices = torch.linspace(len(sigma_levels) - 1, 0, n_steps + 1).long()
            
            iterator = trange(n_steps) if verbose else range(n_steps)
            for i in iterator:
                sigma = sigma_levels[step_indices[i]].expand(n_samples)
                x = self.consistency_function(x, sigma, conditioning, param_conditioning)
                
                # Add noise for next step (except last)
                if i < n_steps - 1:
                    next_sigma = sigma_levels[step_indices[i + 1]]
                    x = x + next_sigma * torch.randn_like(x)
                
                if return_trajectory:
                    trajectory.append(x.clone())
        
        if return_trajectory:
            return x, trajectory
        return x


# =============================================================================
# DSM (Denoising Score Matching) Method Implementation
# =============================================================================

@MethodRegistry.register("dsm")
class DSMMethod(BaseMethod):
    """
    Denoising Score Matching method.
    
    VP-SDE formulation that predicts noise epsilon from noisy samples.
    Uses the same architecture as VDM for fair comparison.
    
    Training:
        - Forward: z_t = alpha_t * x + sigma_t * epsilon
        - Loss: || model(z_t, t, cond) - epsilon ||^2 * w(t)
    
    Sampling:
        - DDPM-style reverse diffusion from noise to data
    
    Reference: Song et al. (2021) "Score-Based Generative Modeling through SDEs"
    """
    
    method_name = "dsm"
    early_stopping_metric = "val/loss"
    
    def __init__(
        self,
        backbone: BackboneBase,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        warmup_steps: int = 0,
        # DSM-specific parameters
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        use_snr_weighting: bool = True,
        # Loss weighting
        channel_weights: Tuple[float, ...] = (1.0, 1.0, 1.0),
        # Sampling
        n_sampling_steps: int = 250,
        # Conditioning
        use_param_conditioning: bool = False,
        # Data shape
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            n_sampling_steps=n_sampling_steps,
            use_param_conditioning=use_param_conditioning,
            image_shape=image_shape,
        )
        
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.use_snr_weighting = use_snr_weighting
        self.channel_weights = channel_weights
        
        # Register channel weights as buffer
        self.register_buffer(
            "channel_weight_tensor",
            torch.tensor(channel_weights).view(1, -1, 1, 1)
        )
        
        self.save_hyperparameters(ignore=['backbone'])
        
        if get_verbose():
            print(f"\n{'='*60}")
            print("DSM METHOD INITIALIZED")
            print("="*60)
            print(f"  Backbone: {backbone.__class__.__name__}")
            print(f"  Beta range: [{beta_min}, {beta_max}]")
            print(f"  SNR weighting: {use_snr_weighting}")
            print(f"  Sampling steps: {n_sampling_steps}")
            print(f"  Channel weights: {channel_weights}")
            print("="*60 + "\n")
    
    def alpha(self, t: Tensor) -> Tensor:
        """Compute alpha(t) for VP-SDE."""
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)
    
    def sigma(self, t: Tensor) -> Tensor:
        """Compute sigma(t) for VP-SDE."""
        alpha_t = self.alpha(t)
        return torch.sqrt(1 - alpha_t ** 2 + 1e-8)
    
    def snr(self, t: Tensor) -> Tensor:
        """Compute SNR = alpha^2 / sigma^2."""
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        return alpha_t ** 2 / (sigma_t ** 2 + 1e-8)
    
    def compute_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        params: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute DSM loss."""
        batch_size = x.shape[0]
        device = x.device
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=device)
        
        # Get schedule values
        alpha_t = self.alpha(t)[:, None, None, None]
        sigma_t = self.sigma(t)[:, None, None, None]
        
        # Sample noise
        epsilon = torch.randn_like(x)
        
        # Forward diffusion: z_t = alpha_t * x + sigma_t * epsilon
        z_t = alpha_t * x + sigma_t * epsilon
        
        # Predict noise
        epsilon_pred = self.backbone(z_t, t, conditioning, params)
        
        # MSE loss
        mse = (epsilon_pred - epsilon) ** 2
        
        # Apply channel weights
        mse_weighted = mse * self.channel_weight_tensor.to(device)
        
        # SNR weighting (optional)
        if self.use_snr_weighting:
            snr = self.snr(t)[:, None, None, None]
            mse_weighted = mse_weighted * snr
        
        loss = mse_weighted.mean()
        
        metrics = {
            'loss': loss.item(),
            'mse': mse.mean().item(),
        }
        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, conditioning, params = self._unpack_batch(batch)
        loss, metrics = self.compute_loss(x, conditioning, params)
        
        for key, value in metrics.items():
            self.log(f"train/{key}", value, prog_bar=(key == "loss"), sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, conditioning, params = self._unpack_batch(batch)
        loss, metrics = self.compute_loss(x, conditioning, params)
        
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=(key == "loss"), sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        n_samples: int = 1,
        n_steps: Optional[int] = None,
        param_conditioning: Optional[Tensor] = None,
        device: str = "cuda",
        return_trajectory: bool = False,
        verbose: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """Sample using DDPM-style reverse process."""
        self.backbone.eval()
        n_steps = n_steps or self.n_sampling_steps
        
        # Move conditioning to device
        conditioning = conditioning.to(device)
        
        # Broadcast conditioning
        if conditioning.shape[0] != n_samples:
            conditioning = conditioning.expand(n_samples, -1, -1, -1)
        if param_conditioning is not None:
            param_conditioning = param_conditioning.to(device)
            if param_conditioning.shape[0] != n_samples:
                param_conditioning = param_conditioning.expand(n_samples, -1)
        
        # Time steps
        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)
        
        # Start from noise
        x = torch.randn(n_samples, *self.image_shape, device=device)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        iterator = tqdm(range(n_steps), desc="Sampling") if verbose else range(n_steps)
        
        for i in iterator:
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            
            # Current time for batch
            t_batch = torch.full((n_samples,), t_curr, device=device)
            
            # Predict noise
            epsilon_pred = self.backbone(x, t_batch, conditioning, param_conditioning)
            
            # DDPM update
            alpha_curr = self.alpha(t_curr)
            sigma_curr = self.sigma(t_curr)
            alpha_next = self.alpha(t_next)
            sigma_next = self.sigma(t_next)
            
            # Predict x0
            x0_pred = (x - sigma_curr * epsilon_pred) / alpha_curr
            
            # Compute posterior mean
            if t_next > 0:
                x = alpha_next * x0_pred + sigma_next * torch.randn_like(x)
            else:
                x = x0_pred
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return x, trajectory
        return x


# =============================================================================
# OT Flow (Optimal Transport Flow Matching) Method Implementation
# =============================================================================

@MethodRegistry.register("ot_flow")
class OTFlowMethod(BaseMethod):
    """
    Optimal Transport Flow Matching method.
    
    Uses optimal transport coupling for straighter interpolation paths.
    
    Training:
        - Compute OT plan between batch samples
        - Transport: x_t = (1-t) * x_0 + t * x_1 with OT coupling
        - Velocity: v = x_1 - x_0 (under OT pairing)
        - Loss: MSE between predicted and OT velocity
    
    Sampling:
        - Integrate ODE: dx/dt = v(t, x) from t=0 to t=1
    
    Reference: Lipman et al. (2022) "Flow Matching for Generative Modeling"
    """
    
    method_name = "ot_flow"
    early_stopping_metric = "val/loss"
    
    def __init__(
        self,
        backbone: BackboneBase,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        warmup_steps: int = 0,
        # OT-specific parameters
        ot_method: str = "exact",  # 'exact' or 'sinkhorn'
        ot_reg: float = 0.01,  # Sinkhorn regularization
        x0_mode: str = "zeros",  # 'zeros', 'noise'
        # Sampling
        n_sampling_steps: int = 50,
        # Conditioning
        use_param_conditioning: bool = False,
        # Data shape
        image_shape: Tuple[int, int, int] = (3, 128, 128),
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            n_sampling_steps=n_sampling_steps,
            use_param_conditioning=use_param_conditioning,
            image_shape=image_shape,
        )
        
        self.ot_method = ot_method
        self.ot_reg = ot_reg
        self.x0_mode = x0_mode
        
        # Check for POT library
        try:
            import ot
            self._has_pot = True
        except ImportError:
            self._has_pot = False
            if get_verbose():
                print("Warning: POT library not installed. Using random coupling instead.")
                print("Install with: pip install POT")
        
        self.save_hyperparameters(ignore=['backbone'])
        
        if get_verbose():
            print(f"\n{'='*60}")
            print("OT FLOW METHOD INITIALIZED")
            print("="*60)
            print(f"  Backbone: {backbone.__class__.__name__}")
            print(f"  OT method: {ot_method}")
            print(f"  x0 mode: {x0_mode}")
            print(f"  Sampling steps: {n_sampling_steps}")
            print(f"  POT available: {self._has_pot}")
            print("="*60 + "\n")
    
    def compute_ot_plan(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Compute optimal transport coupling."""
        batch_size = x0.shape[0]
        device = x0.device
        
        if not self._has_pot:
            # Fallback: random permutation
            perm = torch.randperm(batch_size, device=device)
            return perm
        
        import ot
        
        # Flatten for distance computation
        x0_flat = x0.view(batch_size, -1).cpu().numpy()
        x1_flat = x1.view(batch_size, -1).cpu().numpy()
        
        # Compute cost matrix (squared L2 distance)
        M = ot.dist(x0_flat, x1_flat, metric='sqeuclidean')
        
        # Uniform marginals
        a = np.ones(batch_size) / batch_size
        b = np.ones(batch_size) / batch_size
        
        # Compute OT plan
        if self.ot_method == 'exact':
            plan = ot.emd(a, b, M)
        else:  # sinkhorn
            plan = ot.sinkhorn(a, b, M, self.ot_reg)
        
        # Convert plan to permutation (argmax per row)
        perm = plan.argmax(axis=1)
        
        return torch.tensor(perm, device=device)
    
    def compute_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        params: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute OT Flow loss."""
        batch_size = x.shape[0]
        device = x.device
        
        # Generate source samples (x0)
        if self.x0_mode == "zeros":
            x0 = torch.zeros_like(x)
        else:  # noise
            x0 = torch.randn_like(x)
        
        x1 = x  # Target
        
        # Compute OT coupling
        perm = self.compute_ot_plan(x0, x1)
        x1_coupled = x1[perm]
        
        # If params and conditioning are provided, also couple them
        if params is not None:
            params_coupled = params[perm]
        else:
            params_coupled = None
        
        conditioning_coupled = conditioning[perm]
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=device)[:, None, None, None]
        
        # Interpolate with OT coupling
        x_t = (1 - t) * x0 + t * x1_coupled
        
        # True velocity (OT path)
        v_true = x1_coupled - x0
        
        # Predict velocity
        t_flat = t.squeeze()
        v_pred = self.backbone(x_t, t_flat, conditioning_coupled, params_coupled)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_true)
        
        metrics = {
            'loss': loss.item(),
            'mse': loss.item(),
        }
        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, conditioning, params = self._unpack_batch(batch)
        loss, metrics = self.compute_loss(x, conditioning, params)
        
        for key, value in metrics.items():
            self.log(f"train/{key}", value, prog_bar=(key == "loss"), sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, conditioning, params = self._unpack_batch(batch)
        loss, metrics = self.compute_loss(x, conditioning, params)
        
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=(key == "loss"), sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        n_samples: int = 1,
        n_steps: Optional[int] = None,
        param_conditioning: Optional[Tensor] = None,
        device: str = "cuda",
        return_trajectory: bool = False,
        verbose: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """Sample using Euler integration."""
        self.backbone.eval()
        n_steps = n_steps or self.n_sampling_steps
        
        # Move conditioning to device
        conditioning = conditioning.to(device)
        
        # Broadcast conditioning
        if conditioning.shape[0] != n_samples:
            conditioning = conditioning.expand(n_samples, -1, -1, -1)
        if param_conditioning is not None:
            param_conditioning = param_conditioning.to(device)
            if param_conditioning.shape[0] != n_samples:
                param_conditioning = param_conditioning.expand(n_samples, -1)
        
        # Initialize from source distribution
        if self.x0_mode == "zeros":
            x = torch.zeros(n_samples, *self.image_shape, device=device)
        else:
            x = torch.randn(n_samples, *self.image_shape, device=device)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps
        iterator = tqdm(range(n_steps), desc="Sampling") if verbose else range(n_steps)
        
        for i in iterator:
            t = torch.full((n_samples,), i * dt, device=device)
            
            # Predict velocity
            v = self.backbone(x, t, conditioning, param_conditioning)
            
            # Euler step
            x = x + dt * v
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return x, trajectory
        return x