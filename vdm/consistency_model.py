"""
Consistency Model for VDM-BIND.

.. deprecated:: 2.0.0
    This module is kept for backward compatibility with existing checkpoints.
    For new training, use `vdm.methods.ConsistencyMethod` instead:
    
    >>> from vdm.methods import create_method
    >>> model = create_method('consistency', backbone_type='unet-b', img_size=128)
    
    The new API provides a cleaner Method + Backbone abstraction and is the
    recommended approach going forward.

This module implements Consistency Models (Song et al., 2023) for the DMO -> Hydro mapping.

Key concepts:
- Consistency models learn to map any point on the diffusion trajectory directly to clean data
- f_θ(x_t, t) = x_0 for all t
- Self-consistency: f_θ(x_t, t) = f_θ(x_t', t') for points on same trajectory
- Can be trained via distillation (from pre-trained diffusion) or consistency training
- Enables single-step or few-step high-quality sampling

Advantages over diffusion:
- Single-step or few-step sampling (vs 250-1000 steps for diffusion)
- Maintains diffusion-quality results with proper training
- Can trade off compute vs quality (more steps = better quality)

Training modes:
1. Consistency Distillation (CD): Distill from pre-trained diffusion model
2. Consistency Training (CT): Train from scratch without teacher model
   - Uses discrete-time formulation for stability

Integration with VDM-BIND:
- Uses same AstroDataset and normalization as VDM/DDPM/Interpolant
- Compatible with existing UNet architecture
- Supports multi-scale conditioning (DM + large-scale context)
- Works with BIND inference pipeline

Reference:
    Song et al. (2023) "Consistency Models" https://arxiv.org/abs/2303.01469

Usage:
    from vdm.consistency_model import LightConsistency
    
    # Create model
    model = LightConsistency(
        consistency_model=unet,
        learning_rate=1e-4,
        n_sampling_steps=1,  # Single-step sampling!
    )
    
    # Training (consistency training mode)
    loss = model.training_step(batch, batch_idx)
    
    # Sampling (BIND-compatible)
    samples = model.draw_samples(conditioning, batch_size=8)
"""

import torch
import numpy as np
import math
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Union, List
from tqdm import tqdm

from lightning.pytorch import LightningModule


# ============================================================================
# Noise Schedule for Consistency Models
# ============================================================================

class ConsistencyNoiseSchedule:
    """
    Noise schedule for consistency models.
    
    Uses the variance-preserving (VP) formulation from Song et al.
    The noise schedule defines σ(t) such that:
        x_t = x_0 + σ(t) * ε
    
    For consistency models, we need:
        σ(ε) ≈ 0 (clean data at t=ε)
        σ(T) = σ_max (maximum noise at t=T)
    
    Args:
        sigma_min: Minimum noise level (at t=ε)
        sigma_max: Maximum noise level (at t=T)
        rho: Schedule shape parameter (default=7 from paper)
    """
    
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
    
    def get_sigma(self, t: Tensor) -> Tensor:
        """
        Get noise level at time t.
        
        σ(t) = (σ_min^(1/ρ) + t * (σ_max^(1/ρ) - σ_min^(1/ρ)))^ρ
        
        This gives a smooth schedule from σ_min to σ_max as t goes from 0 to 1.
        """
        sigma_min_inv_rho = self.sigma_min ** (1 / self.rho)
        sigma_max_inv_rho = self.sigma_max ** (1 / self.rho)
        
        sigma = (sigma_min_inv_rho + t * (sigma_max_inv_rho - sigma_min_inv_rho)) ** self.rho
        return sigma
    
    def get_t_from_sigma(self, sigma: Tensor) -> Tensor:
        """Inverse: get t from sigma."""
        sigma_min_inv_rho = self.sigma_min ** (1 / self.rho)
        sigma_max_inv_rho = self.sigma_max ** (1 / self.rho)
        
        t = (sigma ** (1 / self.rho) - sigma_min_inv_rho) / (sigma_max_inv_rho - sigma_min_inv_rho)
        return t
    
    def get_discretized_sigmas(self, n_steps: int, device: torch.device = None) -> Tensor:
        """
        Get discretized sigma values for n_steps.
        
        Returns sigmas in decreasing order: [σ_max, ..., σ_min]
        """
        # Create n_steps + 1 boundaries
        t_values = torch.linspace(1, 0, n_steps + 1, device=device)
        sigmas = self.get_sigma(t_values)
        return sigmas


# ============================================================================
# Consistency Function (Skip Connection Parameterization)
# ============================================================================

class ConsistencyFunction(nn.Module):
    """
    Consistency function with skip connection parameterization.
    
    The consistency function f_θ(x, σ) is parameterized as:
        f_θ(x, σ) = c_skip(σ) * x + c_out(σ) * F_θ(x, σ)
    
    Where:
        c_skip(σ) = σ_data² / (σ² + σ_data²)
        c_out(σ) = σ * σ_data / √(σ² + σ_data²)
    
    This ensures:
        f_θ(x, σ_min) ≈ x (boundary condition at σ ≈ 0)
    
    Args:
        net: Neural network F_θ that predicts the denoised estimate
        sigma_data: Standard deviation of training data (for scaling)
        sigma_min: Minimum sigma value
    """
    
    def __init__(
        self,
        net: nn.Module,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
    ):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
    
    def get_scalings(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute c_skip, c_out, and c_in scalings.
        
        Args:
            sigma: Noise level (B,) or (B, 1, 1, 1)
        
        Returns:
            c_skip: Skip connection weight
            c_out: Network output weight
            c_in: Network input scaling
        """
        # Ensure proper shape for broadcasting
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        
        sigma_data_sq = self.sigma_data ** 2
        
        c_skip = sigma_data_sq / (sigma ** 2 + sigma_data_sq)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + sigma_data_sq)
        c_in = 1 / torch.sqrt(sigma ** 2 + sigma_data_sq)
        
        return c_skip, c_out, c_in
    
    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply consistency function.
        
        f_θ(x, σ) = c_skip(σ) * x + c_out(σ) * F_θ(c_in(σ) * x, σ)
        
        Args:
            x: Noisy input (B, C, H, W)
            sigma: Noise level (B,)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            param_conditioning: Parameter conditioning (B, N_params)
        
        Returns:
            Denoised prediction (B, C, H, W)
        """
        c_skip, c_out, c_in = self.get_scalings(sigma)
        
        # Scale input
        x_scaled = c_in * x
        
        # Network prediction
        # Convert sigma to time-like value for network (expects values in [0, 1])
        # We use log scaling for better numerical stability
        t = torch.log(sigma.view(-1)) / 4  # Rough scaling
        t = torch.clamp(t, 0, 1)
        
        F_x = self.net(x_scaled, t, conditioning=conditioning, param_conditioning=param_conditioning)
        
        # Handle tuple output from UNet
        if isinstance(F_x, tuple):
            F_x = F_x[0]
        
        # Apply skip connection parameterization
        output = c_skip * x + c_out * F_x
        
        return output


# ============================================================================
# Consistency Model Core
# ============================================================================

class ConsistencyModel(nn.Module):
    """
    Core Consistency Model implementation.
    
    Implements consistency training (CT) from scratch.
    The key insight is to enforce self-consistency:
        f_θ(x_t, t) = f_θ-(x_t', t')
    
    where θ- is an EMA of θ (target network) and (x_t, x_t') are adjacent
    points on the same diffusion trajectory.
    
    Args:
        consistency_fn: Consistency function with skip connection
        noise_schedule: Noise schedule for diffusion
        sigma_data: Standard deviation of training data
    """
    
    def __init__(
        self,
        consistency_fn: ConsistencyFunction,
        noise_schedule: ConsistencyNoiseSchedule,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.consistency_fn = consistency_fn
        self.noise_schedule = noise_schedule
        self.sigma_data = sigma_data
    
    def add_noise(self, x: Tensor, sigma: Tensor) -> Tensor:
        """
        Add noise to clean data.
        
        x_t = x_0 + σ * ε
        """
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return x + sigma * noise
    
    def compute_ct_loss(
        self,
        x0: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
        target_model: Optional[nn.Module] = None,
        n_steps: int = 18,
    ) -> Tensor:
        """
        Compute Consistency Training (CT) loss.
        
        For each sample:
        1. Sample discrete time steps n ~ U{1, N-1}
        2. Compute σ_n and σ_{n+1}
        3. Add noise to get x_{σ_{n+1}}
        4. Denoise one step to get x_{σ_n} using ODE
        5. Loss = ||f_θ(x_{σ_{n+1}}, σ_{n+1}) - f_θ-(x_{σ_n}, σ_n)||²
        
        Args:
            x0: Clean data (B, C, H, W)
            conditioning: Spatial conditioning
            param_conditioning: Parameter conditioning
            target_model: EMA model for target (if None, use self)
            n_steps: Number of discretization steps
        
        Returns:
            loss: Consistency training loss
        """
        B = x0.shape[0]
        device = x0.device
        
        # Get discretized sigmas
        sigmas = self.noise_schedule.get_discretized_sigmas(n_steps, device=device)
        
        # Sample random step indices n ~ U{1, N-1}
        n = torch.randint(1, n_steps, (B,), device=device)
        
        # Get σ_n and σ_{n+1}
        sigma_n = sigmas[n]  # Current sigma
        sigma_n_plus_1 = sigmas[n - 1]  # Next sigma (higher noise)
        
        # Add noise at σ_{n+1}
        noise = torch.randn_like(x0)
        x_sigma_n_plus_1 = x0 + sigma_n_plus_1.view(-1, 1, 1, 1) * noise
        
        # One-step ODE to get x at σ_n (using the Euler method with the score)
        # For CT, we use a simpler approach: just add appropriate noise level
        # x_{σ_n} = x_0 + σ_n * ε (same noise direction)
        x_sigma_n = x0 + sigma_n.view(-1, 1, 1, 1) * noise
        
        # Get predictions from student and teacher
        pred_student = self.consistency_fn(
            x_sigma_n_plus_1, sigma_n_plus_1,
            conditioning=conditioning,
            param_conditioning=param_conditioning,
        )
        
        # Teacher prediction (use target model if provided, else self)
        if target_model is not None:
            with torch.no_grad():
                pred_target = target_model.consistency_fn(
                    x_sigma_n, sigma_n,
                    conditioning=conditioning,
                    param_conditioning=param_conditioning,
                )
        else:
            with torch.no_grad():
                pred_target = self.consistency_fn(
                    x_sigma_n, sigma_n,
                    conditioning=conditioning,
                    param_conditioning=param_conditioning,
                ).detach()
        
        # Consistency loss (L2)
        loss = F.mse_loss(pred_student, pred_target)
        
        return loss
    
    def compute_denoising_loss(
        self,
        x0: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute simple denoising loss for pre-training/warm-up.
        
        This is like standard diffusion training but helps bootstrap
        the consistency model before full CT training.
        
        Loss = E_σ E_ε || f_θ(x_0 + σε, σ) - x_0 ||²
        """
        B = x0.shape[0]
        device = x0.device
        
        # Sample sigma uniformly in log space
        log_sigma = torch.rand(B, device=device) * (
            math.log(self.noise_schedule.sigma_max) - 
            math.log(self.noise_schedule.sigma_min)
        ) + math.log(self.noise_schedule.sigma_min)
        sigma = torch.exp(log_sigma)
        
        # Add noise
        x_noisy = self.add_noise(x0, sigma)
        
        # Predict clean data
        x0_pred = self.consistency_fn(
            x_noisy, sigma,
            conditioning=conditioning,
            param_conditioning=param_conditioning,
        )
        
        # MSE loss
        loss = F.mse_loss(x0_pred, x0)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
        n_steps: int = 1,
        sigmas: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generate samples using consistency model.
        
        Single-step: Just apply f_θ(x_T, σ_max)
        Multi-step: Iteratively denoise and add noise
        
        Args:
            x_init: Initial noisy sample (B, C, H, W)
            conditioning: Spatial conditioning
            param_conditioning: Parameter conditioning
            n_steps: Number of sampling steps (1 for single-step)
            sigmas: Custom sigma schedule (optional)
        
        Returns:
            samples: Generated samples (B, C, H, W)
        """
        if sigmas is None:
            sigmas = self.noise_schedule.get_discretized_sigmas(n_steps, device=x_init.device)
        
        x = x_init
        
        for i in range(n_steps):
            sigma = sigmas[i]
            sigma_batch = torch.full((x.shape[0],), sigma, device=x.device)
            
            # Apply consistency function
            x = self.consistency_fn(
                x, sigma_batch,
                conditioning=conditioning,
                param_conditioning=param_conditioning,
            )
            
            # Add noise for next step (except last step)
            if i < n_steps - 1:
                sigma_next = sigmas[i + 1]
                noise = torch.randn_like(x)
                x = x + sigma_next * noise
        
        return x


# ============================================================================
# Consistency Net Wrapper
# ============================================================================

class ConsistencyNetWrapper(nn.Module):
    """
    Wrapper around UNet to adapt it for consistency model.
    
    Takes the existing UNet and adapts it for the consistency function.
    
    Args:
        net: Neural network (e.g., UNetVDM from networks_clean.py)
        output_channels: Number of output channels
        conditioning_channels: Number of conditioning channels
    """
    
    def __init__(
        self,
        net: nn.Module,
        output_channels: int = 3,
        conditioning_channels: int = 4,
    ):
        super().__init__()
        self.net = net
        self.output_channels = output_channels
        self.conditioning_channels = conditioning_channels
    
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for consistency prediction.
        
        Args:
            x: Noisy input (B, C, H, W)
            t: Time (B,) in [0, 1]
            conditioning: Spatial conditioning (B, C_cond, H, W)
            param_conditioning: Parameter conditioning (B, N_params)
        
        Returns:
            Prediction (B, C, H, W)
        """
        output = self.net(x, t, conditioning=conditioning, param_conditioning=param_conditioning)
        
        if isinstance(output, tuple):
            return output[0]
        return output


# ============================================================================
# Lightning Module
# ============================================================================

class LightConsistency(LightningModule):
    """
    PyTorch Lightning wrapper for Consistency Model.
    
    Provides:
    - Training with consistency training (CT) loss
    - Optional denoising pre-training
    - Single-step or multi-step sampling
    - EMA target network management
    - BIND-compatible sampling interface
    
    Args:
        consistency_model: The core consistency model
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        lr_scheduler: Learning rate scheduler type
        n_sampling_steps: Number of steps for sampling
        use_param_conditioning: Whether to use parameter conditioning
        x0_mode: How to initialize x0 ('zeros', 'noise', 'dm_copy')
        use_denoising_pretraining: Use denoising loss for warm-up
        denoising_warmup_epochs: Number of epochs for denoising warm-up
        ct_n_steps: Number of discretization steps for CT
        ema_decay: EMA decay for target network
    """
    
    def __init__(
        self,
        consistency_model: ConsistencyModel,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        # Sampling parameters
        n_sampling_steps: int = 1,
        # Conditioning
        use_param_conditioning: bool = False,
        # x0 initialization for training
        x0_mode: str = "zeros",
        # Training mode
        use_denoising_pretraining: bool = True,
        denoising_warmup_epochs: int = 10,
        ct_n_steps: int = 18,
        # EMA for target network
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.n_sampling_steps = n_sampling_steps
        self.use_param_conditioning = use_param_conditioning
        self.x0_mode = x0_mode
        self.use_denoising_pretraining = use_denoising_pretraining
        self.denoising_warmup_epochs = denoising_warmup_epochs
        self.ct_n_steps = ct_n_steps
        self.ema_decay = ema_decay
        
        # Main model
        self.consistency_model = consistency_model
        
        # Create EMA target model
        self.target_model = None  # Will be initialized in setup
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['consistency_model'])
        
        print(f"\n{'='*60}")
        print(f"INITIALIZED LIGHT CONSISTENCY MODEL")
        print(f"{'='*60}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Sampling steps: {n_sampling_steps}")
        print(f"  CT discretization steps: {ct_n_steps}")
        print(f"  x0 mode: {x0_mode}")
        print(f"  Param conditioning: {use_param_conditioning}")
        print(f"  Denoising pretraining: {use_denoising_pretraining} ({denoising_warmup_epochs} epochs)")
        print(f"  EMA decay: {ema_decay}")
        print(f"{'='*60}\n")
    
    def setup(self, stage: str):
        """Initialize target model as copy of main model."""
        if self.target_model is None:
            import copy
            self.target_model = copy.deepcopy(self.consistency_model)
            # Freeze target model
            for param in self.target_model.parameters():
                param.requires_grad = False
    
    def _update_target_model(self):
        """Update target model with EMA of main model."""
        with torch.no_grad():
            for param, target_param in zip(
                self.consistency_model.parameters(),
                self.target_model.parameters()
            ):
                target_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def _unpack_batch(self, batch: Tuple) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        Unpack batch from AstroDataset.
        
        Returns:
            x1: Target (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            dm_condition: DM condition only (B, 1, H, W)
            params: Cosmological parameters or None
        """
        m_dm, large_scale, m_target, conditions = batch
        
        x1 = m_target
        conditioning = torch.cat([m_dm, large_scale], dim=1)
        dm_condition = m_dm
        params = conditions if self.use_param_conditioning else None
        
        return x1, conditioning, dm_condition, params
    
    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Training step with CT or denoising loss."""
        x1, conditioning, dm_condition, params = self._unpack_batch(batch)
        
        # Determine training mode
        current_epoch = self.current_epoch
        use_denoising = (
            self.use_denoising_pretraining and 
            current_epoch < self.denoising_warmup_epochs
        )
        
        if use_denoising:
            # Denoising pre-training
            loss = self.consistency_model.compute_denoising_loss(
                x0=x1,
                conditioning=conditioning,
                param_conditioning=params,
            )
            self.log("train/denoising_loss", loss, on_step=True, on_epoch=True, 
                     prog_bar=True, sync_dist=True)
        else:
            # Consistency training
            loss = self.consistency_model.compute_ct_loss(
                x0=x1,
                conditioning=conditioning,
                param_conditioning=params,
                target_model=self.target_model,
                n_steps=self.ct_n_steps,
            )
            self.log("train/ct_loss", loss, on_step=True, on_epoch=True, 
                     prog_bar=True, sync_dist=True)
        
        # Update target model
        self._update_target_model()
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Validation step."""
        x1, conditioning, dm_condition, params = self._unpack_batch(batch)
        
        # Use CT loss for validation
        loss = self.consistency_model.compute_ct_loss(
            x0=x1,
            conditioning=conditioning,
            param_conditioning=params,
            target_model=self.target_model,
            n_steps=self.ct_n_steps,
        )
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
        steps: Optional[int] = None,
        verbose: bool = True,
    ) -> Tensor:
        """
        Generate samples using consistency model.
        
        Args:
            shape: Shape of samples (B, C, H, W)
            conditioning: Spatial conditioning
            param_conditioning: Parameter conditioning
            steps: Number of sampling steps (1 for single-step)
            verbose: Show progress
        
        Returns:
            samples: Generated samples
        """
        if steps is None:
            steps = self.n_sampling_steps
        
        device = conditioning.device if conditioning is not None else self.device
        
        # Initialize with noise at maximum sigma
        sigma_max = self.consistency_model.noise_schedule.sigma_max
        x = torch.randn(shape, device=device) * sigma_max
        
        # Apply consistency model
        samples = self.consistency_model.sample(
            x_init=x,
            conditioning=conditioning,
            param_conditioning=param_conditioning,
            n_steps=steps,
        )
        
        return samples
    
    def draw_samples(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: Optional[int] = None,
        param_conditioning: Optional[Tensor] = None,
        verbose: bool = False,
    ) -> Tensor:
        """
        BIND-compatible sampling interface.
        
        Args:
            conditioning: Spatial conditioning (B, C_cond, H, W)
            batch_size: Number of samples
            n_sampling_steps: Sampling steps
            param_conditioning: Parameter conditioning
            verbose: Show progress
        
        Returns:
            samples: Generated samples (B, C_out, H, W)
        """
        B, C_cond, H, W = conditioning.shape
        C_out = 3
        
        samples = self.sample(
            shape=(batch_size, C_out, H, W),
            conditioning=conditioning,
            param_conditioning=param_conditioning,
            steps=n_sampling_steps or self.n_sampling_steps,
            verbose=verbose,
        )
        
        return samples
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        if self.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=self.learning_rate * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        elif self.lr_scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=25,
                factor=0.5,
                mode="min",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val/loss",
                }
            }
        else:
            return optimizer


# ============================================================================
# Factory Functions
# ============================================================================

def create_consistency_model(
    output_channels: int = 3,
    conditioning_channels: int = 4,
    embedding_dim: int = 256,
    n_blocks: int = 32,
    norm_groups: int = 8,
    n_attention_heads: int = 8,
    learning_rate: float = 1e-4,
    n_sampling_steps: int = 1,
    use_fourier_features: bool = True,
    fourier_legacy: bool = False,
    add_attention: bool = True,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    **kwargs,
) -> LightConsistency:
    """
    Create a consistency model using the VDM-BIND UNet architecture.
    
    Args:
        output_channels: Number of output channels (3)
        conditioning_channels: Number of conditioning channels (4)
        embedding_dim: Embedding dimension
        n_blocks: Number of residual blocks
        norm_groups: Number of groups for GroupNorm
        n_attention_heads: Number of attention heads
        learning_rate: Learning rate
        n_sampling_steps: Sampling steps (1 for single-step)
        use_fourier_features: Use Fourier features
        fourier_legacy: Use legacy Fourier features
        add_attention: Add attention blocks
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        sigma_data: Data standard deviation
    
    Returns:
        LightConsistency instance
    """
    from vdm.networks_clean import UNet
    
    input_channels = output_channels + conditioning_channels
    
    unet = UNet(
        in_channels=input_channels,
        out_channels=output_channels,
        embedding_dim=embedding_dim,
        n_blocks=n_blocks,
        norm_groups=norm_groups,
        n_attention_heads=n_attention_heads,
        use_fourier_features=use_fourier_features,
        fourier_legacy=fourier_legacy,
        add_attention=add_attention,
    )
    
    # Wrap UNet
    net_wrapper = ConsistencyNetWrapper(
        net=unet,
        output_channels=output_channels,
        conditioning_channels=conditioning_channels,
    )
    
    # Create noise schedule
    noise_schedule = ConsistencyNoiseSchedule(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    
    # Create consistency function
    consistency_fn = ConsistencyFunction(
        net=net_wrapper,
        sigma_data=sigma_data,
        sigma_min=sigma_min,
    )
    
    # Create consistency model
    consistency_model = ConsistencyModel(
        consistency_fn=consistency_fn,
        noise_schedule=noise_schedule,
        sigma_data=sigma_data,
    )
    
    return LightConsistency(
        consistency_model=consistency_model,
        learning_rate=learning_rate,
        n_sampling_steps=n_sampling_steps,
        **kwargs,
    )
