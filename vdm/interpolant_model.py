"""
Stochastic Interpolant Model for VDM-BIND.

This module implements flow matching / stochastic interpolants for the DMO -> Hydro mapping,
following the approach in BaryonBridge (Sadr et al.).

Key concepts:
- Flow matching learns a velocity field v(t, x) that transports x_0 -> x_1
- Stochastic Interpolant formulation (BaryonBridge):
  - x_t = alpha(t)*x_0 + beta(t)*x_1 + sqrt(t)*sigma(t)*epsilon
  - where alpha(t)=1-t, beta(t)=t^2, sigma(t)=1-t
  - Target velocity r_t = alpha_dot*x_0 + beta_dot*x_1 + sigma_dot*sqrt(t)*epsilon
- Deterministic Interpolant (linear flow matching):
  - x_t = (1-t)*x_0 + t*x_1 (linear interpolation)
  - Velocity: v_t = x_1 - x_0 (constant velocity)
  - Sampling: Integrate ODE dx/dt = v(t, x) from t=0 to t=1

Integration with VDM-BIND:
- Uses same AstroDataset and normalization as VDM/DDPM
- Compatible with existing UNet architecture (with time embedding)
- Supports multi-scale conditioning (DM + large-scale context)
- Works with BIND inference pipeline

Usage:
    from vdm.interpolant_model import LightInterpolant
    
    # Create model
    model = LightInterpolant(
        velocity_model=unet,
        learning_rate=1e-4,
        n_sampling_steps=50,
    )
    
    # Training
    loss = model.training_step(batch, batch_idx)
    
    # Sampling (BIND-compatible)
    samples = model.draw_samples(conditioning, batch_size=8)
"""

import math
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Union, List
from tqdm import tqdm

from lightning.pytorch import LightningModule


# ============================================================================
# Interpolant Core
# ============================================================================

class Interpolant(nn.Module):
    """
    Stochastic Interpolant for flow matching.
    
    Implements the BaryonBridge stochastic interpolant formulation where we learn
    a velocity field that transports from x_0 to x_1 via a stochastic path.
    
    For stochastic interpolants (BaryonBridge formulation):
        x_t = alpha(t)*x_0 + beta(t)*x_1 + sqrt(t)*sigma(t)*epsilon
        where alpha(t)=1-t, beta(t)=t^2, sigma(t)=1-t
        
        Target velocity: r_t = alpha_dot*x_0 + beta_dot*x_1 + sigma_dot*sqrt(t)*epsilon
        where alpha_dot=-1, beta_dot=2t, sigma_dot=-1
    
    For deterministic interpolants (linear flow matching):
        x_t = (1-t)*x_0 + t*x_1
        Target velocity: v = x_1 - x_0
    
    Args:
        velocity_model: Neural network that predicts velocity field
        use_stochastic_interpolant: Whether to use stochastic interpolant (BaryonBridge)
        sigma: Base noise scale (only for deterministic mode, stochastic uses 1-t)
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        use_stochastic_interpolant: bool = False,
        sigma: float = 0.0,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.use_stochastic_interpolant = use_stochastic_interpolant
        self.sigma = sigma
    
    # =========================================================================
    # BaryonBridge Stochastic Interpolant functions (when use_stochastic=True)
    # =========================================================================
    
    def get_alpha_t(self, t: Tensor) -> Tensor:
        """alpha(t) = 1 - t"""
        return 1 - t
    
    def get_beta_t(self, t: Tensor) -> Tensor:
        """beta(t) = t^2"""
        return t ** 2
    
    def get_sigma_t_stochastic(self, t: Tensor) -> Tensor:
        """sigma(t) = 1 - t for stochastic interpolant"""
        return 1 - t
    
    def get_alpha_t_dot(self, t: Tensor) -> Tensor:
        """d/dt[alpha(t)] = -1"""
        return -torch.ones_like(t)
    
    def get_beta_t_dot(self, t: Tensor) -> Tensor:
        """d/dt[beta(t)] = 2t"""
        return 2.0 * t
    
    def get_sigma_t_dot(self, t: Tensor) -> Tensor:
        """d/dt[sigma(t)] = -1"""
        return -torch.ones_like(t)
    
    def get_xt_stochastic(self, x0: Tensor, x1: Tensor, t: Tensor, epsilon: Tensor) -> Tensor:
        """
        Compute x_t for stochastic interpolant (BaryonBridge formulation).
        
        x_t = alpha(t)*x_0 + beta(t)*x_1 + sqrt(t)*sigma(t)*epsilon
        """
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        return (
            self.get_alpha_t(t) * x0
            + self.get_beta_t(t) * x1
            + torch.sqrt(t) * self.get_sigma_t_stochastic(t) * epsilon
        )
    
    def get_rt(self, x0: Tensor, x1: Tensor, t: Tensor, epsilon: Tensor) -> Tensor:
        """
        Compute target velocity r_t for stochastic interpolant (BaryonBridge formulation).
        
        r_t = alpha_dot(t)*x_0 + beta_dot(t)*x_1 + sigma_dot(t)*sqrt(t)*epsilon
        """
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        return (
            self.get_alpha_t_dot(t) * x0
            + self.get_beta_t_dot(t) * x1
            + self.get_sigma_t_dot(t) * torch.sqrt(t) * epsilon
        )
    
    # =========================================================================
    # Deterministic (Linear) Interpolant functions (when use_stochastic=False)
    # =========================================================================
    
    def get_mu_t(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Compute mean of interpolant at time t (linear interpolation).
        
        Linear interpolation: mu_t = t * x_1 + (1 - t) * x_0
        """
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        return t * x1 + (1 - t) * x0
    
    def get_sigma_t(self, t: Tensor) -> Tensor:
        """
        Compute noise scale at time t for deterministic interpolant with optional noise.
        
        sigma(t) = sigma * sqrt(2 * t * (1 - t))
        This is 0 at t=0 and t=1, maximal at t=0.5.
        """
        t = t.view(t.shape[0], *([1] * 3))
        return self.sigma * torch.sqrt(2 * t * (1 - t))
    
    def sample_xt(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Sample x_t from the interpolant distribution.
        
        For stochastic (BaryonBridge): x_t = alpha(t)*x_0 + beta(t)*x_1 + sqrt(t)*sigma(t)*eps
        For deterministic: x_t = (1-t)*x_0 + t*x_1 (optionally + sigma*noise)
        
        Returns:
            x_t: Interpolated sample (B, C, H, W)
            epsilon: The noise used (B, C, H, W) if stochastic, else None
        """
        if self.use_stochastic_interpolant:
            # BaryonBridge stochastic interpolant
            epsilon = torch.randn_like(x0)
            x_t = self.get_xt_stochastic(x0, x1, t, epsilon)
            return x_t, epsilon
        else:
            # Deterministic linear interpolant
            mu_t = self.get_mu_t(x0, x1, t)
            
            if self.sigma > 0:
                sigma_t = self.get_sigma_t(t)
                epsilon = torch.randn_like(mu_t)
                return mu_t + sigma_t * epsilon, epsilon
            
            return mu_t, None
    
    def get_velocity(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        Compute the true velocity field for deterministic interpolant.
        
        For linear interpolant: v = x_1 - x_0 (constant velocity)
        """
        return x1 - x0
    
    def compute_loss(
        self,
        x0: Tensor,
        x1: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the flow matching loss.
        
        For stochastic interpolants (BaryonBridge):
            Loss = E_t E_{x_0, x_1, epsilon} || v_theta(t, x_t, cond) - r_t ||^2
            where r_t = alpha_dot*x_0 + beta_dot*x_1 + sigma_dot*sqrt(t)*epsilon
        
        For deterministic interpolants:
            Loss = E_t E_{x_0, x_1} || v_theta(t, x_t, cond) - (x_1 - x_0) ||^2
        
        Args:
            x0: Source (B, C, H, W)
            x1: Target (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            param_conditioning: Parameter conditioning (B, N_params)
            t: Optional time (B,). If None, uniformly sampled.
        
        Returns:
            loss: Scalar loss value
        """
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        
        # Sample x_t from interpolant (also returns epsilon if stochastic)
        x_t, epsilon = self.sample_xt(x0, x1, t)
        
        # Compute true velocity target
        if self.use_stochastic_interpolant:
            # BaryonBridge: target is r_t (derivative of x_t)
            v_true = self.get_rt(x0, x1, t, epsilon)
        else:
            # Deterministic: target is x_1 - x_0
            v_true = self.get_velocity(x0, x1)
        
        # Predict velocity
        v_pred = self.velocity_model(t, x_t, conditioning, param_conditioning)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_true)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        x0: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
        n_steps: int = 50,
        return_trajectory: bool = False,
        stochastic: bool = None,
        sde_noise_scale: float = None,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Generate samples by integrating the ODE/SDE.
        
        For stochastic interpolants (BaryonBridge), uses Euler-Maruyama with
        diffusion coefficient sigma(t) = 1 - t.
        
        For deterministic interpolants, uses Euler integration.
        
        Args:
            x0: Initial state (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            param_conditioning: Parameter conditioning (B, N_params)
            n_steps: Number of integration steps
            return_trajectory: Whether to return full trajectory
            stochastic: Whether to use SDE sampling. If None, uses self.use_stochastic_interpolant
            sde_noise_scale: Noise scale for SDE. If None, uses 1.0 for stochastic mode
        
        Returns:
            x1: Final sample (B, C, H, W), or trajectory list
        """
        if stochastic is None:
            stochastic = self.use_stochastic_interpolant
        if sde_noise_scale is None:
            sde_noise_scale = 1.0  # Default for BaryonBridge-style SDE
        
        dt = 1.0 / n_steps
        x = x0.clone()
        
        trajectory = [x.clone()] if return_trajectory else None
        
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
            
            # Predict velocity
            v = self.velocity_model(t, x, conditioning, param_conditioning)
            
            # Euler step (deterministic part)
            x = x + v * dt
            
            # Add noise for SDE sampling (Euler-Maruyama)
            # For BaryonBridge: diffusion = sigma(t) = 1 - t
            # dW ~ sqrt(dt) * N(0,1)
            if stochastic and i < n_steps - 1:  # Don't add noise on last step
                if self.use_stochastic_interpolant:
                    # BaryonBridge diffusion: sigma(t) = 1 - t
                    diffusion = (1 - t_val) * sde_noise_scale
                else:
                    # Custom diffusion with decay
                    diffusion = sde_noise_scale * (1.0 - (i + 1) * dt)
                
                noise_scale = diffusion * math.sqrt(dt)
                x = x + noise_scale * torch.randn_like(x)
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return trajectory
        
        return x


# ============================================================================
# Velocity Network Wrapper
# ============================================================================

class VelocityNetWrapper(nn.Module):
    """
    Wrapper around UNet to adapt it for velocity prediction.
    
    Takes the existing UNet (designed for noise prediction) and adapts it
    for velocity prediction in the interpolant framework.
    
    The network signature is: v = net(t, x_t, conditioning, param_conditioning)
    
    Args:
        net: Neural network (e.g., UNet from networks_clean.py)
        output_channels: Number of output channels (3 for [DM, Gas, Stars])
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
        t: Tensor,
        x: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass to predict velocity.
        
        Args:
            t: Time (B,) in [0, 1]
            x: Current state x_t (B, C_out, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            param_conditioning: Parameter conditioning (B, N_params) - cosmological params
        
        Returns:
            v: Predicted velocity (B, C_out, H, W)
        """
        # UNetVDM.forward expects: forward(z, g_t, conditioning, param_conditioning)
        # - z: target tensor (x here, which is x_t in interpolant notation)
        # - g_t: time (we use t directly)
        # - conditioning: spatial conditioning (DM + large-scale maps)
        # - param_conditioning: cosmological parameters for FiLM conditioning
        # 
        # UNetVDM internally handles:
        # 1. Fourier features on conditioning (if enabled)
        # 2. Concatenation of z with conditioning (or cross-attention)
        # So we pass them separately - do NOT concatenate here!
        output = self.net(x, t, conditioning=conditioning, param_conditioning=param_conditioning)
        
        # UNetVDM may return a tuple when param_prediction or auxiliary_mask is enabled:
        # - (prediction,) or prediction  -- just the velocity/noise prediction
        # - (prediction, predicted_params) -- with param prediction
        # - (prediction, mask_logits) -- with auxiliary mask
        # - (prediction, predicted_params, mask_logits) -- with both
        # We only need the velocity prediction (first element)
        if isinstance(output, tuple):
            return output[0]
        return output


# ============================================================================
# Lightning Module
# ============================================================================

class LightInterpolant(LightningModule):
    """
    PyTorch Lightning wrapper for Stochastic Interpolant.
    
    Provides:
    - Training with flow matching loss
    - Validation with loss computation
    - Sampling interface compatible with BIND pipeline
    - Checkpointing and logging
    
    Args:
        velocity_model: Neural network that predicts velocity
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        lr_scheduler: Learning rate scheduler type
        n_sampling_steps: Number of steps for sampling
        use_stochastic_interpolant: Use stochastic interpolant
        sigma: Noise scale for stochastic interpolant
        use_param_conditioning: Whether to include astro params
        x0_mode: How to initialize x0 ('zeros', 'noise', 'dm_copy')
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        # Sampling parameters
        n_sampling_steps: int = 50,
        # Interpolant parameters
        use_stochastic_interpolant: bool = False,
        sigma: float = 0.0,
        # Conditioning
        use_param_conditioning: bool = False,
        # x0 initialization
        x0_mode: str = "zeros",  # 'zeros', 'noise', 'dm_copy'
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.n_sampling_steps = n_sampling_steps
        self.use_param_conditioning = use_param_conditioning
        self.x0_mode = x0_mode
        
        # Create interpolant
        self.interpolant = Interpolant(
            velocity_model=velocity_model,
            use_stochastic_interpolant=use_stochastic_interpolant,
            sigma=sigma,
        )
        
        # Store velocity model reference for convenience
        self.velocity_model = velocity_model
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['velocity_model'])
        
        print(f"\n{'='*60}")
        print(f"INITIALIZED LIGHT INTERPOLANT MODEL")
        print(f"{'='*60}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Sampling steps: {n_sampling_steps}")
        print(f"  Stochastic: {use_stochastic_interpolant} (sigma={sigma})")
        print(f"  x0 mode: {x0_mode}")
        print(f"  Param conditioning: {use_param_conditioning}")
        print(f"{'='*60}\n")
    
    def _get_x0(self, x1: Tensor, dm_condition: Optional[Tensor] = None) -> Tensor:
        """
        Initialize x0 based on the configured mode.
        
        Args:
            x1: Target tensor (B, C, H, W) - used for shape
            dm_condition: DM condition (B, 1, H, W) - used if x0_mode='dm_copy'
        
        Returns:
            x0: Initialized source tensor (B, C, H, W)
        """
        B, C, H, W = x1.shape
        
        if self.x0_mode == "zeros":
            return torch.zeros_like(x1)
        elif self.x0_mode == "noise":
            return torch.randn_like(x1)
        elif self.x0_mode == "dm_copy":
            # Copy DM condition to all channels
            if dm_condition is not None:
                return dm_condition.expand(-1, C, -1, -1).clone()
            else:
                return torch.zeros_like(x1)
        else:
            raise ValueError(f"Unknown x0_mode: {self.x0_mode}")
    
    def _unpack_batch(self, batch: Tuple) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        Unpack batch from AstroDataset.
        
        AstroDataset returns: (m_dm, large_scale, m_target, conditions)
        
        Returns:
            x1: Target (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            dm_condition: DM condition only (B, 1, H, W)
            params: Cosmological parameters or None
        """
        m_dm, large_scale, m_target, conditions = batch
        
        # Target is what we want to generate
        x1 = m_target  # (B, 3, H, W)
        
        # Spatial conditioning: concatenate DM and large-scale context
        conditioning = torch.cat([m_dm, large_scale], dim=1)  # (B, 1+N, H, W)
        
        # DM condition alone for x0 initialization if needed
        dm_condition = m_dm  # (B, 1, H, W)
        
        # Parameters
        params = conditions if self.use_param_conditioning else None
        
        return x1, conditioning, dm_condition, params
    
    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Training step with flow matching loss."""
        x1, conditioning, dm_condition, params = self._unpack_batch(batch)
        
        # Initialize x0
        x0 = self._get_x0(x1, dm_condition)
        
        # Compute loss (pass param_conditioning for FiLM modulation)
        loss = self.interpolant.compute_loss(
            x0=x0,
            x1=x1,
            conditioning=conditioning,
            param_conditioning=params,
        )
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Validation step."""
        x1, conditioning, dm_condition, params = self._unpack_batch(batch)
        
        # Initialize x0
        x0 = self._get_x0(x1, dm_condition)
        
        # Compute loss (pass param_conditioning for FiLM modulation)
        loss = self.interpolant.compute_loss(
            x0=x0,
            x1=x1,
            conditioning=conditioning,
            param_conditioning=params,
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
        Generate samples using ODE integration.
        
        Args:
            shape: Shape of samples to generate (B, C, H, W)
            conditioning: Spatial conditioning tensor (B, C_cond, H, W)
                          First channel is DM condition, used for dm_copy mode
            param_conditioning: Parameter conditioning (B, N_params) - cosmological params
            steps: Number of sampling steps
            verbose: Show progress bar
        
        Returns:
            samples: Generated samples tensor
        """
        if steps is None:
            steps = self.n_sampling_steps
        
        # Initialize x0 based on mode
        device = conditioning.device if conditioning is not None else self.device
        B, C_out, H, W = shape
        
        if self.x0_mode == "noise":
            x0 = torch.randn(shape, device=device)
        elif self.x0_mode == "dm_copy":
            # Copy DM condition (first channel of conditioning) to all output channels
            # This matches training where x0 = dm_condition.expand(-1, C, -1, -1)
            if conditioning is not None:
                dm_condition = conditioning[:, :1]  # (B, 1, H, W)
                x0 = dm_condition.expand(-1, C_out, -1, -1).clone()
            else:
                x0 = torch.zeros(shape, device=device)
        else:  # zeros
            x0 = torch.zeros(shape, device=device)
        
        # Generate samples (pass param_conditioning for FiLM modulation)
        samples = self.interpolant.sample(
            x0=x0,
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
            conditioning: Spatial conditioning tensor (B, C_cond, H, W)
            batch_size: Number of samples to generate
            n_sampling_steps: Sampling steps
            param_conditioning: Parameter conditioning (B, N_params) - cosmological params
            verbose: Show progress
        
        Returns:
            samples: Generated samples (B, C_out, H, W)
        """
        B, C_cond, H, W = conditioning.shape
        C_out = 3  # [DM, Gas, Stars]
        
        # Generate samples (pass param_conditioning for FiLM modulation)
        samples = self.sample(
            shape=(batch_size, C_out, H, W),
            conditioning=conditioning,
            param_conditioning=param_conditioning,
            steps=n_sampling_steps or self.n_sampling_steps,
            verbose=verbose,
        )
        
        return samples
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
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
# Factory functions
# ============================================================================

def create_interpolant_model(
    output_channels: int = 3,
    conditioning_channels: int = 4,
    embedding_dim: int = 256,
    n_blocks: int = 32,
    norm_groups: int = 8,
    n_attention_heads: int = 8,
    learning_rate: float = 1e-4,
    n_sampling_steps: int = 50,
    use_fourier_features: bool = True,
    fourier_legacy: bool = False,
    add_attention: bool = True,
    x0_mode: str = "zeros",
    **kwargs,
) -> LightInterpolant:
    """
    Create an interpolant model using the VDM-BIND UNet architecture.
    
    Args:
        output_channels: Number of output channels (3 for [DM, Gas, Stars])
        conditioning_channels: Number of conditioning channels (4 = DM + 3 large-scale)
        embedding_dim: Embedding dimension for UNet
        n_blocks: Number of residual blocks
        norm_groups: Number of groups for GroupNorm
        n_attention_heads: Number of attention heads
        learning_rate: Learning rate
        n_sampling_steps: Sampling steps
        use_fourier_features: Use Fourier features for input
        fourier_legacy: Use legacy Fourier feature implementation
        add_attention: Add attention blocks
        x0_mode: How to initialize x0 ('zeros', 'noise', 'dm_copy')
    
    Returns:
        LightInterpolant instance
    """
    from vdm.networks_clean import UNet
    
    # Total input channels = output + conditioning
    input_channels = output_channels + conditioning_channels
    
    # Create UNet
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
    
    # Wrap for velocity prediction
    velocity_model = VelocityNetWrapper(
        net=unet,
        output_channels=output_channels,
        conditioning_channels=conditioning_channels,
    )
    
    return LightInterpolant(
        velocity_model=velocity_model,
        learning_rate=learning_rate,
        n_sampling_steps=n_sampling_steps,
        x0_mode=x0_mode,
        **kwargs,
    )


# ============================================================================
# Alias for consistency
# ============================================================================

LightFlowMatching = LightInterpolant
