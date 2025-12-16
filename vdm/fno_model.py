"""
PyTorch Lightning module for FNO-based generative model training.

This module wraps the FNO architecture and can be used with multiple
generative modeling approaches:
- VDM (Variational Diffusion Model)
- Flow Matching / Stochastic Interpolants
- Denoising Score Matching

The key insight is that all these methods need a backbone network that:
1. Takes noisy/interpolated data x_t
2. Takes a time/noise level indicator t
3. Takes conditioning (spatial + optional parameters)
4. Outputs a prediction (noise, velocity, or score)

FNO is well-suited for astrophysical data because:
- Global receptive field captures large-scale correlations
- Spectral learning may better preserve power spectrum statistics
- Resolution-invariant architecture

Usage:
    from vdm.fno_model import LightFNOVDM
    
    # Create model (VDM-style training)
    model = LightFNOVDM(
        fno_variant='FNO-B',
        img_size=128,
        learning_rate=1e-4,
        n_sampling_steps=256,
    )
    
    # Or with custom FNO
    from vdm.fno import FNO2d
    fno = FNO2d(hidden_channels=128, n_layers=8, ...)
    model = LightFNOVDM(fno_model=fno, ...)
"""

from typing import Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from lightning.pytorch import LightningModule

from .fno import FNO2d, create_fno_model


class LightFNOVDM(LightningModule):
    """
    Lightning module for training FNO with VDM loss.
    
    Uses the Variational Diffusion Model training objective with
    FNO as the backbone instead of UNet or DiT.
    
    Args:
        fno_model: FNO model or variant string ('FNO-S', 'FNO-B', etc.)
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization
        gamma_min: Minimum log SNR (gamma) value
        gamma_max: Maximum log SNR (gamma) value
        n_sampling_steps: Number of steps for sampling
        loss_type: 'mse', 'l1', or 'huber'
        lr_scheduler: Type of LR scheduler
        warmup_steps: Number of warmup steps for scheduler
        image_shape: Shape of output images (C, H, W)
        
        # FNO-specific args (if fno_model is string)
        img_size: Image size
        hidden_channels: FNO hidden dimension
        n_layers: Number of FNO blocks
        modes: Number of Fourier modes
        n_params: Number of conditioning parameters
        conditioning_channels: Number of DM conditioning channels
        large_scale_channels: Number of large-scale context channels
    """
    
    def __init__(
        self,
        fno_model: Union[FNO2d, str] = 'FNO-B',
        # Training params
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        n_sampling_steps: int = 256,
        loss_type: str = 'mse',
        lr_scheduler: str = 'cosine',
        warmup_steps: int = 1000,
        image_shape: Tuple[int, int, int] = (3, 64, 64),
        # FNO params (used if fno_model is string)
        img_size: int = 64,
        hidden_channels: int = 64,
        n_layers: int = 6,
        modes: int = 16,
        n_params: int = 35,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        param_min: Optional[List[float]] = None,
        param_max: Optional[List[float]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Build or use provided model
        if isinstance(fno_model, str):
            self.score_model = create_fno_model(
                variant=fno_model,
                img_size=img_size,
                n_params=n_params,
                conditioning_channels=conditioning_channels,
                large_scale_channels=large_scale_channels,
                param_min=param_min,
                param_max=param_max,
                dropout=dropout,
            )
        else:
            self.score_model = fno_model
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.n_sampling_steps = n_sampling_steps
        self.loss_type = loss_type
        self.lr_scheduler_type = lr_scheduler
        self.warmup_steps = warmup_steps
        self.image_shape = image_shape
        
        # Derived quantities
        self.n_channels = image_shape[0]
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['fno_model'])
        
        print(f"\n{'='*60}")
        print(f"INITIALIZED LightFNOVDM")
        print(f"{'='*60}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gamma range: [{gamma_min}, {gamma_max}]")
        print(f"  Sampling steps: {n_sampling_steps}")
        print(f"  Loss type: {loss_type}")
        print(f"  Image shape: {image_shape}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # Noise Schedule (same as VDM)
    # =========================================================================
    
    def gamma(self, t: Tensor) -> Tensor:
        """
        Compute gamma(t) = log(SNR(t)) using linear schedule in log-space.
        
        VDM Convention:
        - t=0: high gamma (clean, alpha≈1, sigma≈0)
        - t=1: low gamma (noisy, alpha≈0, sigma≈1)
        """
        return self.gamma_max - (self.gamma_max - self.gamma_min) * t
    
    def sigma_and_alpha(self, gamma: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute sigma and alpha from gamma."""
        sigmoid_gamma = torch.sigmoid(gamma)
        alpha = sigmoid_gamma.sqrt()
        sigma = (1 - sigmoid_gamma).sqrt()
        return sigma, alpha
    
    def t_to_sigma_alpha(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Convenience: get sigma, alpha directly from t."""
        gamma_t = self.gamma(t)
        return self.sigma_and_alpha(gamma_t)
    
    # =========================================================================
    # Forward Diffusion
    # =========================================================================
    
    def q_sample(self, x: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Sample from q(z_t | x) - the forward diffusion process.
        
        z_t = alpha_t * x + sigma_t * eps
        
        Args:
            x: (B, C, H, W) clean data
            t: (B,) time values in [0, 1]
            noise: Optional pre-sampled noise
        
        Returns:
            z_t: (B, C, H, W) noisy data
            noise: (B, C, H, W) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        sigma, alpha = self.t_to_sigma_alpha(t)
        
        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        sigma = sigma.view(-1, 1, 1, 1)
        alpha = alpha.view(-1, 1, 1, 1)
        
        z_t = alpha * x + sigma * noise
        
        return z_t, noise
    
    # =========================================================================
    # Loss Computation
    # =========================================================================
    
    def compute_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute VDM training loss.
        
        Args:
            x: (B, C, H, W) clean target data
            conditioning: (B, cond_channels, H, W) spatial conditioning
            param_conditioning: (B, n_params) optional parameter conditioning
        
        Returns:
            Dict with 'loss' and component losses
        """
        B = x.shape[0]
        device = x.device
        
        # Sample random times
        t = torch.rand(B, device=device)
        
        # Sample noise and create noisy data
        noise = torch.randn_like(x)
        z_t, _ = self.q_sample(x, t, noise)
        
        # Get gamma for the score model
        gamma_t = self.gamma(t)
        
        # Predict noise (using gamma, not t, for VDM compatibility)
        # FNO expects t, so we pass t directly
        eps_pred = self.score_model(t, z_t, conditioning, param_conditioning)
        
        # Compute loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(eps_pred, noise)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(eps_pred, noise)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(eps_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return {
            'loss': loss,
            'diffusion_loss': loss,
        }
    
    # =========================================================================
    # Training Step
    # =========================================================================
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        condition, large_scale, target, params = batch
        
        # Combine conditioning
        conditioning = torch.cat([condition, large_scale], dim=1)
        
        # Compute loss
        losses = self.compute_loss(target, conditioning, params)
        
        # Log
        self.log('train/loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        condition, large_scale, target, params = batch
        
        # Combine conditioning
        conditioning = torch.cat([condition, large_scale], dim=1)
        
        # Compute loss
        losses = self.compute_loss(target, conditioning, params)
        
        # Log
        self.log('val/loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True)
        
        return losses['loss']
    
    # =========================================================================
    # Sampling (DDPM-style)
    # =========================================================================
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        batch_size: int = 1,
        param_conditioning: Optional[Tensor] = None,
        n_steps: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Sample from the model using DDPM-style ancestral sampling.
        
        Args:
            conditioning: (B, cond_channels, H, W) spatial conditioning
            batch_size: Number of samples to generate
            param_conditioning: (B, n_params) optional parameters
            n_steps: Number of sampling steps (default: self.n_sampling_steps)
            return_intermediates: Whether to return intermediate samples
        
        Returns:
            samples: (B, C, H, W) generated samples
            intermediates: List of intermediate samples (if return_intermediates)
        """
        device = conditioning.device
        n_steps = n_steps or self.n_sampling_steps
        
        # Get spatial dimensions from conditioning
        H, W = conditioning.shape[-2:]
        
        # Start from pure noise
        z = torch.randn(batch_size, self.n_channels, H, W, device=device)
        
        # Time steps: from t=1 (noisy) to t=0 (clean)
        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)
        
        intermediates = []
        
        for i in range(n_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            # Current and next noise levels
            t_batch = t.expand(batch_size)
            sigma_t, alpha_t = self.t_to_sigma_alpha(t_batch)
            sigma_next, alpha_next = self.t_to_sigma_alpha(t_next.expand(batch_size))
            
            # Predict noise
            eps_pred = self.score_model(t_batch, z, conditioning, param_conditioning)
            
            # DDPM update (simplified)
            # Predict x_0 from current z_t and predicted noise
            sigma_t = sigma_t.view(-1, 1, 1, 1)
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            sigma_next = sigma_next.view(-1, 1, 1, 1)
            alpha_next = alpha_next.view(-1, 1, 1, 1)
            
            # x_0 prediction
            x0_pred = (z - sigma_t * eps_pred) / alpha_t.clamp(min=1e-8)
            
            # Compute z_{t-1} (DDPM posterior mean)
            if t_next > 0:
                # Add noise for stochastic sampling
                noise = torch.randn_like(z)
                z = alpha_next * x0_pred + sigma_next * noise
            else:
                # Final step: no noise
                z = x0_pred
            
            if return_intermediates and i % (n_steps // 10 + 1) == 0:
                intermediates.append(z.clone())
        
        if return_intermediates:
            return z, intermediates
        return z
    
    # Alias for BIND compatibility
    def draw_samples(
        self,
        conditioning: Tensor,
        batch_size: int = 1,
        conditional_params: Optional[np.ndarray] = None,
        n_sampling_steps: Optional[int] = None,
    ) -> Tensor:
        """BIND-compatible sampling interface."""
        device = conditioning.device
        
        # Convert numpy params to tensor
        param_conditioning = None
        if conditional_params is not None:
            param_conditioning = torch.from_numpy(conditional_params).float().to(device)
            if param_conditioning.dim() == 1:
                param_conditioning = param_conditioning.unsqueeze(0)
            if param_conditioning.shape[0] == 1 and batch_size > 1:
                param_conditioning = param_conditioning.expand(batch_size, -1)
        
        return self.sample(
            conditioning=conditioning,
            batch_size=batch_size,
            param_conditioning=param_conditioning,
            n_steps=n_sampling_steps,
        )
    
    # =========================================================================
    # Optimizer & Scheduler
    # =========================================================================
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        if self.lr_scheduler_type == 'cosine':
            # Cosine annealing with warmup
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / max(1, self.warmup_steps)
                else:
                    progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                },
            }
        
        elif self.lr_scheduler_type == 'constant':
            return optimizer
        
        else:
            return optimizer


class LightFNOFlow(LightningModule):
    """
    Lightning module for training FNO with Flow Matching loss.
    
    This variant uses flow matching (velocity prediction) instead of
    the VDM noise prediction objective. Flow matching often requires
    fewer sampling steps and has a simpler loss function.
    
    Loss: E_t ||v_theta(t, x_t) - (x_1 - x_0)||^2
    
    where x_t = t * x_1 + (1-t) * x_0 (linear interpolation)
    """
    
    def __init__(
        self,
        fno_model: Union[FNO2d, str] = 'FNO-B',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        n_sampling_steps: int = 50,
        lr_scheduler: str = 'cosine',
        warmup_steps: int = 1000,
        image_shape: Tuple[int, int, int] = (3, 64, 64),
        x0_mode: str = 'zeros',  # 'zeros', 'noise', or 'condition'
        # FNO params (if string)
        img_size: int = 64,
        n_params: int = 35,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        param_min: Optional[List[float]] = None,
        param_max: Optional[List[float]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Build or use provided model
        if isinstance(fno_model, str):
            self.velocity_model = create_fno_model(
                variant=fno_model,
                img_size=img_size,
                n_params=n_params,
                conditioning_channels=conditioning_channels,
                large_scale_channels=large_scale_channels,
                param_min=param_min,
                param_max=param_max,
                dropout=dropout,
            )
        else:
            self.velocity_model = fno_model
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_sampling_steps = n_sampling_steps
        self.lr_scheduler_type = lr_scheduler
        self.warmup_steps = warmup_steps
        self.image_shape = image_shape
        self.x0_mode = x0_mode
        self.n_channels = image_shape[0]
        
        self.save_hyperparameters(ignore=['fno_model'])
        
        print(f"\n{'='*60}")
        print(f"INITIALIZED LightFNOFlow (Flow Matching)")
        print(f"{'='*60}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Sampling steps: {n_sampling_steps}")
        print(f"  x0 mode: {x0_mode}")
        print(f"  Image shape: {image_shape}")
        print(f"{'='*60}\n")
    
    def get_x0(self, x1: Tensor, conditioning: Tensor) -> Tensor:
        """Get starting distribution sample."""
        if self.x0_mode == 'zeros':
            return torch.zeros_like(x1)
        elif self.x0_mode == 'noise':
            return torch.randn_like(x1)
        elif self.x0_mode == 'condition':
            # Use DM channel as starting point
            return conditioning[:, :self.n_channels]
        else:
            return torch.zeros_like(x1)
    
    def training_step(self, batch, batch_idx):
        """Training step with flow matching loss."""
        condition, large_scale, target, params = batch
        conditioning = torch.cat([condition, large_scale], dim=1)
        
        B = target.shape[0]
        device = target.device
        
        # Sample time uniformly
        t = torch.rand(B, device=device)
        
        # Get x0 and x1
        x0 = self.get_x0(target, conditioning)
        x1 = target
        
        # Linear interpolation: x_t = t * x1 + (1-t) * x0
        t_view = t.view(-1, 1, 1, 1)
        x_t = t_view * x1 + (1 - t_view) * x0
        
        # True velocity
        v_true = x1 - x0
        
        # Predicted velocity
        v_pred = self.velocity_model(t, x_t, conditioning, params)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_true)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        condition, large_scale, target, params = batch
        conditioning = torch.cat([condition, large_scale], dim=1)
        
        B = target.shape[0]
        device = target.device
        
        t = torch.rand(B, device=device)
        x0 = self.get_x0(target, conditioning)
        x1 = target
        
        t_view = t.view(-1, 1, 1, 1)
        x_t = t_view * x1 + (1 - t_view) * x0
        v_true = x1 - x0
        v_pred = self.velocity_model(t, x_t, conditioning, params)
        
        loss = F.mse_loss(v_pred, v_true)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        batch_size: int = 1,
        param_conditioning: Optional[Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> Tensor:
        """Sample using ODE integration (Euler method)."""
        device = conditioning.device
        n_steps = n_steps or self.n_sampling_steps
        H, W = conditioning.shape[-2:]
        
        # Start from x0
        if self.x0_mode == 'zeros':
            x = torch.zeros(batch_size, self.n_channels, H, W, device=device)
        elif self.x0_mode == 'noise':
            x = torch.randn(batch_size, self.n_channels, H, W, device=device)
        else:
            x = conditioning[:, :self.n_channels].expand(batch_size, -1, -1, -1).clone()
        
        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps, device=device)
            v = self.velocity_model(t, x, conditioning, param_conditioning)
            x = x + v * dt
        
        return x
    
    def draw_samples(
        self,
        conditioning: Tensor,
        batch_size: int = 1,
        conditional_params: Optional[np.ndarray] = None,
        n_sampling_steps: Optional[int] = None,
    ) -> Tensor:
        """BIND-compatible sampling interface."""
        device = conditioning.device
        
        param_conditioning = None
        if conditional_params is not None:
            param_conditioning = torch.from_numpy(conditional_params).float().to(device)
            if param_conditioning.dim() == 1:
                param_conditioning = param_conditioning.unsqueeze(0)
            if param_conditioning.shape[0] == 1 and batch_size > 1:
                param_conditioning = param_conditioning.expand(batch_size, -1)
        
        return self.sample(
            conditioning=conditioning,
            batch_size=batch_size,
            param_conditioning=param_conditioning,
            n_steps=n_sampling_steps,
        )
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        if self.lr_scheduler_type == 'cosine':
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / max(1, self.warmup_steps)
                else:
                    progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        
        return optimizer
