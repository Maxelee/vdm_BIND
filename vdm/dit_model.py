"""
PyTorch Lightning module for DiT-based VDM training.

This module wraps the DiT architecture with the same training interface
as LightCleanVDM, enabling direct comparison with UNet-based models.
"""

from typing import Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from lightning.pytorch import LightningModule

from .dit import DiT, create_dit_model


class LightDiTVDM(LightningModule):
    """
    Lightning module for training DiT with VDM loss.
    
    Uses the same VDM (Variational Diffusion Model) training objective
    as LightCleanVDM but with DiT backbone instead of UNet.
    
    Args:
        dit_model: DiT model or model name string
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization
        gamma_min: Minimum log SNR (gamma) value
        gamma_max: Maximum log SNR (gamma) value
        n_sampling_steps: Number of steps for sampling
        loss_type: 'mse', 'l1', or 'huber'
        lr_scheduler: Type of LR scheduler
        warmup_steps: Number of warmup steps for scheduler
        image_shape: Shape of output images (C, H, W)
        use_ema: Whether to use EMA weights
        ema_decay: EMA decay rate
        
        # DiT-specific args (if dit_model is string)
        img_size: Image size
        patch_size: Patch size
        hidden_size: Transformer hidden dimension
        depth: Number of DiT blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden multiplier
        n_params: Number of conditioning parameters
        conditioning_channels: Number of DM conditioning channels
        large_scale_channels: Number of large-scale context channels
    """
    
    def __init__(
        self,
        dit_model: Union[DiT, str] = 'DiT-B/4',
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
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        # DiT params (used if dit_model is string)
        img_size: int = 64,
        patch_size: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        n_params: int = 35,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Build or use provided model
        if isinstance(dit_model, str):
            self.score_model = create_dit_model(
                dit_model,
                img_size=img_size,
                n_params=n_params,
                conditioning_channels=conditioning_channels,
                large_scale_channels=large_scale_channels,
                dropout=dropout,
            )
        else:
            self.score_model = dit_model
        
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
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Derived quantities
        self.n_channels = image_shape[0]
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['dit_model'])
        
        print(f"\n{'='*60}")
        print(f"INITIALIZED LightDiTVDM")
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
        """
        return self.gamma_max + (self.gamma_min - self.gamma_max) * t
    
    def sigma_and_alpha(self, gamma: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute sigma and alpha from gamma.
        
        gamma = log(alpha^2 / sigma^2)
        => alpha^2 = sigmoid(gamma), sigma^2 = sigmoid(-gamma)
        """
        alpha_squared = torch.sigmoid(gamma)
        sigma_squared = torch.sigmoid(-gamma)
        return torch.sqrt(sigma_squared), torch.sqrt(alpha_squared)
    
    # =========================================================================
    # Diffusion Loss
    # =========================================================================
    
    def diffusion_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute VDM diffusion loss.
        
        Args:
            x: (B, C, H, W) - target images
            conditioning: (B, C_cond, H, W) - DM + large-scale conditioning
            param_conditioning: (B, N_params) - cosmological parameters
        
        Returns:
            loss: Scalar loss value
        """
        B = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.rand(B, device=device)
        
        # Get noise schedule parameters
        gamma_t = self.gamma(t)
        sigma, alpha = self.sigma_and_alpha(gamma_t)
        
        # Add noise to target
        eps = torch.randn_like(x)
        z_t = alpha[:, None, None, None] * x + sigma[:, None, None, None] * eps
        
        # Predict noise
        eps_pred = self.score_model(z_t, gamma_t, conditioning, param_conditioning)
        
        # Compute loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(eps_pred, eps)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(eps_pred, eps)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(eps_pred, eps)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    # =========================================================================
    # Training & Validation
    # =========================================================================
    
    def _unpack_batch(self, batch):
        """Unpack batch into components."""
        if len(batch) == 4:
            dm_condition, large_scale, target, params = batch
        elif len(batch) == 3:
            dm_condition, target, params = batch
            large_scale = None
        else:
            raise ValueError(f"Expected 3 or 4 items in batch, got {len(batch)}")
        
        # Combine conditioning
        if large_scale is not None:
            conditioning = torch.cat([dm_condition, large_scale], dim=1)
        else:
            conditioning = dm_condition
        
        return target, conditioning, params
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        target, conditioning, params = self._unpack_batch(batch)
        loss = self.diffusion_loss(target, conditioning, params)
        
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        target, conditioning, params = self._unpack_batch(batch)
        loss = self.diffusion_loss(target, conditioning, params)
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    # =========================================================================
    # Sampling
    # =========================================================================
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
        n_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Generate samples using DDPM sampling.
        
        Args:
            conditioning: (B, C_cond, H, W) - spatial conditioning
            param_conditioning: (B, N_params) - cosmological parameters
            n_steps: Number of sampling steps (default: self.n_sampling_steps)
            return_trajectory: Whether to return full trajectory
        
        Returns:
            samples: (B, C, H, W) or list of tensors if return_trajectory
        """
        if n_steps is None:
            n_steps = self.n_sampling_steps
        
        B = conditioning.shape[0]
        device = conditioning.device
        
        # Start from pure noise
        x = torch.randn(B, self.n_channels, *self.image_shape[1:], device=device)
        
        trajectory = [x] if return_trajectory else None
        
        # DDPM sampling loop
        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)
        
        for i in range(n_steps):
            t = timesteps[i].expand(B)
            t_next = timesteps[i + 1].expand(B)
            
            gamma_t = self.gamma(t)
            gamma_next = self.gamma(t_next)
            
            sigma_t, alpha_t = self.sigma_and_alpha(gamma_t)
            sigma_next, alpha_next = self.sigma_and_alpha(gamma_next)
            
            # Predict noise
            eps_pred = self.score_model(x, gamma_t, conditioning, param_conditioning)
            
            # DDPM update
            # x_0 estimate
            x0_pred = (x - sigma_t[:, None, None, None] * eps_pred) / alpha_t[:, None, None, None]
            
            # Compute mean for next step
            if i < n_steps - 1:
                # Not the last step - add noise
                c0 = alpha_next / alpha_t
                c1 = sigma_next * torch.sqrt(1 - (alpha_next / alpha_t) ** 2 * (sigma_t / sigma_next) ** 2)
                
                x = c0[:, None, None, None] * x + \
                    (alpha_next - c0 * alpha_t)[:, None, None, None] * x0_pred / alpha_t[:, None, None, None]
                
                # Add noise
                noise = torch.randn_like(x)
                x = x + c1[:, None, None, None] * noise
            else:
                # Last step - just return prediction
                x = x0_pred
            
            if return_trajectory:
                trajectory.append(x)
        
        if return_trajectory:
            return trajectory
        return x
    
    @torch.no_grad()
    def draw_samples(
        self,
        conditioning: Tensor,
        batch_size: Optional[int] = None,
        param_conditioning: Optional[Tensor] = None,
        n_sampling_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Draw samples compatible with BIND interface.
        
        Args:
            conditioning: (B, C_cond, H, W) - spatial conditioning
            batch_size: Not used (inferred from conditioning)
            param_conditioning: (B, N_params) - cosmological parameters
            n_sampling_steps: Number of sampling steps
        
        Returns:
            (B, C, H, W) samples
        """
        return self.sample(
            conditioning,
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
            betas=(0.9, 0.999),
        )
        
        if self.lr_scheduler_type == 'cosine':
            # Cosine annealing with warmup
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / max(1, self.warmup_steps)
                progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        return optimizer
