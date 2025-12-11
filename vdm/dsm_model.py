"""
DSM Model using custom UNet for VDM-BIND.

This module implements Denoising Score Matching (DSM) using the custom UNet
from networks_clean.py, allowing fair comparison with VDM and Interpolant models.

Key differences from ddpm_model.py:
- Uses networks_clean.py UNet instead of score_models' NCSNpp
- Same architecture as VDM/Interpolant (Fourier features, cross-attention, FiLM)
- DSM loss instead of VDM ELBO or flow matching velocity loss

DSM Loss:
- Forward: z_t = alpha_t * x + sigma_t * epsilon (variance-preserving)
- Predict: epsilon_hat = model(z_t, t, condition)
- Loss: || epsilon_hat - epsilon ||^2 weighted by SNR

This is mathematically equivalent to DDPM but:
- More modern implementation (VP-SDE formulation)
- Uses same architecture as other models for fair comparison

Usage:
    from vdm.dsm_model import LightDSM
    from vdm.networks_clean import UNetVDM
    
    # Create network
    unet = UNetVDM(
        embedding_dim=96,
        n_blocks=5,
        conditioning_channels=4,
    )
    
    # Wrap in Lightning module
    model = LightDSM(
        score_model=unet,
        learning_rate=1e-4,
    )
"""

import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Union, List
from tqdm import tqdm

from lightning.pytorch import LightningModule


# ============================================================================
# VP-SDE Noise Schedule
# ============================================================================

class VPSDESchedule:
    """
    Variance Preserving SDE noise schedule.
    
    This matches the VP-SDE formulation from "Score-Based Generative Modeling
    through Stochastic Differential Equations" (Song et al., 2021).
    
    beta(t) = beta_min + t * (beta_max - beta_min)
    
    For VP-SDE:
    - alpha_t = exp(-0.5 * integral_0^t beta(s) ds) = exp(-0.5 * beta_min * t - 0.25 * (beta_max - beta_min) * t^2)
    - sigma_t = sqrt(1 - alpha_t^2)
    
    Args:
        beta_min: Minimum beta (default: 0.1)
        beta_max: Maximum beta (default: 20.0)
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def alpha(self, t: Tensor) -> Tensor:
        """Compute alpha(t) = sqrt(alpha_bar(t))."""
        # Integral of beta(s) from 0 to t:
        # integral = beta_min * t + 0.5 * (beta_max - beta_min) * t^2
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)
    
    def sigma(self, t: Tensor) -> Tensor:
        """Compute sigma(t) = sqrt(1 - alpha_bar(t))."""
        alpha_t = self.alpha(t)
        return torch.sqrt(1 - alpha_t ** 2 + 1e-8)
    
    def snr(self, t: Tensor) -> Tensor:
        """Compute Signal-to-Noise Ratio = alpha^2 / sigma^2."""
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        return alpha_t ** 2 / (sigma_t ** 2 + 1e-8)
    
    def log_snr(self, t: Tensor) -> Tensor:
        """Compute log SNR (gamma in VDM notation)."""
        return torch.log(self.snr(t) + 1e-8)


# ============================================================================
# UNet Wrapper for DSM
# ============================================================================

class UNetDSMWrapper(nn.Module):
    """
    Wrapper to make UNetVDM compatible with DSM training.
    
    The main difference is handling of time embedding:
    - UNetVDM expects time in [0, 1]
    - We ensure proper conditioning is passed
    
    Also handles the output format (may return tuple with aux outputs).
    """
    
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
    
    def forward(
        self,
        t: Tensor,
        x: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through UNet.
        
        Args:
            t: Time tensor (B,) in [0, 1]
            x: Noisy input (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            param_conditioning: Parameter vector (B, N_params)
        
        Returns:
            Predicted noise (B, C, H, W)
        """
        # UNetVDM expects: (z, t, conditioning, param_conditioning)
        # where z is the noisy sample
        output = self.net(x, t, conditioning=conditioning, param_conditioning=param_conditioning)
        
        # Handle tuple output (may include param prediction, mask, etc.)
        if isinstance(output, tuple):
            return output[0]
        return output


# ============================================================================
# Lightning Module for DSM
# ============================================================================

class LightDSM(LightningModule):
    """
    PyTorch Lightning module for Denoising Score Matching with custom UNet.
    
    Uses the same UNet architecture as VDM/Interpolant but with DSM loss:
    - Forward: z_t = alpha_t * x + sigma_t * epsilon
    - Predict: epsilon_hat = model(z_t, t, condition)
    - Loss: || epsilon_hat - epsilon ||^2
    
    This allows fair comparison between:
    - VDM (ELBO loss with learned/fixed gamma schedule)
    - DSM (simple MSE on noise prediction with VP-SDE schedule)
    - Interpolant (MSE on velocity prediction with linear interpolation)
    
    Args:
        score_model: Neural network (UNetVDM or wrapped)
        beta_min: Minimum beta for VP-SDE
        beta_max: Maximum beta for VP-SDE
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        lr_scheduler: Learning rate scheduler type
        n_sampling_steps: Number of steps for sampling
        use_param_conditioning: Whether to include astro params
        use_snr_weighting: Weight loss by SNR (like VDM)
        channel_weights: Per-channel loss weights
    """
    
    def __init__(
        self,
        score_model: nn.Module,
        # Noise schedule
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        # Sampling parameters
        n_sampling_steps: int = 250,
        # Conditioning
        use_param_conditioning: bool = False,
        # Loss weighting
        use_snr_weighting: bool = True,  # Weight by SNR derivative like VDM
        channel_weights: Tuple[float, ...] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.n_sampling_steps = n_sampling_steps
        self.use_param_conditioning = use_param_conditioning
        self.use_snr_weighting = use_snr_weighting
        
        # Wrap model if needed
        if isinstance(score_model, UNetDSMWrapper):
            self.model = score_model
        else:
            self.model = UNetDSMWrapper(score_model)
        
        # Noise schedule
        self.schedule = VPSDESchedule(beta_min=beta_min, beta_max=beta_max)
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Channel weights
        self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['score_model'])
        
        print(f"\n{'='*60}")
        print(f"INITIALIZED LIGHT DSM MODEL (Custom UNet)")
        print(f"{'='*60}")
        print(f"  Beta range: [{beta_min}, {beta_max}]")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Sampling steps: {n_sampling_steps}")
        print(f"  SNR weighting: {use_snr_weighting}")
        print(f"  Channel weights: {channel_weights}")
        print(f"  Param conditioning: {use_param_conditioning}")
        print(f"{'='*60}\n")
    
    def _add_noise(
        self,
        x: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Add noise to data using variance-preserving diffusion.
        
        z_t = alpha_t * x + sigma_t * epsilon
        
        Args:
            x: Clean data (B, C, H, W)
            t: Diffusion times in [0, 1] (B,)
            noise: Optional pre-generated noise
        
        Returns:
            z_t: Noisy data (B, C, H, W)
            noise: The noise that was added (B, C, H, W)
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1, 1)
        t_broadcast = t.view(t.shape[0], 1, 1, 1)
        
        alpha_t = self.schedule.alpha(t_broadcast)
        sigma_t = self.schedule.sigma(t_broadcast)
        
        if noise is None:
            noise = torch.randn_like(x)
        
        z_t = alpha_t * x + sigma_t * noise
        
        return z_t, noise
    
    def _compute_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        params: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute DSM loss.
        
        Args:
            x: Clean target (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            params: Optional cosmological parameters (B, N_params)
        
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics for logging
        """
        B = x.shape[0]
        device = x.device
        
        # Sample random times uniformly in [epsilon, 1-epsilon]
        # Avoid t=0 (no noise) and t=1 (pure noise)
        epsilon = 1e-5
        t = torch.rand(B, device=device) * (1 - 2 * epsilon) + epsilon
        
        # Add noise
        z_t, noise = self._add_noise(x, t)
        
        # Predict noise
        noise_pred = self.model(t, z_t, conditioning, params)
        
        # Compute per-channel MSE
        n_channels = x.shape[1]
        channel_losses = []
        
        for c in range(n_channels):
            mse_c = F.mse_loss(noise_pred[:, c], noise[:, c], reduction='none')
            mse_c = mse_c.flatten(start_dim=1).mean(dim=1)  # (B,)
            weighted_mse_c = self.channel_weights[c] * mse_c
            channel_losses.append(weighted_mse_c)
        
        # Sum across channels
        loss_per_sample = sum(channel_losses)  # (B,)
        
        # Optional: weight by SNR derivative (like VDM)
        if self.use_snr_weighting:
            # Weight by d(log_snr)/dt to match VDM loss weighting
            # For VP-SDE: d(log_snr)/dt ≈ -beta(t)
            beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
            weight = beta_t  # Proportional to noise rate
            loss_per_sample = loss_per_sample * weight
        
        loss = loss_per_sample.mean()
        
        # Compute metrics
        with torch.no_grad():
            total_mse = F.mse_loss(noise_pred, noise)
            dm_mse = F.mse_loss(noise_pred[:, 0], noise[:, 0])
            gas_mse = F.mse_loss(noise_pred[:, 1], noise[:, 1])
            stellar_mse = F.mse_loss(noise_pred[:, 2], noise[:, 2])
        
        metrics = {
            'total_mse': total_mse,
            'dm_mse': dm_mse,
            'gas_mse': gas_mse,
            'stellar_mse': stellar_mse,
        }
        
        return loss, metrics
    
    def _unpack_batch(self, batch: Tuple) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Unpack batch from AstroDataset.
        
        AstroDataset returns: (m_dm, large_scale, m_target, conditions)
        
        Returns:
            x: Target (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            params: Cosmological parameters or None
        """
        m_dm, large_scale, m_target, conditions = batch
        
        # Target is what we want to generate
        x = m_target  # (B, 3, H, W)
        
        # Spatial conditioning: concatenate DM and large-scale context
        conditioning = torch.cat([m_dm, large_scale], dim=1)  # (B, 1+N, H, W)
        
        # Parameters
        params = conditions if self.use_param_conditioning else None
        
        return x, conditioning, params
    
    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Training step with DSM loss."""
        x, conditioning, params = self._unpack_batch(batch)
        
        loss, metrics = self._compute_loss(x, conditioning, params)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mse", metrics['total_mse'], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/dm_mse", metrics['dm_mse'], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/gas_mse", metrics['gas_mse'], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/stellar_mse", metrics['stellar_mse'], on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Validation step."""
        x, conditioning, params = self._unpack_batch(batch)
        
        loss, metrics = self._compute_loss(x, conditioning, params)
        
        # Log metrics
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mse", metrics['total_mse'], on_epoch=True, sync_dist=True)
        self.log("val/dm_mse", metrics['dm_mse'], on_epoch=True, sync_dist=True)
        self.log("val/gas_mse", metrics['gas_mse'], on_epoch=True, sync_dist=True)
        self.log("val/stellar_mse", metrics['stellar_mse'], on_epoch=True, sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
        steps: Optional[int] = None,
        verbose: bool = True,
    ) -> Tensor:
        """
        Generate samples using DDPM reverse process.
        
        Uses the reverse SDE discretization:
        x_{t-dt} = x_t + beta_t * [x_t + 2 * score(x_t, t)] * dt + sqrt(2 * beta_t * dt) * z
        
        For noise prediction, score = -epsilon / sigma_t
        
        Args:
            shape: Shape of samples (B, C, H, W)
            conditioning: Spatial conditioning (B, C_cond, H, W)
            param_conditioning: Optional parameter vector (B, N_params)
            steps: Number of sampling steps
            verbose: Show progress bar
        
        Returns:
            samples: Generated samples (B, C, H, W)
        """
        if steps is None:
            steps = self.n_sampling_steps
        
        device = conditioning.device
        B = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Time discretization (from 1 to 0)
        times = torch.linspace(1.0, 1e-5, steps + 1, device=device)
        
        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling")
        
        for i in iterator:
            t = times[i]
            t_next = times[i + 1]
            dt = t - t_next
            
            # Expand t for batch
            t_batch = t.expand(B)
            t_broadcast = t.view(1, 1, 1, 1).expand(B, 1, 1, 1)
            
            # Predict noise
            noise_pred = self.model(t_batch, x, conditioning, param_conditioning)
            
            # Compute score from noise prediction: s = -epsilon / sigma_t
            sigma_t = self.schedule.sigma(t_broadcast)
            score = -noise_pred / (sigma_t + 1e-8)
            
            # Reverse SDE step (Euler-Maruyama)
            beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
            
            # Drift: beta_t * [x_t + 2 * score] * dt
            drift = beta_t * (x + 2 * score) * dt
            
            # Diffusion: sqrt(2 * beta_t * dt) * z (only if not last step)
            if i < steps - 1:
                diffusion = torch.sqrt(2 * beta_t * dt) * torch.randn_like(x)
            else:
                diffusion = 0
            
            x = x + drift + diffusion
        
        return x
    
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
            batch_size: Number of samples (should match conditioning batch size)
            n_sampling_steps: Number of sampling steps
            param_conditioning: Optional parameter conditioning
            verbose: Show progress bar
        
        Returns:
            samples: Generated samples (B, 3, H, W)
        """
        B, C_cond, H, W = conditioning.shape
        C_out = 3  # [DM, Gas, Stars]
        
        return self.sample(
            shape=(batch_size, C_out, H, W),
            conditioning=conditioning,
            param_conditioning=param_conditioning,
            steps=n_sampling_steps,
            verbose=verbose,
        )
    
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
        elif self.lr_scheduler_type == "cosine_warmup":
            def lr_lambda(step):
                warmup_steps = 1000
                if step < warmup_steps:
                    return step / warmup_steps
                total_steps = self.trainer.estimated_stepping_batches if self.trainer else 100000
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        else:
            return optimizer


# ============================================================================
# Alias for backwards compatibility
# ============================================================================

LightDenoisingScoreMatching = LightDSM
