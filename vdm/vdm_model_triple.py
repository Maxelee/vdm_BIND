"""
Triple VDM Model - Train three separate single-channel models simultaneously.

This module wraps three independent CleanVDM models (one for each channel):
- Model 1: Hydro DM (dark matter)
- Model 2: Gas
- Model 3: Stars

IMPORTANT: Complete Independence
================================
Each model is TRULY INDEPENDENT:
- Separate optimizer (AdamW) for each model
- Separate backward pass for each model
- NO gradient sharing between models
- Each model learns based ONLY on its own loss

Training flow per batch:
1. Forward hydro_dm_model → loss1 → backward → update hydro_dm_model
2. Forward gas_model → loss2 → backward → update gas_model  
3. Forward stars_model → loss3 → backward → update stars_model

The combined loss (loss1 + loss2 + loss3) is computed only for logging,
NOT for gradients. Each model's gradients come solely from its own loss.

Key features:
- Three separate 1-channel models (not one 3-channel model)
- Independent noise schedules and parameters per channel
- Independent optimizers and learning rate schedules
- Can save/load each model independently
- More memory efficient than 3-channel model (3x smaller per model)
"""

import torch
import numpy as np
from torch import nn, Tensor
from typing import Optional, Tuple, Dict
from lightning.pytorch import LightningModule

from vdm.vdm_model_clean import CleanVDM
from vdm.networks_clean import UNetVDM


class TripleVDM(nn.Module):
    """
    Triple VDM wrapper that manages three independent single-channel models.
    
    Each model learns to predict one channel independently:
    - hydro_dm_model: Predicts dark matter (channel 0)
    - gas_model: Predicts gas (channel 1)  
    - stars_model: Predicts stars (channel 2)
    
    Args:
        hydro_dm_model: CleanVDM model for dark matter
        gas_model: CleanVDM model for gas
        stars_model: CleanVDM model for stars
        channel_weights: Loss weights for each channel [DM, Gas, Stars] (default: (1.0, 1.0, 1.0))
    """
    
    def __init__(
        self,
        hydro_dm_model: CleanVDM,
        gas_model: CleanVDM,
        stars_model: CleanVDM,
        channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        
        print("\n" + "="*80)
        print("INITIALIZING TRIPLE VDM MODEL")
        print("="*80)
        print("Training three separate single-channel models:")
        print("  Model 1: Hydro DM (dark matter)")
        print("  Model 2: Gas")
        print("  Model 3: Stars")
        print(f"\nChannel weights: {channel_weights}")
        print("="*80 + "\n")
        
        self.hydro_dm_model = hydro_dm_model
        self.gas_model = gas_model
        self.stars_model = stars_model
        
        self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))
    
    def get_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute combined loss from all three models.
        
        Args:
            x: Target data (B, 3, H, W) - contains all three channels
            conditioning: Conditioning information (B, C_cond, H, W)
            param_conditioning: Optional parameter conditioning (B, N_params)
        
        Returns:
            loss: Combined weighted loss
            metrics: Dictionary of metrics from all three models
        """
        # Split 3-channel input into three 1-channel inputs
        hydro_dm_target = x[:, 0:1, :, :]  # (B, 1, H, W)
        gas_target = x[:, 1:2, :, :]       # (B, 1, H, W)
        stars_target = x[:, 2:3, :, :]     # (B, 1, H, W)
        
        # Compute loss for each model independently
        hydro_dm_loss, hydro_dm_metrics = self.hydro_dm_model.get_loss(
            hydro_dm_target, conditioning, param_conditioning
        )
        gas_loss, gas_metrics = self.gas_model.get_loss(
            gas_target, conditioning, param_conditioning
        )
        stars_loss, stars_metrics = self.stars_model.get_loss(
            stars_target, conditioning, param_conditioning
        )
        
        # Ensure losses are scalars (take mean if needed)
        if hydro_dm_loss.numel() > 1:
            hydro_dm_loss = hydro_dm_loss.mean()
        if gas_loss.numel() > 1:
            gas_loss = gas_loss.mean()
        if stars_loss.numel() > 1:
            stars_loss = stars_loss.mean()
        
        # Combine losses with channel weights
        combined_loss = (
            self.channel_weights[0] * hydro_dm_loss +
            self.channel_weights[1] * gas_loss +
            self.channel_weights[2] * stars_loss
        )
        
        # Combine metrics with channel prefixes (ensure all are scalars)
        combined_metrics = {
            # Overall metrics
            'elbo': combined_loss,
        }
        
        # Add hydro_dm metrics
        for key, value in hydro_dm_metrics.items():
            if isinstance(value, Tensor) and value.numel() > 1:
                combined_metrics[f'hydro_dm/{key}'] = value.mean()
            else:
                combined_metrics[f'hydro_dm/{key}'] = value
        
        # Add gas metrics
        for key, value in gas_metrics.items():
            if isinstance(value, Tensor) and value.numel() > 1:
                combined_metrics[f'gas/{key}'] = value.mean()
            else:
                combined_metrics[f'gas/{key}'] = value
        
        # Add stars metrics
        for key, value in stars_metrics.items():
            if isinstance(value, Tensor) and value.numel() > 1:
                combined_metrics[f'stars/{key}'] = value.mean()
            else:
                combined_metrics[f'stars/{key}'] = value
        
        return combined_loss, combined_metrics
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: int,
        device: str = 'cuda',
        param_conditioning: Optional[Tensor] = None,
        return_all: bool = False,
        verbose: bool = False
    ) -> Tensor:
        """
        Generate samples from all three models and combine into 3-channel output.
        
        Args:
            conditioning: Conditioning information (B, C_cond, H, W)
            batch_size: Number of samples to generate
            n_sampling_steps: Number of denoising steps
            device: Device to generate samples on
            param_conditioning: Optional parameter conditioning (B, N_params)
            return_all: Return all intermediate steps
            verbose: Show progress bar
        
        Returns:
            Generated samples (B, 3, H, W) or (n_steps, B, 3, H, W) if return_all=True
        """
        # Sample from each model independently
        hydro_dm_samples = self.hydro_dm_model.sample(
            conditioning=conditioning,
            batch_size=batch_size,
            n_sampling_steps=n_sampling_steps,
            device=device,
            param_conditioning=param_conditioning,
            return_all=return_all,
            verbose=verbose
        )
        
        gas_samples = self.gas_model.sample(
            conditioning=conditioning,
            batch_size=batch_size,
            n_sampling_steps=n_sampling_steps,
            device=device,
            param_conditioning=param_conditioning,
            return_all=return_all,
            verbose=verbose
        )
        
        stars_samples = self.stars_model.sample(
            conditioning=conditioning,
            batch_size=batch_size,
            n_sampling_steps=n_sampling_steps,
            device=device,
            param_conditioning=param_conditioning,
            return_all=return_all,
            verbose=verbose
        )
        
        # Combine into 3-channel output
        if return_all:
            # (n_steps, B, 1, H, W) -> (n_steps, B, 3, H, W)
            combined_samples = torch.cat([hydro_dm_samples, gas_samples, stars_samples], dim=2)
        else:
            # (B, 1, H, W) -> (B, 3, H, W)
            combined_samples = torch.cat([hydro_dm_samples, gas_samples, stars_samples], dim=1)
        
        return combined_samples


class LightTripleVDM(LightningModule):
    """
    PyTorch Lightning wrapper for TripleVDM.
    
    Handles training, validation, and optimization of three models simultaneously.
    Each model has its own optimizer and backward pass for complete independence.
    """
    
    def __init__(
        self,
        hydro_dm_score_model: nn.Module,
        gas_score_model: nn.Module,
        stars_score_model: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 1.0e-5,
        lr_scheduler: str = 'onecycle',
        n_sampling_steps: int = 250,
        draw_figure=None,
        dataset: str = 'illustris',
        # VDM parameters (shared across models unless specified)
        noise_schedule: str = "fixed_linear",
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        antithetic_time_sampling: bool = True,
        image_shape: Tuple[int, int, int] = (1, 128, 128),  # Single channel per model
        data_noise: float = 1e-3,
        lambdas: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        # Per-channel VDM parameters (optional)
        hydro_dm_params: Optional[Dict] = None,
        gas_params: Optional[Dict] = None,
        stars_params: Optional[Dict] = None,
        # Focal loss (typically only for stars)
        use_focal_loss_hydro_dm: bool = False,
        use_focal_loss_gas: bool = False,
        use_focal_loss_stars: bool = False,
        focal_gamma: float = 2.0,
        # Parameter prediction
        use_param_prediction: bool = False,
        param_prediction_weight: float = 0.01,
    ):
        super().__init__()
        
        # Default VDM parameters
        default_vdm_params = {
            'noise_schedule': noise_schedule,
            'gamma_min': gamma_min,
            'gamma_max': gamma_max,
            'antithetic_time_sampling': antithetic_time_sampling,
            'image_shape': image_shape,
            'data_noise': data_noise,
            'lambdas': lambdas,
            'channel_weights': (1.0,),  # Single channel per model
            'use_param_prediction': use_param_prediction,
            'param_prediction_weight': param_prediction_weight,
        }
        
        # Create hydro DM model
        hydro_dm_vdm_params = {**default_vdm_params}
        if hydro_dm_params:
            hydro_dm_vdm_params.update(hydro_dm_params)
        hydro_dm_vdm_params['use_focal_loss'] = use_focal_loss_hydro_dm
        hydro_dm_vdm_params['focal_gamma'] = focal_gamma
        
        hydro_dm_model = CleanVDM(
            score_model=hydro_dm_score_model,
            **hydro_dm_vdm_params
        )
        
        # Create gas model
        gas_vdm_params = {**default_vdm_params}
        if gas_params:
            gas_vdm_params.update(gas_params)
        gas_vdm_params['use_focal_loss'] = use_focal_loss_gas
        gas_vdm_params['focal_gamma'] = focal_gamma
        
        gas_model = CleanVDM(
            score_model=gas_score_model,
            **gas_vdm_params
        )
        
        # Create stars model
        stars_vdm_params = {**default_vdm_params}
        if stars_params:
            stars_vdm_params.update(stars_params)
        stars_vdm_params['use_focal_loss'] = use_focal_loss_stars
        stars_vdm_params['focal_gamma'] = focal_gamma
        
        stars_model = CleanVDM(
            score_model=stars_score_model,
            **stars_vdm_params
        )
        
        # Create triple model
        self.model = TripleVDM(
            hydro_dm_model=hydro_dm_model,
            gas_model=gas_model,
            stars_model=stars_model,
            channel_weights=channel_weights,
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.n_sampling_steps = n_sampling_steps
        self.draw_figure = draw_figure
        self.dataset = dataset
        
        # Store channel weights for reporting
        self.channel_weights = channel_weights
        
        self.save_hyperparameters(ignore=[
            'hydro_dm_score_model', 
            'gas_score_model', 
            'stars_score_model', 
            'draw_figure'
        ])
        
        # Enable manual optimization for independent backward passes
        self.automatic_optimization = False
    
    def forward(self, x: Tensor, conditioning: Tensor, param_conditioning: Optional[Tensor] = None):
        """Forward pass."""
        return self.model.get_loss(x, conditioning, param_conditioning)
    
    def training_step(self, batch: Tuple, batch_idx: int):
        """Training step with manual optimization for independent models."""
        m_dm, large_scale, m_target, param_conditioning = batch
        
        # Get the three optimizers
        opt_hydro_dm, opt_gas, opt_stars = self.optimizers()
        
        # Concatenate spatial conditioning
        conditioning = torch.cat([m_dm, large_scale], dim=1)
        
        # Split 3-channel input into three 1-channel inputs
        hydro_dm_target = m_target[:, 0:1, :, :]
        gas_target = m_target[:, 1:2, :, :]
        stars_target = m_target[:, 2:3, :, :]
        
        # ===== Train Hydro DM Model Independently =====
        opt_hydro_dm.zero_grad()
        hydro_dm_loss, hydro_dm_metrics = self.model.hydro_dm_model.get_loss(
            hydro_dm_target, conditioning, param_conditioning
        )
        # Ensure scalar loss for backward
        if hydro_dm_loss.numel() > 1:
            hydro_dm_loss = hydro_dm_loss.mean()
        self.manual_backward(hydro_dm_loss)
        # Manual gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.hydro_dm_model.parameters(), max_norm=1.0)
        opt_hydro_dm.step()
        
        # ===== Train Gas Model Independently =====
        opt_gas.zero_grad()
        gas_loss, gas_metrics = self.model.gas_model.get_loss(
            gas_target, conditioning, param_conditioning
        )
        # Ensure scalar loss for backward
        if gas_loss.numel() > 1:
            gas_loss = gas_loss.mean()
        self.manual_backward(gas_loss)
        # Manual gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.gas_model.parameters(), max_norm=1.0)
        opt_gas.step()
        
        # ===== Train Stars Model Independently =====
        opt_stars.zero_grad()
        stars_loss, stars_metrics = self.model.stars_model.get_loss(
            stars_target, conditioning, param_conditioning
        )
        # Ensure scalar loss for backward
        if stars_loss.numel() > 1:
            stars_loss = stars_loss.mean()
        self.manual_backward(stars_loss)
        # Manual gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.stars_model.parameters(), max_norm=1.0)
        opt_stars.step()
        
        # Compute combined loss for reporting only
        combined_loss = (
            self.channel_weights[0] * hydro_dm_loss +
            self.channel_weights[1] * gas_loss +
            self.channel_weights[2] * stars_loss
        )
        
        # Combine metrics for logging
        prog_bar_metrics = {'elbo', 'hydro_dm/diffusion_loss', 'gas/diffusion_loss', 'stars/diffusion_loss'}
        
        metrics = {
            'elbo': combined_loss,
        }
        
        # Add all metrics from each model with prefixes
        for key, value in hydro_dm_metrics.items():
            metrics[f'hydro_dm/{key}'] = value
        for key, value in gas_metrics.items():
            metrics[f'gas/{key}'] = value
        for key, value in stars_metrics.items():
            metrics[f'stars/{key}'] = value
        
        for key, value in metrics.items():
            self.log(
                f"train/{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(key in prog_bar_metrics),
                logger=True,
                sync_dist=True,
            )
        
        # Step learning rate schedulers if needed
        sch_hydro_dm, sch_gas, sch_stars = self.lr_schedulers()
        if self.lr_scheduler in ['onecycle', 'cosine_warmup']:
            sch_hydro_dm.step()
            sch_gas.step()
            sch_stars.step()
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Validation step."""
        m_dm, large_scale, m_target, param_conditioning = batch
        
        # Concatenate spatial conditioning
        conditioning = torch.cat([m_dm, large_scale], dim=1)
        
        # Compute loss
        loss, metrics = self.model.get_loss(m_target, conditioning, param_conditioning)
        
        # Log metrics
        prog_bar_metrics = {'elbo', 'hydro_dm/diffusion_loss', 'gas/diffusion_loss', 'stars/diffusion_loss'}
        
        for key, value in metrics.items():
            self.log(
                f"val/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(key in prog_bar_metrics),
                logger=True,
                sync_dist=True,
            )
        
        # Generate samples for visualization (first batch only)
        if batch_idx == 0 and self.draw_figure is not None:
            samples = self.model.sample(
                conditioning=conditioning[:4],
                batch_size=4,
                n_sampling_steps=self.n_sampling_steps,
                param_conditioning=param_conditioning[:4] if param_conditioning is not None else None,
                verbose=False
            )
            
            fig = self.draw_figure(
                samples[:4].cpu().numpy(),
                m_target[:4].cpu().numpy(),
                conditioning[:4].cpu().numpy(),
                self.dataset
            )
            self.logger.experiment.add_figure("val/samples", fig, self.global_step)
        
        return loss
    
    def draw_samples(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: int,
        param_conditioning: Optional[Tensor] = None,
        verbose: bool = False,
    ) -> Tensor:
        """Draw samples from the model."""
        return self.model.sample(
            conditioning=conditioning,
            batch_size=batch_size,
            n_sampling_steps=n_sampling_steps,
            param_conditioning=param_conditioning,
            verbose=verbose,
        )
    
    def configure_optimizers(self):
        """Configure three independent optimizers and schedulers."""
        # Create separate optimizer for each model
        opt_hydro_dm = torch.optim.AdamW(
            self.model.hydro_dm_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        opt_gas = torch.optim.AdamW(
            self.model.gas_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        opt_stars = torch.optim.AdamW(
            self.model.stars_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler == 'onecycle':
            sch_hydro_dm = torch.optim.lr_scheduler.OneCycleLR(
                opt_hydro_dm,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )
            sch_gas = torch.optim.lr_scheduler.OneCycleLR(
                opt_gas,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )
            sch_stars = torch.optim.lr_scheduler.OneCycleLR(
                opt_stars,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )
            
            return [
                {'optimizer': opt_hydro_dm, 'lr_scheduler': {'scheduler': sch_hydro_dm, 'interval': 'step', 'frequency': 1}},
                {'optimizer': opt_gas, 'lr_scheduler': {'scheduler': sch_gas, 'interval': 'step', 'frequency': 1}},
                {'optimizer': opt_stars, 'lr_scheduler': {'scheduler': sch_stars, 'interval': 'step', 'frequency': 1}}
            ]
        elif self.lr_scheduler == 'plateau':
            sch_hydro_dm = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_hydro_dm,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=1e-7
            )
            sch_gas = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_gas,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=1e-7
            )
            sch_stars = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_stars,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=1e-7
            )
            return [
                {'optimizer': opt_hydro_dm, 'lr_scheduler': {'scheduler': sch_hydro_dm, 'monitor': 'val/hydro_dm/elbo', 'interval': 'epoch', 'frequency': 1}},
                {'optimizer': opt_gas, 'lr_scheduler': {'scheduler': sch_gas, 'monitor': 'val/gas/elbo', 'interval': 'epoch', 'frequency': 1}},
                {'optimizer': opt_stars, 'lr_scheduler': {'scheduler': sch_stars, 'monitor': 'val/stars/elbo', 'interval': 'epoch', 'frequency': 1}}
            ]
        elif self.lr_scheduler == 'cosine':
            sch_hydro_dm = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_hydro_dm,
                T_max=self.trainer.max_epochs,
                eta_min=1e-7
            )
            sch_gas = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_gas,
                T_max=self.trainer.max_epochs,
                eta_min=1e-7
            )
            sch_stars = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_stars,
                T_max=self.trainer.max_epochs,
                eta_min=1e-7
            )
            return [
                {'optimizer': opt_hydro_dm, 'lr_scheduler': {'scheduler': sch_hydro_dm, 'interval': 'epoch', 'frequency': 1}},
                {'optimizer': opt_gas, 'lr_scheduler': {'scheduler': sch_gas, 'interval': 'epoch', 'frequency': 1}},
                {'optimizer': opt_stars, 'lr_scheduler': {'scheduler': sch_stars, 'interval': 'epoch', 'frequency': 1}}
            ]
        elif self.lr_scheduler == 'cosine_warmup':
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(0.05 * total_steps)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                else:
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return 0.5 * (1.0 + np.cos(np.pi * progress))
            
            sch_hydro_dm = torch.optim.lr_scheduler.LambdaLR(opt_hydro_dm, lr_lambda)
            sch_gas = torch.optim.lr_scheduler.LambdaLR(opt_gas, lr_lambda)
            sch_stars = torch.optim.lr_scheduler.LambdaLR(opt_stars, lr_lambda)
            return [
                {'optimizer': opt_hydro_dm, 'lr_scheduler': {'scheduler': sch_hydro_dm, 'interval': 'step', 'frequency': 1}},
                {'optimizer': opt_gas, 'lr_scheduler': {'scheduler': sch_gas, 'interval': 'step', 'frequency': 1}},
                {'optimizer': opt_stars, 'lr_scheduler': {'scheduler': sch_stars, 'interval': 'step', 'frequency': 1}}
            ]
        elif self.lr_scheduler == 'cosine_restart':
            sch_hydro_dm = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_hydro_dm,
                T_0=20,
                T_mult=2,
                eta_min=self.learning_rate * 0.01
            )
            sch_gas = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_gas,
                T_0=20,
                T_mult=2,
                eta_min=self.learning_rate * 0.01
            )
            sch_stars = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_stars,
                T_0=20,
                T_mult=2,
                eta_min=self.learning_rate * 0.01
            )
            return [
                {'optimizer': opt_hydro_dm, 'lr_scheduler': {'scheduler': sch_hydro_dm, 'interval': 'epoch', 'frequency': 1}},
                {'optimizer': opt_gas, 'lr_scheduler': {'scheduler': sch_gas, 'interval': 'epoch', 'frequency': 1}},
                {'optimizer': opt_stars, 'lr_scheduler': {'scheduler': sch_stars, 'interval': 'epoch', 'frequency': 1}}
            ]
        elif self.lr_scheduler == 'constant':
            return [opt_hydro_dm, opt_gas, opt_stars]
        else:
            return [opt_hydro_dm, opt_gas, opt_stars]
