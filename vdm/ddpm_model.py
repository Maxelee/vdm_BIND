"""
DDPM/Score Model wrapper for VDM-BIND.

This module integrates the score_models package (https://github.com/AlexandreAdam/score_models)
with the existing VDM-BIND training pipeline. It maintains compatibility with:
- AstroDataset and AstroDataModule (existing data pipeline)
- Multi-scale conditioning (DM + large scale context)
- Parameter conditioning (cosmological parameters)
- PyTorch Lightning training framework

Key features:
- Uses score_models' ScoreModel with DDPM or NCSNpp architecture
- VP-SDE (Variance Preserving) or VE-SDE (Variance Exploding) noise schedules
- Denoising Score Matching (DSM) loss
- Compatible with existing checkpointing and logging

Usage:
    from vdm.ddpm_model import LightScoreModel
    from score_models import DDPM, NCSNpp
    
    # Create network
    net = NCSNpp(
        channels=3,  # Output channels [DM, Gas, Stars]
        nf=96,
        ch_mult=[1, 2, 4, 8],
        condition=["input"],  # Spatial conditioning
        condition_input_channels=4,  # DM + 3 large-scale
    )
    
    # Wrap in Lightning module
    model = LightScoreModel(
        model=net,
        sde="vp",  # or "ve"
        beta_min=0.1,
        beta_max=20.0,
        learning_rate=1e-4,
    )
"""

import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, Union, List
from tqdm import tqdm

from lightning.pytorch import LightningModule

# Import score_models components
try:
    from score_models import ScoreModel, DDPM, NCSNpp
    from score_models.sde import VPSDE, VESDE
    SCORE_MODELS_AVAILABLE = True
except ImportError:
    SCORE_MODELS_AVAILABLE = False
    print("Warning: score_models package not found. Install with: pip install score_models")


# ============================================================================
# Wrapper Dataset for score_models compatibility
# ============================================================================

class ScoreModelDatasetWrapper(Dataset):
    """
    Wraps the AstroDataset output to match score_models expected format.
    
    AstroDataset returns: (m_dm, large_scale, m_target, conditions)
    score_models expects: (x, *conditioning_args) where x is the target
    
    For "input" conditioning: returns (target, spatial_condition)
    For "vector" conditioning: returns (target, spatial_condition, param_vector)
    """
    
    def __init__(
        self, 
        base_dataset: Dataset,
        use_param_conditioning: bool = False,
        concat_dm_largescale: bool = True,
    ):
        """
        Args:
            base_dataset: AstroDataset instance
            use_param_conditioning: Whether to include cosmological parameters
            concat_dm_largescale: Whether to concatenate DM and large_scale as spatial condition
        """
        self.base_dataset = base_dataset
        self.use_param_conditioning = use_param_conditioning
        self.concat_dm_largescale = concat_dm_largescale
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get data from AstroDataset
        m_dm, large_scale, m_target, conditions = self.base_dataset[idx]
        
        # Target is what we want to generate
        x = m_target  # (3, H, W)
        
        # Spatial conditioning: concatenate DM and large-scale context
        if self.concat_dm_largescale:
            spatial_cond = torch.cat([m_dm, large_scale], dim=0)  # (1+N, H, W)
        else:
            spatial_cond = m_dm  # (1, H, W)
        
        if self.use_param_conditioning:
            return x, spatial_cond, conditions
        else:
            return x, spatial_cond


# ============================================================================
# Lightning Module for ScoreModel
# ============================================================================

class LightScoreModel(LightningModule):
    """
    PyTorch Lightning wrapper for score_models.ScoreModel.
    
    Provides:
    - Training with Denoising Score Matching
    - Validation with score computation
    - Sampling interface compatible with BIND pipeline
    - Checkpointing and logging
    
    Args:
        model: Neural network (DDPM, NCSNpp, or custom)
        sde: SDE type ("vp" for VP-SDE, "ve" for VE-SDE)
        beta_min: Minimum beta for VP-SDE (default: 0.1)
        beta_max: Maximum beta for VP-SDE (default: 20.0)
        sigma_min: Minimum sigma for VE-SDE (default: 0.01)
        sigma_max: Maximum sigma for VE-SDE (default: 50.0)
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        lr_scheduler: Learning rate scheduler type
        ema_decay: EMA decay factor (default: 0.9999)
        n_sampling_steps: Number of steps for sampling (default: 250)
    """
    
    def __init__(
        self,
        model: nn.Module,
        sde: str = "vp",
        # VP-SDE parameters
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        # VE-SDE parameters  
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        ema_decay: float = 0.9999,
        # Sampling parameters
        n_sampling_steps: int = 250,
        # Conditioning
        use_param_conditioning: bool = False,  # Whether to pass astro params to model
        # Device
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        
        if not SCORE_MODELS_AVAILABLE:
            raise ImportError("score_models package required. Install with: pip install score_models")
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.ema_decay = ema_decay
        self.n_sampling_steps = n_sampling_steps
        self.sde_type = sde
        self.use_param_conditioning = use_param_conditioning
        
        # Build hyperparameters dict for ScoreModel
        hyperparameters = {"T": 1.0, "epsilon": 1e-5}
        
        if sde.lower() == "vp":
            hyperparameters["beta_min"] = beta_min
            hyperparameters["beta_max"] = beta_max
            self.sde = VPSDE(beta_min=beta_min, beta_max=beta_max, T=1.0, epsilon=1e-5)
        elif sde.lower() == "ve":
            hyperparameters["sigma_min"] = sigma_min
            hyperparameters["sigma_max"] = sigma_max
            self.sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max, T=1.0)
        else:
            raise ValueError(f"Unknown SDE type: {sde}. Use 'vp' or 've'.")
        
        # Create ScoreModel wrapper
        # Note: We pass sde as string, not SDE object, due to score_models internal handling
        # Note: We don't use ScoreModel's fit() method, we use Lightning's training loop
        self.score_model = ScoreModel(model=model, sde=sde.lower(), device=device, **hyperparameters)
        self.model = self.score_model.model  # Direct access to network
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        print(f"\n{'='*60}")
        print(f"INITIALIZED LIGHT SCORE MODEL")
        print(f"{'='*60}")
        print(f"  SDE type: {sde.upper()}")
        if sde.lower() == "vp":
            print(f"  Beta range: [{beta_min}, {beta_max}]")
        else:
            print(f"  Sigma range: [{sigma_min}, {sigma_max}]")
        print(f"  Learning rate: {learning_rate}")
        print(f"  EMA decay: {ema_decay}")
        print(f"  Sampling steps: {n_sampling_steps}")
        print(f"  Param conditioning: {use_param_conditioning}")
        print(f"{'='*60}\n")
    
    def forward(self, t: Tensor, x: Tensor, *args) -> Tensor:
        """Forward pass through the score model."""
        return self.score_model.model(t, x, *args)
    
    def score(self, t: Tensor, x: Tensor, *args) -> Tensor:
        """Compute the score function s(t, x) = ∇_x log p_t(x)."""
        return self.score_model.score(t, x, *args)
    
    def loss_fn(self, x: Tensor, *args) -> Tensor:
        """
        Compute Denoising Score Matching loss.
        
        DSM loss: E_t E_{x_0} E_{ε} [ ||s_θ(x_t, t) - ∇_{x_t} log p(x_t|x_0)||^2 ]
        
        For VP-SDE: ∇ log p(x_t|x_0) = -ε / σ(t)
        """
        return self.score_model.loss_fn(x, *args)
    
    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Training step with DSM loss."""
        # Unpack batch based on format
        # AstroDataset format: (m_dm, large_scale, m_target, conditions)
        # Simple format: (target, spatial_cond) or (target, spatial_cond, params)
        if len(batch) == 2:
            x, spatial_cond = batch
            args = (spatial_cond,)
        elif len(batch) == 3:
            x, spatial_cond, params = batch
            args = (spatial_cond, params)
        elif len(batch) == 4:
            # Standard AstroDataset format
            m_dm, large_scale, m_target, conditions = batch
            x = m_target
            spatial_cond = torch.cat([m_dm, large_scale], dim=1)
            # Include astro params (conditions) if model supports vector conditioning
            if self.use_param_conditioning and conditions is not None:
                args = (spatial_cond, conditions)
            else:
                args = (spatial_cond,)
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        # Compute DSM loss
        loss = self.loss_fn(x, *args)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Validation step."""
        # Same unpacking as training
        if len(batch) == 2:
            x, spatial_cond = batch
            args = (spatial_cond,)
        elif len(batch) == 3:
            x, spatial_cond, params = batch
            args = (spatial_cond, params)
        elif len(batch) == 4:
            m_dm, large_scale, m_target, conditions = batch
            x = m_target
            spatial_cond = torch.cat([m_dm, large_scale], dim=1)
            # Include astro params (conditions) if model supports vector conditioning
            if self.use_param_conditioning and conditions is not None:
                args = (spatial_cond, conditions)
            else:
                args = (spatial_cond,)
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        # Compute loss
        loss = self.loss_fn(x, *args)
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[List[Tensor]] = None,
        steps: Optional[int] = None,
        verbose: bool = True,
    ) -> Tensor:
        """
        Generate samples using Euler-Maruyama discretization of the reverse SDE.
        
        Args:
            shape: Shape of samples to generate (B, C, H, W)
            condition: List of conditioning tensors (e.g., [spatial_cond])
            steps: Number of sampling steps (default: self.n_sampling_steps)
            verbose: Show progress bar
        
        Returns:
            samples: Generated samples tensor
        """
        if steps is None:
            steps = self.n_sampling_steps
        
        if condition is None:
            condition = []
        
        return self.score_model.sample(
            shape=shape,
            steps=steps,
            condition=condition,
        )
    
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
            param_conditioning: Optional parameter conditioning
            verbose: Show progress
        
        Returns:
            samples: Generated samples (B, C_out, H, W)
        """
        B, C_cond, H, W = conditioning.shape
        C_out = 3  # [DM, Gas, Stars]
        
        # Build condition list
        condition = [conditioning]
        if param_conditioning is not None:
            condition.append(param_conditioning)
        
        # Generate samples
        samples = self.sample(
            shape=(batch_size, C_out, H, W),
            condition=condition,
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
        elif self.lr_scheduler_type == "cosine_warmup":
            # Cosine with warmup
            def lr_lambda(step):
                warmup_steps = 1000
                if step < warmup_steps:
                    return step / warmup_steps
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (self.trainer.estimated_stepping_batches - warmup_steps)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # No scheduler
            return optimizer
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" if self.lr_scheduler_type == "cosine_warmup" else "epoch",
            }
        }


# ============================================================================
# Factory functions for common configurations
# ============================================================================

def create_ddpm_model(
    output_channels: int = 3,
    conditioning_channels: int = 4,
    nf: int = 128,
    ch_mult: Tuple[int, ...] = (1, 2, 2, 4),
    num_res_blocks: int = 2,
    attention: bool = True,
    dropout: float = 0.1,
    sde: str = "vp",
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    learning_rate: float = 1e-4,
    **kwargs,
) -> LightScoreModel:
    """
    Create a DDPM-based score model for VDM-BIND.
    
    Args:
        output_channels: Number of output channels (default: 3 for [DM, Gas, Stars])
        conditioning_channels: Number of conditioning channels (default: 4 = DM + 3 large-scale)
        nf: Base number of features
        ch_mult: Channel multipliers for each resolution
        num_res_blocks: Number of residual blocks per resolution
        attention: Use attention blocks
        dropout: Dropout rate
        sde: SDE type ("vp" or "ve")
        beta_min/max: VP-SDE parameters
        learning_rate: Learning rate
    
    Returns:
        LightScoreModel instance
    """
    if not SCORE_MODELS_AVAILABLE:
        raise ImportError("score_models package required. Install with: pip install score_models")
    
    # Create DDPM network with input conditioning
    # Note: DDPM in score_models expects channels = output channels
    # and conditioning is done via concatenation to input
    net = DDPM(
        channels=output_channels + conditioning_channels,  # Input = condition + noise
        dimensions=2,
        nf=nf,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attention=attention,
        dropout=dropout,
    )
    
    # Wrap the network to handle conditioning properly
    wrapped_net = ConditionedDDPMWrapper(
        net=net,
        output_channels=output_channels,
        conditioning_channels=conditioning_channels,
    )
    
    return LightScoreModel(
        model=wrapped_net,
        sde=sde,
        beta_min=beta_min,
        beta_max=beta_max,
        learning_rate=learning_rate,
        **kwargs,
    )


def create_ncsnpp_model(
    output_channels: int = 3,
    conditioning_channels: int = 4,
    nf: int = 128,
    ch_mult: Tuple[int, ...] = (1, 2, 2, 4),
    attention: bool = True,
    sde: str = "vp",
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    learning_rate: float = 1e-4,
    **kwargs,
) -> LightScoreModel:
    """
    Create an NCSNpp-based score model for VDM-BIND.
    
    NCSNpp is Yang Song's architecture from "Score-Based Generative Modeling 
    through Stochastic Differential Equations".
    
    Args:
        output_channels: Number of output channels
        conditioning_channels: Number of conditioning channels
        nf: Base number of features
        ch_mult: Channel multipliers
        attention: Use attention
        sde: SDE type
        beta_min/max: VP-SDE parameters
        learning_rate: Learning rate
    
    Returns:
        LightScoreModel instance
    """
    if not SCORE_MODELS_AVAILABLE:
        raise ImportError("score_models package required. Install with: pip install score_models")
    
    # NCSNpp supports "input" conditioning natively
    net = NCSNpp(
        channels=output_channels,
        dimensions=2,
        nf=nf,
        ch_mult=ch_mult,
        attention=attention,
        condition=["input"],
        condition_input_channels=conditioning_channels,
    )
    
    return LightScoreModel(
        model=net,
        sde=sde,
        beta_min=beta_min,
        beta_max=beta_max,
        learning_rate=learning_rate,
        **kwargs,
    )


# ============================================================================
# Wrapper for DDPM to handle conditioning
# ============================================================================

class ConditionedDDPMWrapper(nn.Module):
    """
    Wrapper around DDPM that handles input conditioning by concatenation.
    
    score_models' DDPM doesn't have built-in conditioning like NCSNpp,
    so we concatenate condition to input and let network output only target channels.
    """
    
    def __init__(
        self,
        net: nn.Module,
        output_channels: int,
        conditioning_channels: int,
    ):
        super().__init__()
        self.net = net
        self.output_channels = output_channels
        self.conditioning_channels = conditioning_channels
        
        # Store hyperparameters for score_models compatibility
        self.hyperparameters = getattr(net, 'hyperparameters', {})
        self.hyperparameters['output_channels'] = output_channels
        self.hyperparameters['conditioning_channels'] = conditioning_channels
    
    def forward(self, t: Tensor, x: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with conditioning.
        
        Args:
            t: Time tensor (B,)
            x: Noisy input (B, C_out, H, W)
            condition: Conditioning tensor (B, C_cond, H, W)
        
        Returns:
            output: Predicted score/noise (B, C_out, H, W)
        """
        if condition is not None:
            # Concatenate condition to input
            x_cond = torch.cat([x, condition], dim=1)
        else:
            # Pad with zeros if no condition provided
            B, C, H, W = x.shape
            zeros = torch.zeros(B, self.conditioning_channels, H, W, device=x.device, dtype=x.dtype)
            x_cond = torch.cat([x, zeros], dim=1)
        
        # Forward through network
        out = self.net(t, x_cond)
        
        # Only return output channels (network outputs same size as input)
        return out[:, :self.output_channels]


# ============================================================================
# Alias for backwards compatibility
# ============================================================================

# Alias for documentation consistency
LightDDPM = LightScoreModel