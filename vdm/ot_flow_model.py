"""
Optimal Transport Flow Matching Model for VDM-BIND.

This module implements Optimal Transport (OT) Flow Matching following 
Lipman et al. (2022) "Flow Matching for Generative Modeling".

Key concepts:
- Standard flow matching uses LINEAR interpolation: x_t = (1-t)x_0 + t*x_1
- OT flow matching uses OPTIMAL TRANSPORT coupling: x_t = OT(x_0, x_1, t)
- OT paths are straighter → better sample quality, especially for structured data
- Mini-batch OT approximation for practical training

Optimal Transport Path:
- Finds optimal pairing between x_0 (source) and x_1 (target) samples
- Minimizes transport cost (e.g., Wasserstein distance)
- Results in straighter, more direct paths through data space

Advantages over linear flow matching:
- Straighter interpolation paths
- Better sample quality for structured data
- May converge faster
- Particularly beneficial for astronomical data with complex structure

Trade-offs:
- More expensive to train (OT computation per mini-batch)
- Requires OT solver (we use POT library)
- Mini-batch OT is an approximation

Integration with VDM-BIND:
- Uses same AstroDataset and normalization
- Compatible with existing UNet architecture
- Supports multi-scale conditioning
- Works with BIND inference pipeline

Reference:
    Lipman et al. (2022) "Flow Matching for Generative Modeling"
    https://arxiv.org/abs/2210.02747

Usage:
    from vdm.ot_flow_model import LightOTFlow
    
    # Create model
    model = LightOTFlow(
        velocity_model=unet,
        learning_rate=1e-4,
        n_sampling_steps=50,
        ot_method='exact',  # or 'sinkhorn' for entropic OT
    )
    
    # Training
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

# Try to import POT (Python Optimal Transport)
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("Warning: POT (Python Optimal Transport) not installed. "
          "Install with: pip install POT")


# ============================================================================
# Optimal Transport Utilities
# ============================================================================

def compute_ot_plan(
    x0: Tensor,
    x1: Tensor,
    method: str = 'exact',
    reg: float = 0.01,
    numItermax: int = 1000,
) -> Tensor:
    """
    Compute optimal transport plan between x0 and x1 samples.
    
    Args:
        x0: Source samples (B, C, H, W)
        x1: Target samples (B, C, H, W)
        method: OT method - 'exact' (EMD) or 'sinkhorn' (entropic)
        reg: Regularization for Sinkhorn (entropic OT)
        numItermax: Max iterations for OT solver
    
    Returns:
        pi: Transport plan (B, B) - permutation/coupling matrix
    """
    if not HAS_POT:
        # Fallback: identity coupling (no OT)
        return torch.eye(x0.shape[0], device=x0.device)
    
    B = x0.shape[0]
    device = x0.device
    
    # Flatten spatial dimensions for distance computation
    x0_flat = x0.reshape(B, -1).detach().cpu().numpy()  # (B, C*H*W)
    x1_flat = x1.reshape(B, -1).detach().cpu().numpy()  # (B, C*H*W)
    
    # Compute cost matrix (squared Euclidean distance)
    M = ot.dist(x0_flat, x1_flat, metric='sqeuclidean')
    
    # Normalize cost matrix for numerical stability
    M = M / M.max()
    
    # Uniform marginals (equal weight per sample)
    a = np.ones(B) / B
    b = np.ones(B) / B
    
    # Compute OT plan
    if method == 'exact':
        # Exact OT (EMD) - returns optimal permutation
        pi = ot.emd(a, b, M, numItermax=numItermax)
    elif method == 'sinkhorn':
        # Entropic OT (Sinkhorn) - smoother, faster
        pi = ot.sinkhorn(a, b, M, reg=reg, numItermax=numItermax)
    else:
        raise ValueError(f"Unknown OT method: {method}")
    
    # Convert to tensor
    pi = torch.from_numpy(pi).float().to(device)
    
    return pi


def apply_ot_coupling(
    x0: Tensor,
    x1: Tensor,
    pi: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Apply OT coupling to reorder samples.
    
    For each x0[i], find the best matching x1[j] according to the OT plan.
    
    Args:
        x0: Source samples (B, C, H, W)
        x1: Target samples (B, C, H, W)
        pi: Transport plan (B, B)
    
    Returns:
        x0_matched: Reordered x0 (kept same)
        x1_matched: Reordered x1 to match x0
    """
    B = x0.shape[0]
    
    # For exact OT, pi is approximately a permutation matrix
    # Find the best match for each x0 sample
    # pi[i, j] = amount of mass from x0[i] to x1[j]
    # We take the argmax for hard assignment
    assignment = pi.argmax(dim=1)  # (B,) indices into x1
    
    # Reorder x1 according to OT assignment
    x1_matched = x1[assignment]
    
    return x0, x1_matched


def sample_ot_pairs(
    x0: Tensor,
    x1: Tensor,
    method: str = 'exact',
    reg: float = 0.01,
) -> Tuple[Tensor, Tensor]:
    """
    Sample paired (x0, x1) using optimal transport coupling.
    
    Args:
        x0: Source samples (B, C, H, W)
        x1: Target samples (B, C, H, W)
        method: OT method
        reg: Regularization for Sinkhorn
    
    Returns:
        x0_paired: Source samples (unchanged order)
        x1_paired: Target samples (reordered by OT)
    """
    pi = compute_ot_plan(x0, x1, method=method, reg=reg)
    return apply_ot_coupling(x0, x1, pi)


# ============================================================================
# OT Flow Matching Core
# ============================================================================

class OTInterpolant(nn.Module):
    """
    Optimal Transport Flow Matching interpolant.
    
    Unlike standard flow matching which uses random pairing,
    OT flow matching pairs x0 and x1 samples optimally to minimize
    transport cost, resulting in straighter paths.
    
    The velocity field is still v = x_1 - x_0, but the pairing
    is determined by OT rather than random batch ordering.
    
    Args:
        velocity_model: Neural network that predicts velocity
        ot_method: OT method ('exact' or 'sinkhorn')
        ot_reg: Regularization for Sinkhorn
        use_stochastic_interpolant: Add noise during interpolation
        sigma: Noise scale for stochastic interpolant
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        ot_method: str = 'exact',
        ot_reg: float = 0.01,
        use_stochastic_interpolant: bool = False,
        sigma: float = 0.0,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.ot_method = ot_method
        self.ot_reg = ot_reg
        self.use_stochastic_interpolant = use_stochastic_interpolant
        self.sigma = sigma
    
    def get_mu_t(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Compute mean of interpolant at time t.
        
        Linear interpolation: mu_t = t * x_1 + (1 - t) * x_0
        """
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        return t * x1 + (1 - t) * x0
    
    def get_sigma_t(self, t: Tensor) -> Tensor:
        """Noise scale for stochastic interpolant."""
        t = t.view(t.shape[0], *([1] * 3))
        return self.sigma * torch.sqrt(2 * t * (1 - t))
    
    def sample_xt(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Sample x_t from the interpolant distribution."""
        mu_t = self.get_mu_t(x0, x1, t)
        
        if self.use_stochastic_interpolant and self.sigma > 0:
            sigma_t = self.get_sigma_t(t)
            epsilon = torch.randn_like(mu_t)
            return mu_t + sigma_t * epsilon
        
        return mu_t
    
    def get_velocity(self, x0: Tensor, x1: Tensor) -> Tensor:
        """True velocity: v = x_1 - x_0"""
        return x1 - x0
    
    def compute_loss(
        self,
        x0: Tensor,
        x1: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        use_ot: bool = True,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute OT flow matching loss.
        
        1. Apply OT coupling to pair x0 with x1 optimally
        2. Compute interpolant x_t
        3. Compute loss on velocity prediction
        
        Args:
            x0: Source samples (B, C, H, W)
            x1: Target samples (B, C, H, W)
            conditioning: Spatial conditioning
            param_conditioning: Parameter conditioning
            t: Time (optional)
            use_ot: Whether to use OT coupling
        
        Returns:
            loss: Scalar loss
            metrics: Dict with additional metrics
        """
        metrics = {}
        
        # Apply OT coupling
        if use_ot and self.training:
            x0_paired, x1_paired = sample_ot_pairs(
                x0, x1,
                method=self.ot_method,
                reg=self.ot_reg,
            )
        else:
            # During eval or if OT disabled, use random pairing
            x0_paired, x1_paired = x0, x1
        
        # Sample time
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        
        # Sample x_t
        x_t = self.sample_xt(x0_paired, x1_paired, t)
        
        # True velocity
        v_true = self.get_velocity(x0_paired, x1_paired)
        
        # Predicted velocity
        v_pred = self.velocity_model(t, x_t, conditioning, param_conditioning)
        
        if isinstance(v_pred, tuple):
            v_pred = v_pred[0]
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_true)
        
        # Compute path straightness metric (for logging)
        if use_ot:
            # Path length: ||x1 - x0|| - straighter paths should be shorter on average
            path_length = torch.norm(x1_paired - x0_paired, dim=(1, 2, 3)).mean()
            metrics['path_length'] = path_length.item()
        
        return loss, metrics
    
    @torch.no_grad()
    def sample(
        self,
        x0: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
        n_steps: int = 50,
        return_trajectory: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Generate samples by integrating the ODE.
        
        Note: Sampling doesn't use OT (no target distribution available).
        OT is only used during training to create better paths.
        """
        dt = 1.0 / n_steps
        x = x0.clone()
        
        trajectory = [x.clone()] if return_trajectory else None
        
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            
            v = self.velocity_model(t, x, conditioning, param_conditioning)
            if isinstance(v, tuple):
                v = v[0]
            
            x = x + v * dt
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return trajectory
        
        return x


# ============================================================================
# Velocity Network Wrapper
# ============================================================================

class OTVelocityNetWrapper(nn.Module):
    """
    Wrapper around UNet for OT flow matching velocity prediction.
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
        """Forward pass."""
        output = self.net(x, t, conditioning=conditioning, param_conditioning=param_conditioning)
        
        if isinstance(output, tuple):
            return output[0]
        return output


# ============================================================================
# Lightning Module
# ============================================================================

class LightOTFlow(LightningModule):
    """
    PyTorch Lightning wrapper for OT Flow Matching.
    
    Provides:
    - Training with OT-paired flow matching loss
    - Optional comparison with non-OT (random) pairing
    - Sampling interface compatible with BIND
    - Logging of path straightness metrics
    
    Args:
        velocity_model: Neural network for velocity prediction
        learning_rate: Learning rate
        weight_decay: Weight decay
        lr_scheduler: Scheduler type
        n_sampling_steps: Sampling steps
        ot_method: OT method ('exact' or 'sinkhorn')
        ot_reg: Regularization for Sinkhorn
        use_stochastic_interpolant: Add noise during interpolation
        sigma: Noise scale
        use_param_conditioning: Use parameter conditioning
        x0_mode: How to initialize x0 ('zeros', 'noise', 'dm_copy')
        use_ot_training: Use OT during training
        ot_warmup_epochs: Epochs before enabling OT (optional warmup)
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
        # OT parameters
        ot_method: str = 'exact',
        ot_reg: float = 0.01,
        # Interpolant parameters
        use_stochastic_interpolant: bool = False,
        sigma: float = 0.0,
        # Conditioning
        use_param_conditioning: bool = False,
        # x0 initialization
        x0_mode: str = "zeros",
        # Training mode
        use_ot_training: bool = True,
        ot_warmup_epochs: int = 0,
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.n_sampling_steps = n_sampling_steps
        self.use_param_conditioning = use_param_conditioning
        self.x0_mode = x0_mode
        self.use_ot_training = use_ot_training
        self.ot_warmup_epochs = ot_warmup_epochs
        
        # Create OT interpolant
        self.interpolant = OTInterpolant(
            velocity_model=velocity_model,
            ot_method=ot_method,
            ot_reg=ot_reg,
            use_stochastic_interpolant=use_stochastic_interpolant,
            sigma=sigma,
        )
        
        self.velocity_model = velocity_model
        
        self.save_hyperparameters(ignore=['velocity_model'])
        
        print(f"\n{'='*60}")
        print(f"INITIALIZED OT FLOW MATCHING MODEL")
        print(f"{'='*60}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Sampling steps: {n_sampling_steps}")
        print(f"  OT method: {ot_method}")
        if ot_method == 'sinkhorn':
            print(f"  OT regularization: {ot_reg}")
        print(f"  Stochastic: {use_stochastic_interpolant} (sigma={sigma})")
        print(f"  x0 mode: {x0_mode}")
        print(f"  Use OT training: {use_ot_training}")
        if ot_warmup_epochs > 0:
            print(f"  OT warmup epochs: {ot_warmup_epochs}")
        print(f"  POT available: {HAS_POT}")
        print(f"{'='*60}\n")
    
    def _get_x0(self, x1: Tensor, dm_condition: Optional[Tensor] = None) -> Tensor:
        """Initialize x0."""
        B, C, H, W = x1.shape
        
        if self.x0_mode == "zeros":
            return torch.zeros_like(x1)
        elif self.x0_mode == "noise":
            return torch.randn_like(x1)
        elif self.x0_mode == "dm_copy":
            if dm_condition is not None:
                return dm_condition.expand(-1, C, -1, -1).clone()
            else:
                return torch.zeros_like(x1)
        else:
            raise ValueError(f"Unknown x0_mode: {self.x0_mode}")
    
    def _unpack_batch(self, batch: Tuple) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Unpack batch from AstroDataset."""
        m_dm, large_scale, m_target, conditions = batch
        
        x1 = m_target
        conditioning = torch.cat([m_dm, large_scale], dim=1)
        dm_condition = m_dm
        params = conditions if self.use_param_conditioning else None
        
        return x1, conditioning, dm_condition, params
    
    def _should_use_ot(self) -> bool:
        """Determine if OT should be used this epoch."""
        if not self.use_ot_training:
            return False
        if self.current_epoch < self.ot_warmup_epochs:
            return False
        return True
    
    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Training step."""
        x1, conditioning, dm_condition, params = self._unpack_batch(batch)
        
        # Initialize x0
        x0 = self._get_x0(x1, dm_condition)
        
        # Determine if using OT
        use_ot = self._should_use_ot()
        
        # Compute loss
        loss, metrics = self.interpolant.compute_loss(
            x0=x0,
            x1=x1,
            conditioning=conditioning,
            param_conditioning=params,
            use_ot=use_ot,
        )
        
        # Log
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/use_ot", float(use_ot), on_step=False, on_epoch=True, sync_dist=True)
        
        if 'path_length' in metrics:
            self.log("train/path_length", metrics['path_length'], on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Validation step."""
        x1, conditioning, dm_condition, params = self._unpack_batch(batch)
        
        x0 = self._get_x0(x1, dm_condition)
        
        # Always use OT for validation if enabled
        use_ot = self.use_ot_training
        
        loss, metrics = self.interpolant.compute_loss(
            x0=x0,
            x1=x1,
            conditioning=conditioning,
            param_conditioning=params,
            use_ot=use_ot,
        )
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if 'path_length' in metrics:
            self.log("val/path_length", metrics['path_length'], on_epoch=True, sync_dist=True)
        
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
        """Generate samples."""
        if steps is None:
            steps = self.n_sampling_steps
        
        device = conditioning.device if conditioning is not None else self.device
        B, C_out, H, W = shape
        
        # Initialize x0
        if self.x0_mode == "noise":
            x0 = torch.randn(shape, device=device)
        elif self.x0_mode == "dm_copy":
            if conditioning is not None:
                dm_condition = conditioning[:, :1]
                x0 = dm_condition.expand(-1, C_out, -1, -1).clone()
            else:
                x0 = torch.zeros(shape, device=device)
        else:
            x0 = torch.zeros(shape, device=device)
        
        # Generate samples
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
        """BIND-compatible sampling interface."""
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

def create_ot_flow_model(
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
    ot_method: str = 'exact',
    ot_reg: float = 0.01,
    x0_mode: str = "zeros",
    **kwargs,
) -> LightOTFlow:
    """
    Create an OT flow matching model.
    
    Args:
        output_channels: Number of output channels (3)
        conditioning_channels: Number of conditioning channels (4)
        embedding_dim: Embedding dimension
        n_blocks: Number of residual blocks
        norm_groups: Groups for GroupNorm
        n_attention_heads: Attention heads
        learning_rate: Learning rate
        n_sampling_steps: Sampling steps
        use_fourier_features: Use Fourier features
        fourier_legacy: Use legacy Fourier
        add_attention: Add attention
        ot_method: OT method ('exact' or 'sinkhorn')
        ot_reg: Sinkhorn regularization
        x0_mode: x0 initialization
    
    Returns:
        LightOTFlow instance
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
    
    velocity_model = OTVelocityNetWrapper(
        net=unet,
        output_channels=output_channels,
        conditioning_channels=conditioning_channels,
    )
    
    return LightOTFlow(
        velocity_model=velocity_model,
        learning_rate=learning_rate,
        n_sampling_steps=n_sampling_steps,
        ot_method=ot_method,
        ot_reg=ot_reg,
        x0_mode=x0_mode,
        **kwargs,
    )


# ============================================================================
# Alias
# ============================================================================

LightOptimalTransportFlow = LightOTFlow
