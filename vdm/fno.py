"""
Fourier Neural Operator (FNO) implementation for VDM-BIND.

Based on "Fourier Neural Operator for Parametric Partial Differential Equations"
(Li et al., 2021) https://arxiv.org/abs/2010.08895

Key concepts:
- Learn operators in Fourier space for global receptive field
- Efficient for PDEs and physics problems with periodic/smooth solutions
- Natural fit for cosmological fields (power spectrum is key statistic)

Advantages for astrophysical fields:
- Global receptive field captures large-scale correlations
- Spectral bias may help with smooth density fields
- Efficient O(N log N) complexity via FFT
- Resolution-invariant (can train at low-res, apply at high-res)

Architecture components:
- Lifting: Project input to high-dimensional feature space
- Fourier Layers: Spectral convolution + local linear transform
- Projection: Map features back to output space
- Time/conditioning injection via FiLM or addition

This implementation is designed to be compatible with ALL generative methods:
- VDM (noise prediction)
- Flow Matching / Interpolants (velocity prediction)
- Score Matching (score prediction)
- Consistency Models

Interface follows the standard backbone API:
"""

# Verbosity control
_VERBOSE = True

def set_verbose(verbose: bool) -> None:
    """Set verbosity for FNO initialization messages."""
    global _VERBOSE
    _VERBOSE = verbose

def get_verbose() -> bool:
    """Get current verbosity setting."""
    return _VERBOSE


import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


# =============================================================================
# Spectral Convolution Layer
# =============================================================================

class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer.
    
    Performs convolution in Fourier space, which is equivalent to
    a global convolution in physical space. This gives FNO its
    characteristic global receptive field.
    
    The operation:
    1. FFT: x -> x_ft (to frequency domain)
    2. Multiply: x_ft * W (learnable weights for each frequency mode)
    3. iFFT: back to physical space
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        modes1: Number of Fourier modes in first dimension
        modes2: Number of Fourier modes in second dimension
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of modes in x direction
        self.modes2 = modes2  # Number of modes in y direction
        
        # Xavier-style initialization for spectral weights
        # Standard deviation that preserves variance through the layer
        std = (2.0 / (in_channels + out_channels)) ** 0.5
        
        # Complex weights for each quadrant of Fourier space
        # Shape: (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(
            torch.complex(
                torch.randn(in_channels, out_channels, modes1, modes2) * std,
                torch.randn(in_channels, out_channels, modes1, modes2) * std
            )
        )
        self.weights2 = nn.Parameter(
            torch.complex(
                torch.randn(in_channels, out_channels, modes1, modes2) * std,
                torch.randn(in_channels, out_channels, modes1, modes2) * std
            )
        )
    
    def compl_mul2d(self, input: Tensor, weights: Tensor) -> Tensor:
        """
        Complex multiplication in Fourier space.
        
        Args:
            input: (B, in_channels, H, W) complex tensor
            weights: (in_channels, out_channels, H, W) complex tensor
        Returns:
            (B, out_channels, H, W) complex tensor
        """
        # Einstein summation for batched matrix multiplication
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) real tensor
        Returns:
            (B, C_out, H, W) real tensor
        """
        B, C, H, W = x.shape
        
        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x)
        
        # Prepare output tensor (complex)
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Multiply relevant Fourier modes
        # Upper-left quadrant (low frequencies)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Lower-left quadrant (high frequencies in first dim)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # Inverse FFT back to physical space
        x = torch.fft.irfft2(out_ft, s=(H, W))
        
        return x


# =============================================================================
# FNO Block
# =============================================================================

class FNOBlock(nn.Module):
    """
    Single FNO block combining spectral convolution with local linear transform.
    
    The block computes:
        x_out = σ(W_spectral(x) + W_local(x) + bias)
    
    where:
        - W_spectral: Spectral convolution (global, in Fourier space)
        - W_local: Pointwise linear transform (local)
        - σ: Activation function (GELU)
    
    This combination allows learning both global patterns (spectral) and
    local features (linear).
    
    Args:
        channels: Number of channels
        modes1: Fourier modes in dimension 1
        modes2: Fourier modes in dimension 2
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Spectral convolution (global)
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        
        # Local linear transform (1x1 convolution)
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Normalization and activation
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        # Parallel paths: spectral (global) + local
        x_spectral = self.spectral_conv(x)
        x_local = self.local_conv(x)
        
        # Combine and apply nonlinearity
        x = x_spectral + x_local
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class FNOBlockWithConditioning(nn.Module):
    """
    FNO block with time and parameter conditioning via FiLM.
    
    FiLM (Feature-wise Linear Modulation):
        x_out = γ(t, params) * FNO(x) + β(t, params)
    
    This allows the network to modulate its behavior based on:
    - Time t (for diffusion/flow models)
    - Physical parameters (cosmological params, etc.)
    
    Args:
        channels: Number of channels
        modes1: Fourier modes in dimension 1
        modes2: Fourier modes in dimension 2
        cond_dim: Dimension of conditioning embedding
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Core FNO operations
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # FiLM conditioning: project conditioning to scale and shift
        self.film = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )
        
        # Initialize FiLM to identity: gamma=1, beta=0
        # This ensures signal passes through unchanged at init
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        # Set gamma bias to 1 (first half of output)
        with torch.no_grad():
            self.film[-1].bias[:channels] = 1.0
        
        # Normalization
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            cond: (B, cond_dim) conditioning embedding
        Returns:
            (B, C, H, W)
        """
        # Save residual
        residual = x
        
        # FNO forward
        x_spectral = self.spectral_conv(x)
        x_local = self.local_conv(x)
        x = x_spectral + x_local
        x = self.norm(x)
        
        # FiLM modulation
        film_params = self.film(cond)  # (B, 2*C)
        gamma, beta = film_params.chunk(2, dim=-1)  # Each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        
        x = gamma * x + beta
        x = self.activation(x)
        x = self.dropout(x)
        
        # Residual connection for better gradient flow
        x = x + residual
        
        return x


# =============================================================================
# Time Embedding
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding following DDPM/VDM convention.
    
    Maps scalar time t ∈ [0, 1] to a high-dimensional embedding
    using sinusoidal positional encoding.
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # MLP to project sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) time values in [0, 1]
        Returns:
            (B, dim) time embeddings
        """
        # Scale to match VDM convention (they multiply by 1000)
        t = t * 1000
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Project through MLP
        embedding = self.mlp(embedding)
        
        return embedding


# =============================================================================
# Parameter Embedding
# =============================================================================

class ParameterEmbedding(nn.Module):
    """
    Embed physical parameters (cosmological, astrophysical) into conditioning space.
    
    Normalizes parameters using provided min/max bounds, then projects to
    embedding dimension.
    """
    
    def __init__(
        self,
        n_params: int,
        embed_dim: int,
        param_min: Optional[List[float]] = None,
        param_max: Optional[List[float]] = None,
    ):
        super().__init__()
        self.n_params = n_params
        self.embed_dim = embed_dim
        
        # Register normalization bounds
        if param_min is not None:
            self.register_buffer('param_min', torch.tensor(param_min, dtype=torch.float32))
        else:
            self.register_buffer('param_min', torch.zeros(n_params))
        
        if param_max is not None:
            self.register_buffer('param_max', torch.tensor(param_max, dtype=torch.float32))
        else:
            self.register_buffer('param_max', torch.ones(n_params))
        
        # MLP to project parameters
        self.mlp = nn.Sequential(
            nn.Linear(n_params, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, params: Tensor) -> Tensor:
        """
        Args:
            params: (B, n_params) physical parameters
        Returns:
            (B, embed_dim) parameter embedding
        """
        # Normalize to [0, 1]
        params_norm = (params - self.param_min) / (self.param_max - self.param_min + 1e-8)
        # Project
        return self.mlp(params_norm)


# =============================================================================
# Full FNO Model
# =============================================================================

class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for generative modeling.
    
    Architecture:
    1. Lifting: Project input + conditioning to hidden channels
    2. FNO Blocks: N layers of spectral convolution with conditioning
    3. Projection: Map back to output channels
    
    Compatible with all generative methods (VDM, Flow, Consistency, etc.)
    via the standard backbone interface.
    
    Args:
        in_channels: Input channels (for x_t, the noisy/interpolated data)
        out_channels: Output channels (noise, velocity, or score prediction)
        conditioning_channels: Spatial conditioning channels (e.g., DM map)
        large_scale_channels: Additional large-scale context channels
        hidden_channels: Width of hidden layers
        n_layers: Number of FNO blocks
        modes1: Fourier modes in first spatial dimension
        modes2: Fourier modes in second spatial dimension
        n_params: Number of conditioning parameters (0 for unconditional)
        param_min: Min values for parameter normalization
        param_max: Max values for parameter normalization
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes1: int = 16,
        modes2: int = 16,
        n_params: int = 35,
        param_min: Optional[List[float]] = None,
        param_max: Optional[List[float]] = None,
        dropout: float = 0.0,
        use_param_conditioning: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditioning_channels = conditioning_channels
        self.large_scale_channels = large_scale_channels
        self.total_conditioning = conditioning_channels + large_scale_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.n_params = n_params
        self.use_param_conditioning = use_param_conditioning and n_params > 0
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(hidden_channels)
        
        # Parameter embedding (optional)
        if self.use_param_conditioning:
            self.param_embed = ParameterEmbedding(
                n_params, hidden_channels, param_min, param_max
            )
            cond_dim = hidden_channels * 2  # time + params
        else:
            self.param_embed = None
            cond_dim = hidden_channels  # time only
        
        # Lifting: project input to hidden dimension
        # Input = x_t (noisy data) + conditioning (DM map + large-scale)
        total_input = in_channels + self.total_conditioning
        self.lifting = nn.Sequential(
            nn.Conv2d(total_input, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
        )
        
        # FNO blocks with conditioning
        self.blocks = nn.ModuleList([
            FNOBlockWithConditioning(
                channels=hidden_channels,
                modes1=modes1,
                modes2=modes2,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Skip connections (optional, can help with gradients)
        self.use_skip = True
        if self.use_skip:
            self.skip_conv = nn.Conv2d(total_input, hidden_channels, kernel_size=1)
        
        # Projection: map back to output dimension
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        
        # Initialize projection layers with proper scaling
        # First layer: standard initialization
        nn.init.kaiming_normal_(self.projection[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.projection[0].bias)
        # Final layer: scale to produce unit variance output
        # With hidden_channels inputs, use std = 1/sqrt(hidden_channels)
        nn.init.normal_(self.projection[-1].weight, mean=0.0, std=(1.0 / hidden_channels) ** 0.5)
        nn.init.zeros_(self.projection[-1].bias)
        
        # Print model info
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model configuration."""
        if not get_verbose():
            return
        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print("INITIALIZED FNO2d")
        print(f"{'='*60}")
        print(f"  Input channels: {self.in_channels}")
        print(f"  Output channels: {self.out_channels}")
        print(f"  Conditioning channels: {self.total_conditioning}")
        print(f"  Hidden channels: {self.hidden_channels}")
        print(f"  FNO layers: {self.n_layers}")
        print(f"  Param conditioning: {self.use_param_conditioning} ({self.n_params} params)")
        print(f"  Total parameters: {n_params:,}")
        print(f"{'='*60}\n")
    
    def forward(
        self,
        t: Tensor,
        x_t: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Standard backbone interface for generative models.
        
        Args:
            t: (B,) Time values in [0, 1]
            x_t: (B, in_channels, H, W) Noisy/interpolated data
            conditioning: (B, cond_channels, H, W) Spatial conditioning
            param_conditioning: (B, n_params) Optional parameter conditioning
        
        Returns:
            (B, out_channels, H, W) Predicted noise/velocity/score
        """
        B, C, H, W = x_t.shape
        
        # Build conditioning embedding
        t_emb = self.time_embed(t)  # (B, hidden)
        
        if self.use_param_conditioning:
            if param_conditioning is not None:
                p_emb = self.param_embed(param_conditioning)  # (B, hidden)
            else:
                # If param_conditioning is None but model expects it, use zeros
                p_emb = torch.zeros(B, self.hidden_channels, device=t_emb.device, dtype=t_emb.dtype)
            cond_emb = torch.cat([t_emb, p_emb], dim=-1)  # (B, 2*hidden)
        else:
            cond_emb = t_emb  # (B, hidden)
        
        # Concatenate input and spatial conditioning
        x = torch.cat([x_t, conditioning], dim=1)  # (B, in+cond, H, W)
        
        # Skip connection input
        if self.use_skip:
            skip = self.skip_conv(x)
        
        # Lifting
        x = self.lifting(x)
        
        # FNO blocks
        for block in self.blocks:
            x = block(x, cond_emb)
        
        # Add skip connection
        if self.use_skip:
            x = x + skip
        
        # Projection to output
        x = self.projection(x)
        
        return x


# =============================================================================
# Model Variants (following DiT convention)
# =============================================================================

def FNO_S(img_size: int = 64, **kwargs) -> FNO2d:
    """FNO-Small: 32 hidden, 4 layers, ~200K params"""
    return FNO2d(
        hidden_channels=32,
        n_layers=4,
        modes1=min(12, img_size // 4),
        modes2=min(12, img_size // 4),
        **kwargs
    )


def FNO_B(img_size: int = 64, **kwargs) -> FNO2d:
    """FNO-Base: 64 hidden, 6 layers, ~800K params"""
    return FNO2d(
        hidden_channels=64,
        n_layers=6,
        modes1=min(16, img_size // 4),
        modes2=min(16, img_size // 4),
        **kwargs
    )


def FNO_L(img_size: int = 64, **kwargs) -> FNO2d:
    """FNO-Large: 128 hidden, 8 layers, ~3M params"""
    return FNO2d(
        hidden_channels=128,
        n_layers=8,
        modes1=min(24, img_size // 4),
        modes2=min(24, img_size // 4),
        **kwargs
    )


def FNO_XL(img_size: int = 64, **kwargs) -> FNO2d:
    """FNO-XL: 256 hidden, 12 layers, ~12M params"""
    return FNO2d(
        hidden_channels=256,
        n_layers=12,
        modes1=min(32, img_size // 4),
        modes2=min(32, img_size // 4),
        **kwargs
    )


FNO_VARIANTS = {
    'FNO-S': FNO_S,
    'FNO-B': FNO_B,
    'FNO-L': FNO_L,
    'FNO-XL': FNO_XL,
}


def create_fno_model(
    variant: str = 'FNO-B',
    img_size: int = 64,
    n_params: int = 35,
    conditioning_channels: int = 1,
    large_scale_channels: int = 3,
    param_min: Optional[List[float]] = None,
    param_max: Optional[List[float]] = None,
    dropout: float = 0.0,
    **kwargs,
) -> FNO2d:
    """
    Create an FNO model variant.
    
    Args:
        variant: Model variant ('FNO-S', 'FNO-B', 'FNO-L', 'FNO-XL')
        img_size: Input image size
        n_params: Number of conditioning parameters
        conditioning_channels: DM conditioning channels
        large_scale_channels: Large-scale context channels
        param_min: Parameter normalization min values
        param_max: Parameter normalization max values
        dropout: Dropout rate
    
    Returns:
        FNO2d model
    """
    if variant not in FNO_VARIANTS:
        available = list(FNO_VARIANTS.keys())
        raise ValueError(f"Unknown variant: {variant}. Available: {available}")
    
    return FNO_VARIANTS[variant](
        img_size=img_size,
        in_channels=3,
        out_channels=3,
        conditioning_channels=conditioning_channels,
        large_scale_channels=large_scale_channels,
        n_params=n_params,
        param_min=param_min,
        param_max=param_max,
        dropout=dropout,
        **kwargs,
    )


# =============================================================================
# Wrapper for VDM-style interface (gamma instead of t)
# =============================================================================

class FNOForVDM(nn.Module):
    """
    Wrapper that adapts FNO to VDM's gamma-based interface.
    
    VDM uses gamma (log-SNR) instead of t. This wrapper converts
    gamma to an equivalent time-like value for the FNO backbone.
    """
    
    def __init__(
        self,
        fno: FNO2d,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
    ):
        super().__init__()
        self.fno = fno
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
    
    def forward(
        self,
        x_t: Tensor,
        gamma: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        VDM-compatible interface.
        
        Args:
            x_t: (B, C, H, W) Noisy data
            gamma: (B,) or (B, 1) Log-SNR values
            conditioning: (B, cond_channels, H, W) Spatial conditioning
            param_conditioning: (B, n_params) Optional parameters
        
        Returns:
            (B, C, H, W) Predicted noise
        """
        # Convert gamma to t ∈ [0, 1]
        # gamma = gamma_max - (gamma_max - gamma_min) * t
        # => t = (gamma_max - gamma) / (gamma_max - gamma_min)
        gamma = gamma.squeeze(-1) if gamma.dim() > 1 else gamma
        t = (self.gamma_max - gamma) / (self.gamma_max - self.gamma_min)
        t = t.clamp(0, 1)
        
        return self.fno(t, x_t, conditioning, param_conditioning)


# =============================================================================
# Utility: Hybrid FNO-UNet (optional advanced architecture)
# =============================================================================

class FNOUNetHybrid(nn.Module):
    """
    Hybrid architecture combining FNO (global) with UNet (local).
    
    The idea is to use FNO for capturing global/spectral features and
    UNet for local/multi-scale features. This could combine the best
    of both worlds for astrophysical data.
    
    Architecture:
    - FNO branch: Global spectral features
    - UNet branch: Local multi-scale features
    - Fusion: Combine via addition or concatenation
    
    Note: This is an experimental architecture for future exploration.
    """
    
    def __init__(
        self,
        fno: FNO2d,
        unet: nn.Module,  # Any UNet with compatible interface
        fusion: str = 'add',  # 'add' or 'concat'
    ):
        super().__init__()
        self.fno = fno
        self.unet = unet
        self.fusion = fusion
        
        if fusion == 'concat':
            # Need projection to combine doubled channels
            out_channels = fno.out_channels
            self.proj = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        else:
            self.proj = None
    
    def forward(
        self,
        t: Tensor,
        x_t: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward through both branches and fuse."""
        # FNO branch (global)
        out_fno = self.fno(t, x_t, conditioning, param_conditioning)
        
        # UNet branch (local)
        out_unet = self.unet(t, x_t, conditioning, param_conditioning)
        
        # Fuse
        if self.fusion == 'add':
            return out_fno + out_unet
        else:
            combined = torch.cat([out_fno, out_unet], dim=1)
            return self.proj(combined)
