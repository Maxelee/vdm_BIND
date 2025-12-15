"""
Diffusion Transformer (DiT) implementation for VDM-BIND.

Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
https://arxiv.org/abs/2212.09748

Key components:
- Patchify: Convert image to sequence of patches
- DiT Blocks: Transformer blocks with adaLN-Zero conditioning
- Unpatchify: Convert patches back to image

The DiT architecture uses adaptive layer normalization (adaLN) to incorporate
timestep and class conditioning, which has been shown to work better than
in-context conditioning or cross-attention for diffusion models.
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


# =============================================================================
# Patch Embedding
# =============================================================================

class PatchEmbed(nn.Module):
    """
    Convert 2D image into sequence of patch embeddings.
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch (assumes square)
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        bias: Whether to use bias in projection
    """
    
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=bias
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}x{W}) doesn't match expected ({self.img_size}x{self.img_size})"
        
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# =============================================================================
# Timestep & Conditioning Embeddings
# =============================================================================

class TimestepEmbedder(nn.Module):
    """
    Embed scalar timesteps into vector representations.
    Uses sinusoidal position embeddings followed by MLP.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: (B,) tensor of timesteps
            dim: Embedding dimension
            max_period: Controls minimum frequency
        Returns:
            (B, dim) embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ParamEmbedder(nn.Module):
    """
    Embed cosmological/astrophysical parameters.
    Supports 0 to N parameters (0 = unconditional).
    """
    
    def __init__(self, n_params: int, hidden_size: int):
        super().__init__()
        self.n_params = n_params
        
        if n_params > 0:
            self.mlp = nn.Sequential(
                nn.Linear(n_params, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        else:
            # Unconditional: output zeros
            self.mlp = None
    
    def forward(self, params: Optional[Tensor]) -> Tensor:
        if self.mlp is None or params is None:
            return None
        return self.mlp(params)


# =============================================================================
# Adaptive Layer Norm (adaLN-Zero)
# =============================================================================

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT block with adaptive layer norm zero (adaLN-Zero).
    
    The conditioning (timestep + optional params) is used to predict
    scale, shift, and gate parameters for the layer norms.
    
    Args:
        hidden_size: Transformer hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim multiplier
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )
        
        # adaLN-Zero: predict 6 parameters (shift, scale, gate for each of 2 sublayers)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, D) - sequence of patch embeddings
            c: (B, D) - conditioning embedding (timestep + params)
        Returns:
            (B, N, D) - updated embeddings
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention block with modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # MLP block with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP with GELU activation."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# Final Layer
# =============================================================================

class FinalLayer(nn.Module):
    """
    Final layer of DiT with adaLN-Zero modulation.
    Projects back to patch space.
    """
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# =============================================================================
# Positional Embeddings
# =============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> Tensor:
    """
    Generate 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Height/width of patch grid
    Returns:
        (grid_size*grid_size, embed_dim) position embeddings
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: Tensor) -> Tensor:
    """Get 2D sincos position embedding from a grid."""
    assert embed_dim % 2 == 0
    
    # Use half of dimensions for each axis
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: Tensor) -> Tensor:
    """Get 1D sincos position embedding."""
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


# =============================================================================
# DiT Model
# =============================================================================

class DiT(nn.Module):
    """
    Diffusion Transformer for image generation.
    
    This implementation is adapted for the VDM-BIND use case:
    - Input: 3-channel target (DM_hydro, Gas, Stars) + conditioning map
    - Conditioning: Timestep + optional cosmological parameters
    - Output: 3-channel prediction (noise, velocity, or score)
    
    Args:
        img_size: Input image size
        patch_size: Patch size (4, 8, or 16 are common)
        in_channels: Input channels (typically 3 for target + conditioning)
        out_channels: Output channels (3 for [DM_hydro, Gas, Stars])
        hidden_size: Transformer hidden dimension
        depth: Number of DiT blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = hidden_size * mlp_ratio
        n_params: Number of conditioning parameters (0 = unconditional)
        dropout: Dropout rate
        conditioning_channels: Number of spatial conditioning channels (DM + large-scale)
    """
    
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        n_params: int = 35,
        dropout: float = 0.0,
        conditioning_channels: int = 1,
        large_scale_channels: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.n_params = n_params
        self.conditioning_channels = conditioning_channels
        self.large_scale_channels = large_scale_channels
        
        # Total input channels: target + conditioning
        total_in_channels = in_channels + conditioning_channels + large_scale_channels
        
        # Patch embedding
        self.x_embedder = PatchEmbed(img_size, patch_size, total_in_channels, hidden_size)
        self.num_patches = self.x_embedder.num_patches
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Parameter embedding (optional)
        self.y_embedder = ParamEmbedder(n_params, hidden_size)
        
        # Positional embedding (fixed, sinusoidal)
        pos_embed = get_2d_sincos_pos_embed(hidden_size, self.x_embedder.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        
        # Initialize weights
        self.initialize_weights()
        
        # Print model info
        print(f"\n{'='*60}")
        print(f"INITIALIZED DiT MODEL")
        print(f"{'='*60}")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Patch size: {patch_size}")
        print(f"  Num patches: {self.num_patches}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Depth: {depth}")
        print(f"  Heads: {num_heads}")
        print(f"  MLP ratio: {mlp_ratio}")
        print(f"  Parameters: {n_params} (0=unconditional)")
        print(f"  Conditioning: {conditioning_channels} + {large_scale_channels} large-scale")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total_params:,}")
        print(f"{'='*60}\n")
    
    def initialize_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        # Initialize patch embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Initialize param embedding MLP if present
        if self.y_embedder.mlp is not None:
            nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)
        
        # Initialize DiT blocks
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            # Zero-init adaLN modulation
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-init final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x: Tensor) -> Tensor:
        """
        Convert patch sequence back to image.
        
        Args:
            x: (B, num_patches, patch_size**2 * out_channels)
        Returns:
            (B, out_channels, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.x_embedder.grid_size
        
        x = x.reshape(B := x.shape[0], h, w, p, p, c)
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(B, c, h * p, w * p)
        return x
    
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        conditioning: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of DiT.
        
        Args:
            x: (B, C, H, W) - input (noisy target)
            t: (B,) - timestep or gamma values
            conditioning: (B, C_cond, H, W) - spatial conditioning (DM + large-scale)
            param_conditioning: (B, N_params) - cosmological parameters
        
        Returns:
            (B, out_channels, H, W) - predicted noise/velocity/score
        """
        # Concatenate input with conditioning along channel dimension
        if conditioning is not None:
            x = torch.cat([x, conditioning], dim=1)
        
        # Patchify and embed
        x = self.x_embedder(x) + self.pos_embed
        
        # Compute conditioning embedding
        t_emb = self.t_embedder(t)
        
        if param_conditioning is not None and self.y_embedder.mlp is not None:
            y_emb = self.y_embedder(param_conditioning)
            c = t_emb + y_emb
        else:
            c = t_emb
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final projection
        x = self.final_layer(x, c)
        
        # Unpatchify
        x = self.unpatchify(x)
        
        return x


# =============================================================================
# DiT Model Variants (following original paper naming)
# =============================================================================

def DiT_S_4(**kwargs) -> DiT:
    """DiT-S/4 (small, patch 4)"""
    return DiT(patch_size=4, hidden_size=384, depth=12, num_heads=6, **kwargs)

def DiT_S_8(**kwargs) -> DiT:
    """DiT-S/8 (small, patch 8)"""
    return DiT(patch_size=8, hidden_size=384, depth=12, num_heads=6, **kwargs)

def DiT_B_4(**kwargs) -> DiT:
    """DiT-B/4 (base, patch 4)"""
    return DiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, **kwargs)

def DiT_B_8(**kwargs) -> DiT:
    """DiT-B/8 (base, patch 8)"""
    return DiT(patch_size=8, hidden_size=768, depth=12, num_heads=12, **kwargs)

def DiT_L_4(**kwargs) -> DiT:
    """DiT-L/4 (large, patch 4)"""
    return DiT(patch_size=4, hidden_size=1024, depth=24, num_heads=16, **kwargs)

def DiT_L_8(**kwargs) -> DiT:
    """DiT-L/8 (large, patch 8)"""
    return DiT(patch_size=8, hidden_size=1024, depth=24, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs) -> DiT:
    """DiT-XL/4 (extra large, patch 4)"""
    return DiT(patch_size=4, hidden_size=1152, depth=28, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs) -> DiT:
    """DiT-XL/8 (extra large, patch 8)"""
    return DiT(patch_size=8, hidden_size=1152, depth=28, num_heads=16, **kwargs)


# Model registry
DIT_MODELS = {
    'DiT-S/4': DiT_S_4,
    'DiT-S/8': DiT_S_8,
    'DiT-B/4': DiT_B_4,
    'DiT-B/8': DiT_B_8,
    'DiT-L/4': DiT_L_4,
    'DiT-L/8': DiT_L_8,
    'DiT-XL/4': DiT_XL_4,
    'DiT-XL/8': DiT_XL_8,
}


def create_dit_model(model_name: str, **kwargs) -> DiT:
    """Create a DiT model by name."""
    if model_name not in DIT_MODELS:
        raise ValueError(f"Unknown DiT model: {model_name}. Available: {list(DIT_MODELS.keys())}")
    return DIT_MODELS[model_name](**kwargs)
