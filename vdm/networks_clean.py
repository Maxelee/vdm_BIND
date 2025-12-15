import numpy as np
import torch
from torch import einsum, nn, pi, softmax
from vdm.utils import zero_init

def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    timesteps *= 1000
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        base=10.0,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)

def attention_inner_heads(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension.

    Args:
        qkv: Tensor of shape (B, 3*H*C, T) with Qs, Ks, and Vs, where:
            H = number of heads,
            C = number of channels per head.
        num_heads: number of heads.

    Returns:
        Attention output of shape (B, H*C, T).
    """

    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) of shape (B, H*C, T).
    q, k, v = qkv.chunk(3, dim=1)

    # Rescale q and k. This makes them contiguous in memory.
    scale = ch ** (-1 / 4)  # scale with 4th root = scaling output by sqrt
    q = q * scale
    k = k * scale

    # Reshape qkv to (B*H, C, T).
    new_shape = (bs * num_heads, ch, length)
    q = q.view(*new_shape)
    k = k.view(*new_shape)
    v = v.reshape(*new_shape)

    # Compute attention.
    weight = einsum("bct,bcs->bts", q, k)  # (B*H, T, T)
    weight = softmax(weight.float(), dim=-1).to(weight.dtype)  # (B*H, T, T)
    out = einsum("bts,bcs->bct", weight, v)  # (B*H, C, T)
    return out.reshape(bs, num_heads * ch, length)  # (B, H*C, T)

class Attention(nn.Module):
    """Based on https://github.com/openai/guided-diffusion."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        assert qkv.dim() >= 3, qkv.dim()
        assert qkv.shape[1] % (3 * self.n_heads) == 0
        spatial_dims = qkv.shape[2:]
        qkv = qkv.view(*qkv.shape[:2], -1)  # (B, 3*H*C, T)
        out = attention_inner_heads(qkv, self.n_heads)  # (B, H*C, T)
        return out.view(*out.shape[:2], *spatial_dims)


class AttentionBlock(nn.Module):
    """Self-attention residual block."""

    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        assert n_channels % n_heads == 0
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            nn.Conv2d(n_channels, 3 * n_channels , kernel_size=1),  # (B, 3 * C, H, W)
            Attention(n_heads),
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        return self.layers(x) + x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between feature map and conditioning.
    
    Allows the model to selectively attend to conditioning information (dark matter,
    large-scale fields) when predicting baryonic properties.
    
    Args:
        n_heads: Number of attention heads
        n_channels: Number of feature channels
        cond_channels: Number of conditioning channels
        norm_groups: Number of groups for GroupNorm
        use_chunked_attention: Use memory-efficient chunked attention (default: True)
        chunk_size: Chunk size for chunked attention (default: 512)
    """
    
    def __init__(self, n_heads, n_channels, cond_channels, norm_groups, 
                 use_chunked_attention=True, chunk_size=512, dropout=0.1,
                 downsample_cond=True, cond_downsample_factor=4):
        super().__init__()
        self.n_heads = n_heads
        self.use_chunked_attention = use_chunked_attention
        self.chunk_size = chunk_size
        self.downsample_cond = downsample_cond
        self.cond_downsample_factor = cond_downsample_factor
        assert n_channels % n_heads == 0
        
        # Normalization (adjust norm_groups if cond_channels is too small)
        self.norm_features = nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels)
        # For conditioning, find largest divisor of cond_channels that's <= norm_groups
        cond_norm_groups = norm_groups
        while cond_channels % cond_norm_groups != 0 and cond_norm_groups > 1:
            cond_norm_groups -= 1
        self.norm_cond = nn.GroupNorm(num_groups=cond_norm_groups, num_channels=cond_channels)
        
        # Q from features, K/V from conditioning
        self.to_q = nn.Conv2d(n_channels, n_channels, kernel_size=1)
        self.to_k = nn.Conv2d(cond_channels, n_channels, kernel_size=1)
        self.to_v = nn.Conv2d(cond_channels, n_channels, kernel_size=1)
        
        # Output projection with configurable dropout
        self.to_out = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # CRITICAL FIX: Use small random initialization instead of zero
        # Zero initialization prevents learning - the network never escapes the local minimum
        # Small random initialization allows gradients to flow and the model to learn
        nn.init.normal_(self.to_out[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.to_out[0].bias)
    
    def _chunked_attention(self, q, k, v):
        """
        Compute attention in chunks to save memory.
        
        Args:
            q: Query (B, n_heads, head_dim, HW)
            k: Key (B, n_heads, head_dim, HW_cond)
            v: Value (B, n_heads, head_dim, HW_cond)
        
        Returns:
            Attention output (B, n_heads, head_dim, HW)
        """
        B, n_heads, head_dim, HW = q.shape
        HW_cond = k.shape[-1]
        
        scale = head_dim ** -0.5
        out_chunks = []
        
        for i in range(0, HW, self.chunk_size):
            q_chunk = q[:, :, :, i:i+self.chunk_size]  # (B, n_heads, head_dim, chunk_size)
            
            # Attention over all conditioning
            attn = torch.einsum('bhdn,bhdm->bhnm', q_chunk, k) * scale  # (B, n_heads, chunk_size, HW_cond)
            attn = torch.softmax(attn, dim=-1)
            
            # Apply to values
            out_chunk = torch.einsum('bhnm,bhdm->bhdn', attn, v)  # (B, n_heads, head_dim, chunk_size)
            out_chunks.append(out_chunk)
        
        return torch.cat(out_chunks, dim=-1)
    
    def forward(self, x, conditioning, return_attention_stats=False):
        """
        Args:
            x: Feature map (B, C, H, W)
            conditioning: Conditioning (B, C_cond, H_cond, W_cond)
            return_attention_stats: If True, return attention statistics for logging
        
        Returns:
            Attended features (B, C, H, W)
            If return_attention_stats=True, also returns dict with attention metrics
        """
        B, C, H, W = x.shape
        _, C_cond, H_cond, W_cond = conditioning.shape
        
        # Store statistics if requested
        stats = {} if return_attention_stats else None
        
        # Normalize
        x_norm = self.norm_features(x)
        cond_norm = self.norm_cond(conditioning)
        
        # OPTIMIZATION 1: Downsample conditioning to reduce K/V size (4-16x speedup!)
        if self.downsample_cond and H_cond > 32:
            target_size = max(32, H_cond // self.cond_downsample_factor)
            cond_norm = torch.nn.functional.interpolate(
                cond_norm,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
            H_cond, W_cond = target_size, target_size
        
        # Project to Q, K, V
        q = self.to_q(x_norm)  # (B, C, H, W)
        k = self.to_k(cond_norm)  # (B, C, H_cond, W_cond) - now potentially downsampled!
        v = self.to_v(cond_norm)  # (B, C, H_cond, W_cond)
        
        # Reshape for multi-head attention
        # (B, C, H, W) -> (B, n_heads, C//n_heads, H*W)
        head_dim = C // self.n_heads
        q = q.reshape(B, self.n_heads, head_dim, H * W)
        k = k.reshape(B, self.n_heads, head_dim, H_cond * W_cond)
        v = v.reshape(B, self.n_heads, head_dim, H_cond * W_cond)
        
        # OPTIMIZATION 2: Use PyTorch's efficient scaled_dot_product_attention (Flash Attention)
        # This is 2-4x faster and uses less memory than manual implementation
        # Requires PyTorch >= 2.0
        try:
            # Transpose to (B, n_heads, seq_len, head_dim) format for SDPA
            q = q.transpose(2, 3)  # (B, n_heads, H*W, head_dim)
            k = k.transpose(2, 3)  # (B, n_heads, H_cond*W_cond, head_dim)
            v = v.transpose(2, 3)  # (B, n_heads, H_cond*W_cond, head_dim)
            
            # Use PyTorch's optimized attention (includes Flash Attention when available)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,  # Dropout handled in to_out
                is_causal=False
            )  # (B, n_heads, H*W, head_dim)
            
            # Transpose back
            out = out.transpose(2, 3)  # (B, n_heads, head_dim, H*W)
            
        except (AttributeError, RuntimeError):
            # Fallback for PyTorch < 2.0 or if SDPA fails
            # Need to restore original shape if tensors were transposed
            if q.shape[2] > q.shape[3]:  # Already transposed to (B, n_heads, H*W, head_dim)
                q = q.transpose(2, 3)  # Back to (B, n_heads, head_dim, H*W)
                k = k.transpose(2, 3)  # Back to (B, n_heads, head_dim, H_cond*W_cond)
                v = v.transpose(2, 3)  # Back to (B, n_heads, head_dim, H_cond*W_cond)
            
            # Use chunked attention for memory efficiency
            if self.use_chunked_attention:
                out = self._chunked_attention(q, k, v)
            else:
                # Standard attention
                scale = head_dim ** -0.5
                attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
                attn = torch.softmax(attn, dim=-1)
                out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        
        # Reshape back to spatial dimensions
        out = out.reshape(B, C, H, W)
        
        # Output projection
        out = self.to_out(out)
        
        # Collect statistics if requested
        if return_attention_stats:
            with torch.no_grad():
                # Compute attention magnitude (how much the features changed)
                feature_change = (out - x).abs().mean().item()
                feature_norm = x.abs().mean().item()
                relative_change = feature_change / (feature_norm + 1e-8)
                
                # Conditioning statistics
                cond_mean = conditioning.mean().item()
                cond_std = conditioning.std().item()
                
                stats = {
                    'feature_change': feature_change,
                    'relative_change': relative_change,
                    'conditioning_mean': cond_mean,
                    'conditioning_std': cond_std,
                    'feature_spatial_res': (H, W),
                    'cond_spatial_res': (H_cond, W_cond),
                }
        
        # Residual connection
        output = x + out
        
        if return_attention_stats:
            return output, stats
        return output


class UpDownBlock(nn.Module):
    def __init__(self, resnet_block, attention_block=None):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention_block = attention_block

    def forward(self, x, cond):
        x = self.resnet_block(x, cond)
        if self.attention_block is not None:
            x = self.attention_block(x)
        return x

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for stronger conditioning"""
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        # Predict both scale (gamma) and shift (beta)
        self.film_proj = nn.Linear(condition_dim, feature_dim * 2)
        # Initialize with small random values to enable gradient flow
        # Small std (0.01) keeps behavior near identity but allows learning
        nn.init.normal_(self.film_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.film_proj.bias)
        
    def forward(self, features, condition):
        # features: (B, C, H, W)
        # condition: (B, condition_dim)
        film_params = self.film_proj(condition)  # (B, C*2)
        gamma, beta = film_params.chunk(2, dim=1)  # (B, C) each
        
        # Add 1 to gamma so default behavior is identity
        gamma = gamma + 1.0
        
        # Reshape for broadcasting
        gamma = gamma[:, :, None, None]  # (B, C, 1, 1)
        beta = beta[:, :, None, None]
        
        # Apply FiLM: scale and shift
        return gamma * features + beta


class ResnetBlock(nn.Module):
    def __init__(self,ch_in,ch_out=None,condition_dim=None,dropout_prob=0.0,norm_groups=32,use_film=True):
        super().__init__()
        ch_out = ch_in if ch_out is None else ch_out
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.condition_dim = condition_dim
        self.use_film = use_film
        
        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1,padding_mode="zeros") ,
        )
        
        if condition_dim is not None:
            if use_film:
                # Use FiLM layer for stronger conditioning
                self.film_layer = FiLMLayer(condition_dim, ch_out)
                self.cond_proj = None  # Not used with FiLM
            else:
                # Original projection-based conditioning
                self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))
                self.film_layer = None
        else:
            self.cond_proj = None
            self.film_layer = None
            
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1,padding_mode="zeros")),
        )
        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, condition):
        h = self.net1(x)
        
        if condition is not None:
            assert condition.shape == (x.shape[0], self.condition_dim), \
                f"Expected condition shape ({x.shape[0]}, {self.condition_dim}), got {condition.shape}"
            
            if self.use_film and self.film_layer is not None:
                # Apply FiLM modulation
                h = self.film_layer(h, condition)
            elif self.cond_proj is not None:
                # Original projection-based conditioning
                condition_proj = self.cond_proj(condition)
                condition_proj = condition_proj[:, :, None, None]  # 2d
                h = h + condition_proj
                
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        assert x.shape == h.shape
        return x + h

class DownBlock(nn.Module):
    def __init__(self, resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.down=nn.Conv2d(self.resnet_block.ch_out,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, cond, spatial_cond=None):
        # spatial_cond ignored for regular DownBlock (for compatibility)
        xskip = self.resnet_block(x, cond)
        x=self.down(xskip)
        return x,xskip


class DownBlockWithCrossAttention(nn.Module):
    """Down block with optional cross-attention for conditioning."""
    def __init__(self, resnet_block, cross_attn_block=None):
        super().__init__()
        self.resnet_block = resnet_block
        self.down = nn.Conv2d(self.resnet_block.ch_out, self.resnet_block.ch_out, 2, stride=2)
        self.cross_attn_block = cross_attn_block
    
    def forward(self, x, cond, spatial_cond=None):
        """
        Args:
            x: Feature map (B, C, H, W)
            cond: Time/param conditioning (B, cond_dim)
            spatial_cond: Spatial conditioning for cross-attention (B, C_cond, H, W)
        """
        # Apply ResNet block
        xskip = self.resnet_block(x, cond)
        
        # Apply cross-attention if available
        if self.cross_attn_block is not None and spatial_cond is not None:
            xskip = self.cross_attn_block(xskip, spatial_cond)
        
        # Downsample
        x = self.down(xskip)
        return x, xskip


class UpBlock(nn.Module):
    def __init__(self, resnet_block, upsample_ch):
        super().__init__()
        self.resnet_block = resnet_block
        # Create upsampling layer that reduces channels by half
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(upsample_ch, upsample_ch // 2, kernel_size=1)
        )

    def forward(self, x, xskip, cond):
        # x: current feature map from previous layer
        # xskip: skip connection from corresponding down block
        
        # Upsample and reduce channels
        x_up = self.upsample(x)
        
        # Concatenate upsampled features with skip connection
        x_concat = torch.cat([x_up, xskip], dim=1)
        
        # Apply ResNet block
        x = self.resnet_block(x_concat, cond)
        return x


class UpBlockWithCrossAttention(nn.Module):
    """Up block with optional cross-attention for conditioning."""
    def __init__(self, resnet_block, upsample_ch, cross_attn_block=None):
        super().__init__()
        self.resnet_block = resnet_block
        self.cross_attn_block = cross_attn_block
        # Create upsampling layer that reduces channels by half
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(upsample_ch, upsample_ch // 2, kernel_size=1)
        )

    def forward(self, x, xskip, cond, spatial_cond=None):
        # x: current feature map from previous layer
        # xskip: skip connection from corresponding down block
        
        # Upsample and reduce channels
        x_up = self.upsample(x)
        
        # Concatenate upsampled features with skip connection
        x_concat = torch.cat([x_up, xskip], dim=1)
        
        # Apply ResNet block
        x = self.resnet_block(x_concat, cond)
        
        # Apply cross-attention if available
        if self.cross_attn_block is not None and spatial_cond is not None:
            x = self.cross_attn_block(x, spatial_cond)
        
        return x

class FourierFeatures(nn.Module):
    """
    Fourier feature encoder with configurable frequency bands.
    
    Supports two modes:
    - New mode (default): Uses explicit frequency list [1, 2, 4, 8] * π, applies to all channels
    - Legacy mode: Uses exponential range 2^(first:last:step) * 2π, applies only to first channel
    
    Args:
        frequencies: List of frequencies in units of π (e.g., [1, 2, 4, 8] means [π, 2π, 4π, 8π])
        first: Legacy mode - first exponent for 2^n (e.g., -2.0)
        last: Legacy mode - last exponent for 2^n (e.g., 1.0)
        step: Legacy mode - step between exponents (e.g., 1.0)
        legacy_mode: If True, use old behavior for backward compatibility
        
    For scale-appropriate frequency bands (new mode):
    - Halo (m_dm): High frequencies [1, 2, 4, 8] -> wavelengths 24-195 kpc at 6.25 Mpc scale
    - Large-scale: Low frequencies [0.25, 0.5, 1, 2] -> wavelengths 49-391 kpc at 12.5 Mpc scale
    """
    def __init__(self, frequencies=None, first=None, last=None, step=None, legacy_mode=False):
        super().__init__()
        
        # Determine mode based on parameters
        if legacy_mode or (first is not None and last is not None):
            # Legacy mode: exponential frequency range for backward compatibility
            self.legacy_mode = True
            if first is None:
                first = -2.0
            if last is None:
                last = 1.0
            if step is None:
                step = 1.0
            # Register as buffer so it's saved in state dict
            self.register_buffer('freqs_exponent', torch.arange(first, last + 1e-8, step))
        else:
            # New mode: explicit frequency list
            self.legacy_mode = False
            if frequencies is None:
                frequencies = [1, 2, 4, 8]  # Default high frequencies
            self.register_buffer('frequencies', torch.tensor(frequencies, dtype=torch.float32))

    @property
    def num_features(self):
        # Return number of features: 2 per frequency (sin and cos)
        if self.legacy_mode:
            return len(self.freqs_exponent) * 2
        else:
            return len(self.frequencies) * 2

    def forward(self, x):
        """
        Apply Fourier features to input.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Legacy mode: Fourier features of shape (B, num_features, H, W) - only first channel
            New mode: Fourier features of shape (B, num_features*C, H, W) - all channels
        """
        assert len(x.shape) >= 2

        if self.legacy_mode:
            # Legacy behavior: exponential frequencies, only first channel
            # Compute (2π * 2^n) for n in freqs.
            freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
            freqs = 2.0**freqs_exponent * 2 * pi  # (F, )
            freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

            # Apply Fourier features to first channel only
            x_for_fourier = x[:, :1]  # (B, 1, H, W)
            
            # Compute (2π * 2^n * x) for n in freqs.
            features = freqs * x_for_fourier.unsqueeze(1)  # (B, F, 1, H, W)
            features = features.flatten(1, 2)  # (B, F, H, W)

            # Output features are cos and sin of above. Shape (B, 2 * F, H, W).
            return torch.cat([features.sin(), features.cos()], dim=1)
        else:
            # New behavior: explicit frequencies, all channels
            # Compute (f * π) for f in frequencies
            freqs = self.frequencies.to(dtype=x.dtype, device=x.device) * pi  # (F, )
            freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

            # Apply Fourier features to each channel
            # x shape: (B, C, H, W)
            # Compute (f * π * x) for f in frequencies
            features = freqs * x.unsqueeze(1)  # (B, F, C, H, W)
            features = features.flatten(1, 2)  # (B, F*C, H, W)

            # Output features are cos and sin. Shape (B, 2*F*C, H, W).
            return torch.cat([features.sin(), features.cos()], dim=1)

class OmegaMEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, omega_m):
        if omega_m.dim() == 1:
            omega_m = omega_m.unsqueeze(-1)  # [batch] -> [batch, 1]
        return self.embedding(omega_m)

class HaloMassEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Typical halo mass range: 1e10 to 1e15 M_sun
        # In your units (1e10 M_sun): 1 to 1e5
        # Log10 range: 0 to 5
        self.log_min = 0.0  # log10(1) = 0
        self.log_max = 5.0  # log10(1e5) = 5
        
        self.embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, halo_mass):
        if halo_mass.dim() == 1:
            halo_mass = halo_mass.unsqueeze(-1)  # [batch] -> [batch, 1]
        
        # Take log10 and normalize to [0, 1]
        log_mass = torch.log10(halo_mass + 1e-8)  # Add small epsilon to avoid log(0)
        normalized_mass = (log_mass - self.log_min) / (self.log_max - self.log_min)
        
        # Clamp to [0, 1] range for safety
        normalized_mass = torch.clamp(normalized_mass, 0.0, 1.0)
        
        return self.embedding(normalized_mass)

class ASN1Embedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.log_min = -0.045173997906125286
        self.log_max = 1.1583323317089809
        
        self.embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, ASN1):
        if ASN1.dim() == 1:
            ASN1 = ASN1.unsqueeze(-1)  # [batch] -> [batch, 1]
        
        # Take log10 and normalize to [0, 1]
        log_ASN1 = torch.log10(ASN1)  
        normalized_ASN1 = (log_ASN1 - self.log_min) / (self.log_max - self.log_min)
        
        # Clamp to [0, 1] range for safety
        normalized_ASN1 = torch.clamp(normalized_ASN1, 0.0, 1.0)
        
        return self.embedding(normalized_ASN1)
    
class ParamEmbedding(nn.Module):
    """
    Parameter conditioning embedding for flexible cosmological/astrophysical parameters.
    
    Supports:
    - n_params = 0: Returns zero embedding (unconditional generation)
    - n_params > 0: Normalizes and embeds parameters
    
    Args:
        embed_dim: Output embedding dimension
        param_min: Minimum values for each parameter (list/array) or None for unconditional
        param_max: Maximum values for each parameter (list/array) or None for unconditional
        
    Example:
        # Unconditional (n_params=0)
        embed = ParamEmbedding(192, None, None)
        
        # 6 custom parameters
        embed = ParamEmbedding(192, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1])
        
        # 35 CAMELS parameters (original)
        embed = ParamEmbedding(192, min_vals, max_vals)
    """
    def __init__(self, embed_dim, param_min, param_max):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Handle unconditional case (n_params = 0)
        if param_min is None or param_max is None or len(param_min) == 0:
            self.Nparams = 0
            self.min = None
            self.max = None
            self.embedding = None
            print(f"⚙️  ParamEmbedding: Unconditional mode (n_params=0)")
        else:
            self.min = torch.tensor(param_min, dtype=torch.float32)
            self.max = torch.tensor(param_max, dtype=torch.float32)
            self.Nparams = len(self.min)
            
            if len(self.min) != len(self.max):
                raise ValueError(f"param_min ({len(self.min)}) and param_max ({len(self.max)}) must have same length")
            
            # IMPROVED: Deeper, stronger network with normalization
            self.embedding = nn.Sequential(
                nn.Linear(self.Nparams, embed_dim * 2),  # Expand
                nn.LayerNorm(embed_dim * 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim),  # Project to final dim
                nn.LayerNorm(embed_dim),
            )
            print(f"⚙️  ParamEmbedding: Conditional mode (n_params={self.Nparams})")
    
    def forward(self, conditional_params):
        """
        Embed conditional parameters.
        
        Args:
            conditional_params: (B, N_params) or (N_params,) tensor, or None for unconditional
        
        Returns:
            (B, embed_dim) embedding tensor
        """
        # Handle unconditional case
        if self.Nparams == 0 or conditional_params is None:
            # Return zeros - the model will use only time conditioning
            if conditional_params is not None:
                batch_size = conditional_params.shape[0] if conditional_params.dim() > 1 else 1
            else:
                batch_size = 1
            device = conditional_params.device if conditional_params is not None else 'cpu'
            return torch.zeros(batch_size, self.embed_dim, device=device)
        
        if conditional_params.dim() == 1:
            conditional_params = conditional_params.unsqueeze(0)
        
        # Validate parameter count
        if conditional_params.shape[-1] != self.Nparams:
            raise ValueError(
                f"Expected {self.Nparams} parameters, got {conditional_params.shape[-1]}. "
                f"Ensure your data and config specify the same number of parameters."
            )
        
        min_vals = self.min.to(conditional_params.device)
        max_vals = self.max.to(conditional_params.device)
        
        # Normalize to [0, 1]
        normalized_conditional_params = (conditional_params - min_vals) / (max_vals - min_vals + 1e-8)
        normalized_conditional_params = torch.clamp(normalized_conditional_params, 0.0, 1.0)
        
        return self.embedding(normalized_conditional_params)

class UNetVDM(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        conditioning_channels: int = 1,
        large_scale_channels: int = 1,  # NEW: Number of large-scale field channels
        embedding_dim: int=48,
        n_blocks: int = 4,  
        norm_groups: int = 8,
        n_attention_heads: int = 1,
        add_attention: bool = False,
        attention_everywhere: bool = False,
        use_fourier_features: bool = False,
        legacy_fourier: bool = False,  # NEW: Use old Fourier features for backward compatibility
        dropout_prob: float = 0.1,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        add_downsampling: bool = True,
        use_param_conditioning: bool = False,
        use_param_prediction: bool = True,
        param_min: list = None,
        param_max: list = None,
        use_auxiliary_mask: bool = False,  # NEW: Enable auxiliary mask head
        # Cross-attention parameters
        use_cross_attention: bool = False,  # NEW: Enable cross-attention
        cross_attention_heads: int = 8,  # NEW: Number of cross-attention heads  
        use_chunked_cross_attention: bool = True,  # NEW: Use memory-efficient chunked attention
        cross_attention_chunk_size: int = 512,  # NEW: Chunk size for attention
        cross_attention_dropout: float = 0.1,  # NEW: Dropout for attention (Phase 2)
        cross_attention_location: str = 'bottleneck',  # NEW: 'bottleneck', 'decoder', 'full', 'everywhere' (Phase 2)
        # Speed optimizations
        downsample_cross_attn_cond: bool = True,  # NEW: Downsample conditioning for cross-attention (4-8x speedup)
        cross_attn_cond_downsample_factor: int = 2,  # NEW: Downsample factor for conditioning
        cross_attn_max_resolution: int = 128,  # NEW: Maximum resolution for cross-attention (skip higher res)
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_fourier_features = use_fourier_features
        self.legacy_fourier = legacy_fourier
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.add_attention = add_attention
        self.add_downsampling = add_downsampling
        self.use_param_conditioning = use_param_conditioning
        self.use_param_prediction = use_param_prediction  # ← ADD THIS
        self.use_cross_attention = use_cross_attention
        self.cross_attention_heads = cross_attention_heads
        self.cross_attention_dropout = cross_attention_dropout  # Phase 2
        self.cross_attention_location = cross_attention_location  # Phase 2
        self.downsample_cross_attn_cond = downsample_cross_attn_cond  # Speed optimization
        self.cross_attn_cond_downsample_factor = cross_attn_cond_downsample_factor  # Speed optimization
        self.cross_attn_max_resolution = cross_attn_max_resolution  # Speed optimization



        # Calculate input channels
        # With cross-attention: only target (conditioning is separate)
        # Without cross-attention: target (3) + m_dm (1) + large_scale (N)
        if use_cross_attention:
            # Cross-attention: only process target, conditioning is separate
            base_input_ch = input_channels
            print(f"✓ Cross-attention mode: Input channels = {input_channels} (target only)")
        else:
            # Original mode: concatenate everything
            # conditioning_channels = m_dm channels = 1
            # large_scale_channels = number of large-scale fields = N (can be 1 or more)
            base_input_ch = input_channels + conditioning_channels + large_scale_channels
        
        # Store channel counts for Fourier feature application
        self.input_channels = input_channels
        self.conditioning_channels = conditioning_channels
        self.large_scale_channels = large_scale_channels
        
        if use_fourier_features:
            if legacy_fourier:
                # Legacy mode: single Fourier feature module applied to all concatenated inputs
                # Uses exponential frequencies (2^n * 2π) and only processes first channel
                self.fourier_features = FourierFeatures(
                    first=-2.0,
                    last=1.0,
                    step=1.0,
                    legacy_mode=True
                )
                self.fourier_features_halo = None
                self.fourier_features_largescale = None
                
                # Legacy: Apply to all concatenated inputs (returns features from first channel only)
                total_input_ch = base_input_ch + self.fourier_features.num_features
                
                print(f"✓ Legacy Fourier features enabled (backward compatibility):")
                print(f"  - Exponential frequencies: 2^(-2:1) * 2π")
                print(f"  - Applied to first channel of concatenated input")
                print(f"  - {self.fourier_features.num_features} Fourier features")
                print(f"  - Total input channels: {total_input_ch}")
            else:
                # New mode: scale-appropriate frequency bands for different inputs
                # High frequencies for halo (m_dm) - captures small-scale structure
                # Low frequencies for large-scale - captures environmental modulation
                # No Fourier features for target - learns naturally across all frequencies
                
                self.fourier_features = None  # Not used in new mode
                self.fourier_features_halo = FourierFeatures(frequencies=[1, 2, 4, 8])
                self.fourier_features_largescale = FourierFeatures(frequencies=[0.25, 0.5, 1, 2])
                
                # Calculate total input channels with Fourier features
                # Cross-attention mode: only target goes through conv_in
                # Baseline mode: target + conditioning with Fourier features
                if use_cross_attention:
                    # Only target channels (conditioning handled separately via cross-attention)
                    total_input_ch = input_channels  # Just 3 for target, no Fourier on target
                else:
                    # Original: Target + m_dm + large_scale (all with appropriate Fourier)
                    total_input_ch = (
                        input_channels +  # Target (no Fourier)
                        conditioning_channels * (1 + self.fourier_features_halo.num_features) +  # m_dm with high-freq Fourier
                        large_scale_channels * (1 + self.fourier_features_largescale.num_features)  # large_scale with low-freq Fourier
                    )
                
                print(f"✓ Scale-appropriate Fourier features enabled:")
                print(f"  - Halo (m_dm): High frequencies [1, 2, 4, 8]π -> {self.fourier_features_halo.num_features} features per channel")
                print(f"  - Large-scale: Low frequencies [0.25, 0.5, 1, 2]π -> {self.fourier_features_largescale.num_features} features per channel")
                print(f"  - Target: No Fourier features (learns naturally)")
                print(f"  - Total input channels: {total_input_ch}")
        else:
            self.fourier_features = None
            self.fourier_features_halo = None
            self.fourier_features_largescale = None
            total_input_ch = base_input_ch
    
        # Calculate conditioning dimension based on enabled features
        condition_dim = 4 * embedding_dim  # Base time conditioning
        
        if use_param_conditioning:
            # FIXED: Make parameter embedding MUCH STRONGER
            # Use 4x embedding_dim to match time embedding strength
            param_embed_dim = 4 * embedding_dim
            self.param_conditioning_embedding = ParamEmbedding(
                param_embed_dim, param_min, param_max
            )
            condition_dim += param_embed_dim
            
            # Add learnable scales to balance time vs parameter conditioning
            self.time_scale = nn.Parameter(torch.ones(1))
            self.param_scale = nn.Parameter(torch.ones(1) * 1.0)  # Initialize higher
            
            print(f"✓ Parameter conditioning enabled:")
            print(f"  - Time embedding dim: {4 * embedding_dim}")
            print(f"  - Param embedding dim: {param_embed_dim}")
            print(f"  - Total condition dim: {condition_dim}")
        else:
            self.param_conditioning_embedding = None
            self.time_scale = None
            self.param_scale = None
            print("Warning: Parameter conditioning is disabled.")
        total_condition_dim = condition_dim

        resnet_params = dict(
            condition_dim=total_condition_dim,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
        )

        self.embed_t_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
            nn.SiLU(),
        )
        
        # Conditioning fusion layer (if any conditioning beyond time is used)
        if use_param_conditioning:
            self.conditioning_fusion = nn.Sequential(
                nn.Linear(total_condition_dim, total_condition_dim),
                nn.SiLU(),
                nn.Linear(total_condition_dim, total_condition_dim)
            )

        # Input convolution: takes concatenated input and outputs embedding_dim channels
        self.conv_in = nn.Conv2d(total_input_ch, embedding_dim, kernel_size=3, padding=1, padding_mode="zeros")
        
        # Calculate spatial conditioning channels for cross-attention (used by encoder, bottleneck, and decoder)
        # Need to account for Fourier features that will be added to conditioning
        if self.use_cross_attention:
            if use_fourier_features and not legacy_fourier:
                # New mode: conditioning with Fourier features
                # [m_dm | fourier(m_dm) | large_scale | fourier(large_scale)]
                spatial_cond_channels = (
                    conditioning_channels * (1 + self.fourier_features_halo.num_features) +
                    large_scale_channels * (1 + self.fourier_features_largescale.num_features)
                )
            elif use_fourier_features and legacy_fourier:
                # Legacy mode: conditioning + Fourier features
                spatial_cond_channels = conditioning_channels + large_scale_channels + self.fourier_features.num_features
            else:
                # No Fourier features
                spatial_cond_channels = conditioning_channels + large_scale_channels
        else:
            spatial_cond_channels = 0
        
        # Determine if we need cross-attention in encoder
        use_encoder_cross_attn = (self.use_cross_attention and 
                                   self.cross_attention_location == 'everywhere')
        
        # Down blocks with proper channel progression
        self.down_blocks = nn.ModuleList()
        self.down_channels = []
        
        # Channel progression: e.g., 64 -> 64, 128, 256, 512, 1024 (capped)
        max_channels = 1024  # Cap to prevent memory explosion
        channels = [min(embedding_dim * (2 ** i), max_channels) for i in range(n_blocks)]
        
        print(f"✓ Channel progression (encoder): {channels}")
        
        for i in range(n_blocks):
            ch_in = embedding_dim if i == 0 else channels[i-1]  # First block: 48, others: previous output
            ch_out = channels[i]
            
            self.down_channels.append(ch_out)  # Store for skip connections: [48, 96, 192, 384]
            
            # Calculate ACTUAL spatial resolution at this encoder level
            # Assume input is 128x128, resolution halves with each down block
            # Block 0: 128 -> 64, Block 1: 64 -> 32, Block 2: 32 -> 16, Block 3: 16 -> 8
            input_spatial_size = 128  # Standard input size for this model
            current_resolution = input_spatial_size // (2 ** i)  # [128, 64, 32, 16] for i=[0,1,2,3]
            
            # Create ResNet block
            down_resnet = ResnetBlock(ch_in=ch_in, ch_out=ch_out, **resnet_params)
            
            # Add cross-attention in encoder if 'everywhere' mode (skip very high res for speed)
            add_cross_attn_here = (use_encoder_cross_attn and 
                                   current_resolution <= self.cross_attn_max_resolution)
            
            if add_cross_attn_here:
                cross_attn = CrossAttentionBlock(
                    n_heads=cross_attention_heads,
                    n_channels=ch_out,
                    cond_channels=spatial_cond_channels,
                    norm_groups=norm_groups,
                    use_chunked_attention=use_chunked_cross_attention,
                    chunk_size=cross_attention_chunk_size,
                    dropout=self.cross_attention_dropout,
                    downsample_cond=self.downsample_cross_attn_cond,
                    cond_downsample_factor=self.cross_attn_cond_downsample_factor
                )
                self.down_blocks.append(DownBlockWithCrossAttention(
                    resnet_block=down_resnet,
                    cross_attn_block=cross_attn
                ))
            else:
                self.down_blocks.append(DownBlock(resnet_block=down_resnet))
        
        # Print encoder cross-attention summary
        if use_encoder_cross_attn:
            encoder_cross_attn_count = sum(1 for block in self.down_blocks if isinstance(block, DownBlockWithCrossAttention))
            print(f"✓ Encoder cross-attention enabled ('everywhere' mode):")
            print(f"  - {encoder_cross_attn_count}/{len(self.down_blocks)} encoder blocks have cross-attention")
            print(f"  - Max resolution: {self.cross_attn_max_resolution}")
            print(f"  - Heads: {cross_attention_heads}, Conditioning channels: {spatial_cond_channels}")

        # Middle block
        mid_ch_in = self.down_channels[-1]  # 384
        mid_ch_out = channels[-1] * 2  # 768
        
        if self.add_attention:
            self.mid_resnet_block_1 = ResnetBlock(ch_in=mid_ch_in, ch_out=mid_ch_out, **resnet_params)
            self.mid_attn_block = AttentionBlock(n_heads=n_attention_heads, n_channels=mid_ch_out, norm_groups=norm_groups)
            self.mid_resnet_block_2 = ResnetBlock(ch_in=mid_ch_out, ch_out=mid_ch_out, **resnet_params)
        else:
            self.mid_resnet_block = ResnetBlock(ch_in=mid_ch_in, ch_out=mid_ch_out, **resnet_params)
        
        # Cross-attention at bottleneck (Phase 1, Phase 3, or everywhere)
        use_bottleneck_cross_attn = (self.use_cross_attention and 
                                      self.cross_attention_location in ['bottleneck', 'full', 'everywhere'])
        
        if use_bottleneck_cross_attn:
            self.mid_cross_attn_block = CrossAttentionBlock(
                n_heads=cross_attention_heads,
                n_channels=mid_ch_out,
                cond_channels=spatial_cond_channels,
                norm_groups=norm_groups,
                use_chunked_attention=use_chunked_cross_attention,
                chunk_size=cross_attention_chunk_size,
                dropout=self.cross_attention_dropout,
                downsample_cond=self.downsample_cross_attn_cond,
                cond_downsample_factor=self.cross_attn_cond_downsample_factor
            )
            print(f"✓ Cross-attention enabled at bottleneck:")
            print(f"  - Heads: {cross_attention_heads}")
            print(f"  - Feature channels: {mid_ch_out}")
            print(f"  - Conditioning channels: {spatial_cond_channels}")
            print(f"  - Chunked: {use_chunked_cross_attention} (chunk_size={cross_attention_chunk_size})")
            print(f"  - Downsample conditioning: {self.downsample_cross_attn_cond} ({self.cross_attn_cond_downsample_factor}x)")
        else:
            self.mid_cross_attn_block = None

        # Up blocks with proper channel calculations
        self.up_blocks = nn.ModuleList()
        current_ch = mid_ch_out  # Start with 768
        
        # Determine if we need cross-attention in decoder
        use_decoder_cross_attn = (self.use_cross_attention and 
                                   self.cross_attention_location in ['decoder', 'full', 'everywhere'])
        
        for i in range(n_blocks):
            # Skip connection channels from corresponding down block (in reverse order)
            skip_ch = self.down_channels[-(i+1)]  # [384, 192, 96, 48]
            
            # After upsampling: current_ch // 2
            # After concatenation: (current_ch // 2) + skip_ch
            up_ch_in = current_ch // 2 + skip_ch  
            up_ch_out = current_ch // 2  # Output channels
            
            # Calculate ACTUAL spatial resolution at this decoder level
            # Bottleneck is at 8x8 (after 4 down blocks from 128x128)
            # Decoder upsamples: Block 0: 8 -> 16, Block 1: 16 -> 32, Block 2: 32 -> 64, Block 3: 64 -> 128
            bottleneck_spatial_size = 128 // (2 ** n_blocks)  # 128 // 16 = 8
            current_resolution = bottleneck_spatial_size * (2 ** (i + 1))  # [16, 32, 64, 128] for i=[0,1,2,3]
            
            # Create the up block
            up_resnet = ResnetBlock(ch_in=up_ch_in, ch_out=up_ch_out, **resnet_params)
            
            # OPTIMIZATION 4: Only add cross-attention at lower resolutions (skip high-res for 2-4x speedup)
            add_cross_attn_here = (use_decoder_cross_attn and 
                                   current_resolution <= self.cross_attn_max_resolution)
            
            if add_cross_attn_here:
                cross_attn = CrossAttentionBlock(
                    n_heads=cross_attention_heads,
                    n_channels=up_ch_out,
                    cond_channels=spatial_cond_channels,
                    norm_groups=norm_groups,
                    use_chunked_attention=use_chunked_cross_attention,
                    chunk_size=cross_attention_chunk_size,
                    dropout=self.cross_attention_dropout,
                    downsample_cond=self.downsample_cross_attn_cond,
                    cond_downsample_factor=self.cross_attn_cond_downsample_factor
                )
                self.up_blocks.append(UpBlockWithCrossAttention(
                    resnet_block=up_resnet, 
                    upsample_ch=current_ch,
                    cross_attn_block=cross_attn
                ))
            else:
                self.up_blocks.append(UpBlock(resnet_block=up_resnet, upsample_ch=current_ch))
            
            current_ch = up_ch_out
        
        # Print cross-attention configuration
        if use_decoder_cross_attn:
            n_decoder_xattn = sum(1 for block in self.up_blocks 
                                  if hasattr(block, 'cross_attn_block') and block.cross_attn_block is not None)
            print(f"✓ Cross-attention enabled in decoder blocks:")
            print(f"  - Location: {self.cross_attention_location}")
            print(f"  - Heads: {cross_attention_heads}")
            print(f"  - Dropout: {self.cross_attention_dropout}")
            print(f"  - Active in {n_decoder_xattn}/{n_blocks} decoder blocks (resolution ≤{self.cross_attn_max_resolution})")
            print(f"  - Chunked: {use_chunked_cross_attention} (chunk_size={cross_attention_chunk_size})")
            print(f"  - Downsample conditioning: {self.downsample_cross_attn_cond} ({self.cross_attn_cond_downsample_factor}x)")
        
        # Store final decoder channel count for conv_out
        # NOTE: current_ch after all up blocks may NOT equal embedding_dim due to channel capping
        final_decoder_ch = current_ch
        
        # Find a valid norm_groups for final_decoder_ch
        final_norm_groups = norm_groups
        while final_decoder_ch % final_norm_groups != 0 and final_norm_groups > 1:
            final_norm_groups -= 1
        
        # Output convolution: should output 3 channels (target)
        # BUGFIX: Use final_decoder_ch instead of embedding_dim (they may differ with deep networks)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=final_norm_groups, num_channels=final_decoder_ch),
            nn.SiLU(),
            zero_init(nn.Conv2d(final_decoder_ch, input_channels, 3, padding=1, padding_mode="zeros")),
        )
        
        print(f"✓ Final decoder channels: {final_decoder_ch} (GroupNorm groups: {final_norm_groups})")
        
        # Parameter predictor for auxiliary loss (predicts params from bottleneck features)
        if use_param_conditioning and use_param_prediction:
            self.param_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Global average pooling
                nn.Flatten(),
                nn.Linear(mid_ch_out, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.param_conditioning_embedding.Nparams)
            )
            print(f"✓ Parameter predictor added (predicts {self.param_conditioning_embedding.Nparams} params)")
        else:
            self.param_predictor = None
        # NEW: Auxiliary mask prediction head
        self.use_auxiliary_mask = use_auxiliary_mask
        if use_auxiliary_mask:
            self.mask_head = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embedding_dim // 2, norm_groups, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=norm_groups, num_channels=norm_groups),
                nn.SiLU(),
                nn.Conv2d(norm_groups, 1, kernel_size=1),  # (B, 1, H, W) logits
            )
            # Initialize final layer with small weights
            nn.init.normal_(self.mask_head[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.mask_head[-1].bias)
            print(f"✓ Auxiliary mask prediction head added at decoder output (full resolution)")
        else:
            self.mask_head = None

    def forward(self, z, g_t, conditioning=None, param_conditioning=None):  
        # STEP 1: Apply Fourier features to conditioning (if provided and enabled)
        if conditioning is not None and self.use_fourier_features:
            # conditioning shape: (B, 1+N, H, W) where 1=m_dm, N=large_scale_channels
            # Split conditioning into m_dm and large_scale
            m_dm = conditioning[:, :self.conditioning_channels]  # (B, 1, H, W)
            large_scale = conditioning[:, self.conditioning_channels:]  # (B, N, H, W)
            
            if self.legacy_fourier:
                # Legacy: apply Fourier to all conditioning
                fourier_cond = self.fourier_features(conditioning)
                conditioning_with_fourier = torch.cat([conditioning, fourier_cond], dim=1)
            else:
                # New mode: scale-appropriate Fourier features
                # m_dm: high frequencies for fine dark matter structure
                fourier_halo = self.fourier_features_halo(m_dm)
                # large_scale: low frequencies for environmental modulation
                fourier_largescale = self.fourier_features_largescale(large_scale)
                # Concatenate: [m_dm | fourier(m_dm) | large_scale | fourier(large_scale)]
                conditioning_with_fourier = torch.cat([m_dm, fourier_halo, large_scale, fourier_largescale], dim=1)
        elif conditioning is not None:
            # No Fourier features - use conditioning as-is
            conditioning_with_fourier = conditioning
        else:
            conditioning_with_fourier = None
        
        # STEP 2: Decide how to use conditioning based on cross-attention mode
        if self.use_cross_attention and conditioning_with_fourier is not None:
            # Cross-attention: Keep conditioning separate, only pass target to UNet
            spatial_cond = conditioning_with_fourier  # (B, C_cond_with_fourier, H, W)
            z_input = z  # (B, 3, H, W) - target only
        elif conditioning_with_fourier is not None:
            # No cross-attention: Concatenate target with conditioning
            z_input = torch.cat((z, conditioning_with_fourier), dim=1)  # (B, 3+C_cond_with_fourier, H, W)
            spatial_cond = None
        else:
            # No conditioning provided
            z_input = z
            spatial_cond = None

        # Get gamma to shape (B, ).
        g_t = g_t.expand(z_input.shape[0])  # shape () or (1,) or (B,) -> (B,)
        assert g_t.shape == (z_input.shape[0],)

        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        g_t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_embedding = get_timestep_embedding(g_t, self.embedding_dim) #(B, embedding_dim)
        # We will condition on time embedding.
        t_cond = self.embed_t_conditioning(t_embedding) # (B, 4 * embedding_dim)

        # Build combined conditioning
        combined_cond = t_cond
        
         # Build combined conditioning with learnable scales
        if self.use_param_conditioning and param_conditioning is not None:
            param_conditioning = param_conditioning.to(t_cond.device)
            param_emb = self.param_conditioning_embedding(param_conditioning)
            
            # Apply learnable scales for balanced conditioning
            combined_cond = torch.cat([
                self.time_scale * t_cond,
                self.param_scale * param_emb
            ], dim=-1)
            
        elif self.use_param_conditioning:
            # Pad with zeros if parameters not provided
            param_embed_dim = self.param_conditioning_embedding.embedding[-1].out_features
            zero_param = torch.zeros(
                t_cond.shape[0], param_embed_dim,
                device=t_cond.device, dtype=t_cond.dtype
            )
            combined_cond = torch.cat([
                self.time_scale * t_cond,
                zero_param
            ], dim=-1)
        else:
            combined_cond = t_cond

        # Apply fusion layer if any additional conditioning is used
        if self.use_param_conditioning:
            # pass
            combined_cond = self.conditioning_fusion(combined_cond)

        # z_input is ready to go into UNet (Fourier already applied to conditioning if needed)
        # With cross-attention: z_input = target only (3 channels)
        # Without cross-attention: z_input = target + conditioning_with_fourier (3 + C_cond_with_fourier channels)
        h = self.conv_in(z_input)  # (B, embedding_dim, H, W)
        hs = []

        for down_block in self.down_blocks:  # n_blocks times
            # Pass spatial_cond for DownBlockWithCrossAttention, ignored by regular DownBlock
            if isinstance(down_block, DownBlockWithCrossAttention):
                h, hskip = down_block(h, cond=combined_cond, spatial_cond=spatial_cond)
            else:
                h, hskip = down_block(h, cond=combined_cond)
            hs.append(hskip)

        # Middle block with optional self-attention
        if self.add_attention:
            h = self.mid_resnet_block_1(h, combined_cond)
            h = self.mid_attn_block(h)
            h = self.mid_resnet_block_2(h, combined_cond)
        else:
            h = self.mid_resnet_block(h, combined_cond)
        
        # Apply cross-attention at bottleneck (Phase 1)
        if self.use_cross_attention and self.mid_cross_attn_block is not None:
            h = self.mid_cross_attn_block(h, spatial_cond)
        
        # Store bottleneck features for parameter prediction
        bottleneck_features = h

        for up_block in self.up_blocks:  # n_blocks times
            # Pass spatial_cond for UpBlockWithCrossAttention, ignored by regular UpBlock
            if isinstance(up_block, UpBlockWithCrossAttention):
                h = up_block(x=h, xskip=hs.pop(), cond=combined_cond, spatial_cond=spatial_cond)
            else:
                h = up_block(x=h, xskip=hs.pop(), cond=combined_cond)
            
        # Compute auxiliary mask from final decoder features (full resolution)
        if self.use_auxiliary_mask and self.mask_head is not None:
            mask_logits = self.mask_head(h)  # (B, 1, H, W) at full resolution
        else:
            mask_logits = None
        
        prediction = self.conv_out(h)
        
        # Return based on what's enabled
        if self.use_param_prediction and self.param_predictor is not None:
            predicted_params = self.param_predictor(bottleneck_features)
            
            if self.use_auxiliary_mask and mask_logits is not None:
                # Return all three: noise prediction, params, mask
                return prediction + z, predicted_params, mask_logits
            else:
                # Return noise prediction and params only
                return prediction + z, predicted_params
        else:
            if self.use_auxiliary_mask and mask_logits is not None:
                # Return noise prediction and mask (no params)
                return prediction + z, mask_logits
            else:
                # Return only noise prediction
                return prediction + z

    def maybe_concat_fourier(self, z):
        """
        Apply scale-appropriate Fourier features to different components.
        
        Supports three modes:
        - Cross-attention mode: Only target channels (no conditioning concatenated)
        - Legacy mode: Apply Fourier features to all concatenated inputs (first channel only)
        - New mode: Apply scale-appropriate features to m_dm (high freq) and large_scale (low freq)
        
        Args:
            z: Input tensor
               Cross-attention mode: (B, input_channels, H, W) - target only
               Concatenated mode: (B, input_channels + conditioning_channels + large_scale_channels, H, W)
               Structure: [target | m_dm | large_scale]
               
        Returns:
            Cross-attention mode: (B, target, H, W) or (B, target + fourier(target), H, W) if legacy
            Legacy mode: (B, base_channels + fourier_features, H, W)
            New mode: (B, target + m_dm + fourier(m_dm) + large_scale + fourier(large_scale), H, W)
        """
        if not self.use_fourier_features:
            return z
        
        # Cross-attention mode: z only contains target, no conditioning
        if self.use_cross_attention:
            if self.legacy_fourier:
                # Apply Fourier to target only
                return torch.cat([z, self.fourier_features(z)], dim=1)
            else:
                # No Fourier features for target in new mode
                return z
            
        if self.legacy_fourier:
            # Legacy mode: apply Fourier features to all concatenated inputs
            # This matches the old behavior for backward compatibility
            return torch.cat([z, self.fourier_features(z)], dim=1)
        else:
            # New mode: apply scale-appropriate Fourier features to conditioning only
            # Split input into components
            # z has shape (B, input_channels + conditioning_channels + large_scale_channels, H, W)
            target = z[:, :self.input_channels]  # (B, input_channels, H, W)
            m_dm = z[:, self.input_channels:self.input_channels + self.conditioning_channels]  # (B, conditioning_channels, H, W)
            large_scale = z[:, self.input_channels + self.conditioning_channels:]  # (B, large_scale_channels, H, W)
            
            # Apply scale-appropriate Fourier features
            # Target: no Fourier features (diffusion learns naturally across all frequencies)
            # m_dm: high frequencies for fine dark matter structure
            fourier_halo = self.fourier_features_halo(m_dm)  # (B, conditioning_channels * num_features_halo, H, W)
            # large_scale: low frequencies for environmental modulation
            fourier_largescale = self.fourier_features_largescale(large_scale)  # (B, large_scale_channels * num_features_largescale, H, W)
            
            # Concatenate: [target | m_dm | fourier(m_dm) | large_scale | fourier(large_scale)]
            return torch.cat([target, m_dm, fourier_halo, large_scale, fourier_largescale], dim=1)
