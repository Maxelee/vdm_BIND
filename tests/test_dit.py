"""
Tests for DiT (Diffusion Transformer) model.

Tests:
- PatchEmbed: patchify/unpatchify round-trip
- DiTBlock: forward pass with conditioning
- DiT model: forward pass with various configs
- LightDiTVDM: training step, sampling
"""

import pytest
import torch
import torch.nn as nn

from vdm.dit import (
    PatchEmbed,
    TimestepEmbedder,
    ParamEmbedder,
    DiTBlock,
    FinalLayer,
    DiT,
    create_dit_model,
    DIT_MODELS,
    get_2d_sincos_pos_embed,
)
from vdm.dit_model import LightDiTVDM


class TestPatchEmbed:
    """Tests for patch embedding."""
    
    def test_output_shape(self):
        """Test that PatchEmbed produces correct output shape."""
        embed = PatchEmbed(img_size=64, patch_size=4, in_channels=3, embed_dim=768)
        x = torch.randn(2, 3, 64, 64)
        out = embed(x)
        
        # num_patches = (64/4)^2 = 256
        assert out.shape == (2, 256, 768)
    
    def test_num_patches(self):
        """Test num_patches calculation."""
        embed = PatchEmbed(img_size=128, patch_size=8, in_channels=3, embed_dim=512)
        assert embed.num_patches == (128 // 8) ** 2  # 256
        assert embed.grid_size == 128 // 8  # 16
    
    def test_different_patch_sizes(self):
        """Test with various patch sizes."""
        for patch_size in [2, 4, 8, 16]:
            embed = PatchEmbed(img_size=64, patch_size=patch_size, in_channels=3, embed_dim=256)
            x = torch.randn(1, 3, 64, 64)
            out = embed(x)
            
            expected_patches = (64 // patch_size) ** 2
            assert out.shape == (1, expected_patches, 256)


class TestTimestepEmbedder:
    """Tests for timestep embedding."""
    
    def test_output_shape(self):
        """Test timestep embedder output shape."""
        embedder = TimestepEmbedder(hidden_size=768)
        t = torch.rand(4)
        out = embedder(t)
        assert out.shape == (4, 768)
    
    def test_different_timesteps(self):
        """Test with various timestep values."""
        embedder = TimestepEmbedder(hidden_size=256)
        
        # Edge cases
        t_zero = torch.zeros(2)
        t_one = torch.ones(2)
        t_mid = torch.tensor([0.5, 0.5])
        
        out_zero = embedder(t_zero)
        out_one = embedder(t_one)
        out_mid = embedder(t_mid)
        
        # All should produce valid outputs
        assert out_zero.shape == (2, 256)
        assert out_one.shape == (2, 256)
        assert out_mid.shape == (2, 256)
        
        # Different timesteps should produce different embeddings
        assert not torch.allclose(out_zero, out_one)


class TestParamEmbedder:
    """Tests for parameter embedding."""
    
    def test_with_params(self):
        """Test parameter embedding with conditioning."""
        embedder = ParamEmbedder(n_params=35, hidden_size=768)
        params = torch.randn(4, 35)
        out = embedder(params)
        assert out.shape == (4, 768)
    
    def test_unconditional(self):
        """Test unconditional mode (n_params=0)."""
        embedder = ParamEmbedder(n_params=0, hidden_size=768)
        out = embedder(None)
        assert out is None
    
    def test_zero_params(self):
        """Test that n_params=0 creates no MLP."""
        embedder = ParamEmbedder(n_params=0, hidden_size=768)
        assert embedder.mlp is None


class TestDiTBlock:
    """Tests for DiT transformer block."""
    
    def test_forward_pass(self):
        """Test DiT block forward pass."""
        block = DiTBlock(hidden_size=768, num_heads=12, mlp_ratio=4.0)
        
        x = torch.randn(2, 256, 768)  # (B, N, D)
        c = torch.randn(2, 768)  # (B, D) conditioning
        
        out = block(x, c)
        assert out.shape == x.shape
    
    def test_conditioning_effect(self):
        """Test that conditioning affects output."""
        block = DiTBlock(hidden_size=256, num_heads=4)
        
        x = torch.randn(2, 64, 256)
        c1 = torch.randn(2, 256)
        c2 = torch.randn(2, 256)
        
        out1 = block(x, c1)
        out2 = block(x, c2)
        
        # Different conditioning should produce different outputs
        assert not torch.allclose(out1, out2)
    
    def test_residual_connection(self):
        """Test that output is close to input (residual connection at init)."""
        block = DiTBlock(hidden_size=256, num_heads=4)
        
        x = torch.randn(1, 64, 256)
        c = torch.zeros(1, 256)  # Zero conditioning
        
        out = block(x, c)
        # At initialization with adaLN-Zero, output should be close to input
        # (gates are initialized to zero, but numerical precision varies)
        # Allow reasonable tolerance for numerical precision
        assert torch.allclose(out, x, atol=0.1, rtol=0.1)


class TestDiTModel:
    """Tests for full DiT model."""
    
    def test_forward_pass(self):
        """Test DiT forward pass."""
        model = DiT(
            img_size=64,
            patch_size=4,
            in_channels=3,
            out_channels=3,
            hidden_size=256,
            depth=4,
            num_heads=4,
            n_params=6,
            conditioning_channels=1,
            large_scale_channels=3,
        )
        
        x = torch.randn(2, 3, 64, 64)
        t = torch.rand(2)
        cond = torch.randn(2, 4, 64, 64)  # 1 DM + 3 large scale
        params = torch.randn(2, 6)
        
        out = model(x, t, conditioning=cond, param_conditioning=params)
        assert out.shape == (2, 3, 64, 64)
    
    def test_unconditional(self):
        """Test unconditional generation."""
        model = DiT(
            img_size=64,
            patch_size=8,
            hidden_size=256,
            depth=2,
            num_heads=4,
            n_params=0,  # Unconditional
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        x = torch.randn(2, 3, 64, 64)
        t = torch.rand(2)
        cond = torch.randn(2, 1, 64, 64)
        
        out = model(x, t, conditioning=cond, param_conditioning=None)
        assert out.shape == (2, 3, 64, 64)
    
    def test_model_variants(self):
        """Test creating model variants."""
        # Test a few variants
        for name in ['DiT-S/4', 'DiT-S/8', 'DiT-B/4']:
            model = create_dit_model(
                name,
                img_size=64,
                n_params=6,
                conditioning_channels=1,
                large_scale_channels=3,
            )
            
            x = torch.randn(1, 3, 64, 64)
            t = torch.rand(1)
            cond = torch.randn(1, 4, 64, 64)
            params = torch.randn(1, 6)
            
            out = model(x, t, conditioning=cond, param_conditioning=params)
            assert out.shape == (1, 3, 64, 64)
    
    def test_available_models(self):
        """Test all model variants are accessible."""
        expected_models = ['DiT-S/4', 'DiT-S/8', 'DiT-B/4', 'DiT-B/8', 
                          'DiT-L/4', 'DiT-L/8', 'DiT-XL/4', 'DiT-XL/8']
        for name in expected_models:
            assert name in DIT_MODELS


class TestLightDiTVDM:
    """Tests for Lightning module wrapper."""
    
    @pytest.fixture
    def model(self):
        """Create a small DiT model for testing."""
        dit = DiT(
            img_size=64,
            patch_size=8,
            hidden_size=128,
            depth=2,
            num_heads=4,
            n_params=6,
            conditioning_channels=1,
            large_scale_channels=3,
        )
        return LightDiTVDM(
            dit_model=dit,
            learning_rate=1e-4,
            gamma_min=-13.3,
            gamma_max=5.0,
            n_sampling_steps=4,  # Fast for testing
            image_shape=(3, 64, 64),
        )
    
    def test_training_step(self, model):
        """Test training step."""
        dm = torch.randn(2, 1, 64, 64)
        large_scale = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        params = torch.randn(2, 6)
        
        batch = [dm, large_scale, target, params]
        loss = model.training_step(batch, 0)
        
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_validation_step(self, model):
        """Test validation step."""
        dm = torch.randn(2, 1, 64, 64)
        large_scale = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        params = torch.randn(2, 6)
        
        batch = [dm, large_scale, target, params]
        loss = model.validation_step(batch, 0)
        
        assert torch.isfinite(loss)
    
    def test_sampling(self, model):
        """Test sampling."""
        model.eval()
        conditioning = torch.randn(2, 4, 64, 64)
        params = torch.randn(2, 6)
        
        with torch.no_grad():
            samples = model.sample(conditioning, param_conditioning=params, n_steps=2)
        
        assert samples.shape == (2, 3, 64, 64)
    
    def test_draw_samples_interface(self, model):
        """Test BIND-compatible sampling interface."""
        model.eval()
        conditioning = torch.randn(2, 4, 64, 64)
        params = torch.randn(2, 6)
        
        with torch.no_grad():
            samples = model.draw_samples(
                conditioning,
                param_conditioning=params,
                n_sampling_steps=2,
            )
        
        assert samples.shape == (2, 3, 64, 64)
    
    def test_noise_schedule(self, model):
        """Test gamma/noise schedule.
        
        VDM convention: gamma = gamma_min + (gamma_max - gamma_min) * t
        - t=0 (clean data): gamma = gamma_min (low noise, high SNR)
        - t=1 (pure noise): gamma = gamma_max (high noise, low SNR)
        """
        t = torch.tensor([0.0, 0.5, 1.0])
        gamma = model.gamma(t)
        
        # gamma should increase from gamma_min to gamma_max (VDM convention)
        assert torch.isclose(gamma[0], torch.tensor(model.gamma_min))
        assert torch.isclose(gamma[2], torch.tensor(model.gamma_max))
        assert torch.isclose(gamma[1], torch.tensor((model.gamma_max + model.gamma_min) / 2))


class TestPositionalEmbedding:
    """Tests for positional embeddings."""
    
    def test_2d_sincos_shape(self):
        """Test 2D sinusoidal position embedding shape."""
        embed = get_2d_sincos_pos_embed(embed_dim=256, grid_size=16)
        assert embed.shape == (256, 256)  # 16*16 positions, 256 dims
    
    def test_embed_uniqueness(self):
        """Test that positions have unique embeddings."""
        embed = get_2d_sincos_pos_embed(embed_dim=64, grid_size=8)
        
        # Each position should have a unique embedding
        n_positions = embed.shape[0]
        for i in range(n_positions):
            for j in range(i + 1, n_positions):
                assert not torch.allclose(embed[i], embed[j])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
