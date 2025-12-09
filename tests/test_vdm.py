"""
Tests for VDM model components.
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUNetVDM:
    """Test UNet architecture."""
    
    @pytest.fixture
    def small_unet(self):
        """Create a small UNet for testing."""
        from vdm.networks_clean import UNetVDM
        return UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,  # Small for testing
            norm_groups=4,
            n_blocks=2,  # Fewer blocks for speed
            add_attention=False,  # Skip attention for speed
            n_attention_heads=4,
            use_fourier_features=False,  # Simpler for testing
            use_param_conditioning=False,
            conditioning_channels=1,
            large_scale_channels=0,
        )
    
    def test_unet_forward_shape(self, small_unet):
        """UNet should output same shape as input."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)
        gamma = torch.randn(batch_size)  # Network expects (B,) shape
        cond = torch.randn(batch_size, 1, 64, 64)
        
        out = small_unet(x, gamma, cond)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_unet_deterministic(self, small_unet):
        """UNet should be deterministic (no dropout in eval mode)."""
        small_unet.eval()
        x = torch.randn(1, 3, 64, 64)
        gamma = torch.randn(1)  # Network expects (B,) shape
        cond = torch.randn(1, 1, 64, 64)
        
        with torch.no_grad():
            out1 = small_unet(x, gamma, cond)
            out2 = small_unet(x, gamma, cond)
        
        assert torch.allclose(out1, out2)
    
    def test_unet_with_large_scale(self):
        """UNet should work with large-scale conditioning."""
        from vdm.networks_clean import UNetVDM
        
        unet = UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,
            norm_groups=4,
            n_blocks=2,
            add_attention=False,
            use_fourier_features=False,
            use_param_conditioning=False,
            conditioning_channels=1,
            large_scale_channels=3,  # Additional context maps
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)
        gamma = torch.randn(batch_size)  # Network expects (B,) shape
        # Conditioning: 1 base + 3 large-scale = 4 total channels
        cond = torch.randn(batch_size, 4, 64, 64)
        
        out = unet(x, gamma, cond)
        assert out.shape == x.shape


class TestCleanVDM:
    """Test VDM model logic."""
    
    @pytest.fixture
    def vdm_model(self):
        """Create a small VDM for testing."""
        from vdm.networks_clean import UNetVDM
        from vdm.vdm_model_clean import CleanVDM
        
        score_model = UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,
            norm_groups=4,
            n_blocks=2,
            add_attention=False,
            use_fourier_features=False,
            use_param_conditioning=False,
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        return CleanVDM(
            score_model=score_model,
            noise_schedule="fixed_linear",
            gamma_min=-10.0,
            gamma_max=10.0,
            image_shape=(3, 64, 64),
        )
    
    def test_variance_preserving_map(self, vdm_model):
        """Variance-preserving diffusion should produce valid outputs."""
        x = torch.randn(2, 3, 64, 64)
        times = torch.rand(2)
        
        z_t, gamma_t = vdm_model.variance_preserving_map(x, times)
        
        assert z_t.shape == x.shape
        assert gamma_t.shape == (2, 1, 1, 1)
    
    def test_sample_times(self, vdm_model):
        """Sampled times should be in [0, 1]."""
        times = vdm_model.sample_times(10, 'cpu')
        
        assert times.shape == (10,)
        assert torch.all(times >= 0)
        assert torch.all(times <= 1)
    
    def test_snr_computation(self, vdm_model):
        """SNR should be positive for valid gamma."""
        gamma = torch.tensor([-5.0, 0.0, 5.0])
        snr = vdm_model.get_snr(gamma)
        
        assert torch.all(snr > 0)


class TestNoiseSchedules:
    """Test noise schedule implementations."""
    
    def test_fixed_linear_schedule(self):
        """Fixed linear schedule should interpolate correctly."""
        from vdm.utils import FixedLinearSchedule
        
        schedule = FixedLinearSchedule(gamma_min=-10.0, gamma_max=10.0)
        
        # At t=0, gamma should be gamma_min (start of diffusion)
        gamma_0 = schedule(torch.tensor([0.0]))
        assert torch.isclose(gamma_0, torch.tensor([-10.0]), atol=0.1)
        
        # At t=1, gamma should be gamma_max (end of diffusion)
        gamma_1 = schedule(torch.tensor([1.0]))
        assert torch.isclose(gamma_1, torch.tensor([10.0]), atol=0.1)
        
        # At t=0.5, gamma should be near 0 (midpoint)
        gamma_half = schedule(torch.tensor([0.5]))
        assert torch.isclose(gamma_half, torch.tensor([0.0]), atol=0.5)
