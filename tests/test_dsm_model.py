"""
Tests for vdm/dsm_model.py module.

Tests cover:
- VPSDESchedule class
- UNetDSMWrapper
- LightDSM Lightning module
- BIND-compatible sampling interface
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test VPSDESchedule
# ============================================================================

class TestVPSDESchedule:
    """Test VPSDESchedule class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        from vdm.dsm_model import VPSDESchedule
        
        schedule = VPSDESchedule()
        
        assert schedule.beta_min == 0.1
        assert schedule.beta_max == 20.0
    
    def test_init_custom(self):
        """Test custom initialization."""
        from vdm.dsm_model import VPSDESchedule
        
        schedule = VPSDESchedule(beta_min=0.05, beta_max=30.0)
        
        assert schedule.beta_min == 0.05
        assert schedule.beta_max == 30.0
    
    def test_alpha_at_zero(self):
        """Test alpha(0) = 1 (no noise at t=0)."""
        from vdm.dsm_model import VPSDESchedule
        
        schedule = VPSDESchedule()
        
        alpha_0 = schedule.alpha(torch.tensor(0.0))
        assert torch.isclose(alpha_0, torch.tensor(1.0), rtol=1e-5)
    
    def test_alpha_decreasing(self):
        """Test alpha is monotonically decreasing in t."""
        from vdm.dsm_model import VPSDESchedule
        
        schedule = VPSDESchedule()
        
        t = torch.linspace(0, 1, 100)
        alpha = schedule.alpha(t)
        
        # Check monotonically decreasing
        diffs = alpha[1:] - alpha[:-1]
        assert torch.all(diffs < 0), "alpha should decrease with t"
    
    def test_sigma_at_zero(self):
        """Test sigma(0) ≈ 0 (no noise at t=0)."""
        from vdm.dsm_model import VPSDESchedule
        
        schedule = VPSDESchedule()
        
        sigma_0 = schedule.sigma(torch.tensor(0.0))
        assert sigma_0 < 0.01, f"sigma(0) should be near 0, got {sigma_0}"
    
    def test_sigma_increasing(self):
        """Test sigma is monotonically increasing in t."""
        from vdm.dsm_model import VPSDESchedule
        
        schedule = VPSDESchedule()
        
        t = torch.linspace(0.01, 1, 99)  # Avoid exact 0
        sigma = schedule.sigma(t)
        
        # Check monotonically increasing
        diffs = sigma[1:] - sigma[:-1]
        assert torch.all(diffs > 0), "sigma should increase with t"
    
    def test_snr_decreasing(self):
        """Test SNR decreases with t (more noise = lower SNR)."""
        from vdm.dsm_model import VPSDESchedule
        
        schedule = VPSDESchedule()
        
        t = torch.linspace(0.01, 0.99, 50)  # Avoid endpoints
        snr = schedule.snr(t)
        
        # Check monotonically decreasing
        diffs = snr[1:] - snr[:-1]
        assert torch.all(diffs < 0), "SNR should decrease with t"
    
    def test_variance_preserving(self):
        """Test that alpha^2 + sigma^2 ≈ 1 (variance preserving)."""
        from vdm.dsm_model import VPSDESchedule
        
        schedule = VPSDESchedule()
        
        t = torch.linspace(0, 1, 100)
        alpha = schedule.alpha(t)
        sigma = schedule.sigma(t)
        
        total_variance = alpha ** 2 + sigma ** 2
        assert torch.allclose(total_variance, torch.ones_like(total_variance), atol=1e-5), \
            f"VP-SDE should preserve variance: alpha^2 + sigma^2 = 1, got {total_variance}"


# ============================================================================
# Test UNetDSMWrapper
# ============================================================================

class TestUNetDSMWrapper:
    """Test UNetDSMWrapper class."""
    
    @pytest.fixture
    def mock_unet(self):
        """Create a simple mock UNet for testing."""
        class SimpleUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Minimal conv that outputs same shape as input
                self.conv = torch.nn.Conv2d(7, 3, 1)  # 3 input + 4 cond -> 3 output
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                if conditioning is not None:
                    x = torch.cat([x, conditioning], dim=1)
                return self.conv(x)
        
        return SimpleUNet()
    
    def test_wrapper_forward_shape(self, mock_unet):
        """Test wrapper forward pass produces correct shape."""
        from vdm.dsm_model import UNetDSMWrapper
        
        wrapper = UNetDSMWrapper(mock_unet)
        
        B, C, H, W = 2, 3, 32, 32
        t = torch.rand(B)
        x = torch.randn(B, C, H, W)
        cond = torch.randn(B, 4, H, W)
        
        out = wrapper(t, x, cond)
        
        assert out.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {out.shape}"
    
    def test_wrapper_without_conditioning(self, mock_unet):
        """Test wrapper works without conditioning."""
        from vdm.dsm_model import UNetDSMWrapper
        
        # Need a UNet that works without conditioning
        class NoCondUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                return self.conv(x)
        
        wrapper = UNetDSMWrapper(NoCondUNet())
        
        B, C, H, W = 2, 3, 32, 32
        t = torch.rand(B)
        x = torch.randn(B, C, H, W)
        
        out = wrapper(t, x, conditioning=None)
        
        assert out.shape == (B, C, H, W)


# ============================================================================
# Test LightDSM
# ============================================================================

class TestLightDSM:
    """Test LightDSM Lightning module."""
    
    @pytest.fixture
    def simple_score_model(self):
        """Create a simple score model for testing."""
        class SimpleScoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(7, 3, 1)  # 3 input + 4 cond -> 3 output
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                if conditioning is not None:
                    x = torch.cat([x, conditioning], dim=1)
                return self.conv(x)
        
        return SimpleScoreModel()
    
    def test_init(self, simple_score_model):
        """Test LightDSM initialization."""
        from vdm.dsm_model import LightDSM
        
        model = LightDSM(
            score_model=simple_score_model,
            beta_min=0.1,
            beta_max=20.0,
            learning_rate=1e-4,
            n_sampling_steps=100,
        )
        
        assert model.beta_min == 0.1
        assert model.beta_max == 20.0
        assert model.n_sampling_steps == 100
        assert model.learning_rate == 1e-4
    
    def test_init_with_snr_weighting(self, simple_score_model):
        """Test initialization with SNR weighting."""
        from vdm.dsm_model import LightDSM
        
        model = LightDSM(
            score_model=simple_score_model,
            use_snr_weighting=True,
        )
        
        assert model.use_snr_weighting == True
    
    def test_init_channel_weights(self, simple_score_model):
        """Test channel weights are registered correctly."""
        from vdm.dsm_model import LightDSM
        
        model = LightDSM(
            score_model=simple_score_model,
            channel_weights=(1.0, 2.0, 3.0),
        )
        
        assert torch.allclose(
            model.channel_weights,
            torch.tensor([1.0, 2.0, 3.0])
        )
    
    def test_add_noise_variance_preserving(self, simple_score_model):
        """Test _add_noise produces variance-preserving noisy samples."""
        from vdm.dsm_model import LightDSM
        
        model = LightDSM(score_model=simple_score_model)
        
        B, C, H, W = 4, 3, 32, 32
        x = torch.randn(B, C, H, W)
        t = torch.rand(B)
        
        z_t, noise = model._add_noise(x, t)
        
        # Check output shapes
        assert z_t.shape == x.shape
        assert noise.shape == x.shape
        
        # Verify variance-preserving property: z_t = alpha_t * x + sigma_t * noise
        t_broadcast = t.view(B, 1, 1, 1)
        alpha_t = model.schedule.alpha(t_broadcast)
        sigma_t = model.schedule.sigma(t_broadcast)
        expected_z_t = alpha_t * x + sigma_t * noise
        
        assert torch.allclose(z_t, expected_z_t, atol=1e-5)
    
    def test_add_noise_at_t0(self, simple_score_model):
        """Test that t=0 gives no noise."""
        from vdm.dsm_model import LightDSM
        
        model = LightDSM(score_model=simple_score_model)
        
        B, C, H, W = 4, 3, 32, 32
        x = torch.randn(B, C, H, W)
        t = torch.zeros(B)
        
        z_t, noise = model._add_noise(x, t)
        
        # At t=0, alpha=1, sigma≈0, so z_t ≈ x
        assert torch.allclose(z_t, x, atol=0.01)


class TestLightDSMSampling:
    """Test LightDSM sampling interface."""
    
    @pytest.fixture
    def dsm_model(self):
        """Create a DSM model with simple network for sampling tests."""
        from vdm.dsm_model import LightDSM
        
        class SimpleScoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Output zeros = predict no noise = identity sampling
                self.register_buffer('zero', torch.zeros(1))
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                return torch.zeros_like(x)
        
        return LightDSM(
            score_model=SimpleScoreModel(),
            n_sampling_steps=10,  # Few steps for fast testing
        )
    
    def test_sample_shape(self, dsm_model):
        """Test sample() produces correct shape."""
        B, C, H, W = 2, 3, 32, 32
        cond_channels = 4
        
        conditioning = torch.randn(B, cond_channels, H, W)
        
        samples = dsm_model.sample(
            shape=(B, C, H, W),
            conditioning=conditioning,
            steps=5,
            verbose=False,
        )
        
        assert samples.shape == (B, C, H, W)
    
    def test_draw_samples_interface(self, dsm_model):
        """Test BIND-compatible draw_samples interface."""
        B, C_cond, H, W = 2, 4, 32, 32
        
        conditioning = torch.randn(B, C_cond, H, W)
        
        samples = dsm_model.draw_samples(
            conditioning=conditioning,
            batch_size=B,
            n_sampling_steps=5,
            verbose=False,
        )
        
        # Should output 3 channels [DM, Gas, Stars]
        assert samples.shape == (B, 3, H, W)


class TestLightDSMTraining:
    """Test LightDSM training functionality."""
    
    @pytest.fixture
    def dsm_model_with_unet(self):
        """Create DSM with more realistic network for training tests."""
        from vdm.dsm_model import LightDSM
        
        class TrainableScoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(7, 3, 3, padding=1)
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                if conditioning is not None:
                    x = torch.cat([x, conditioning], dim=1)
                return self.conv(x)
        
        return LightDSM(
            score_model=TrainableScoreModel(),
            learning_rate=1e-4,
        )
    
    def test_compute_loss_returns_scalar(self, dsm_model_with_unet):
        """Test _compute_loss returns scalar loss."""
        B, C, H, W = 4, 3, 32, 32
        x = torch.randn(B, C, H, W)
        cond = torch.randn(B, 4, H, W)
        
        loss, metrics = dsm_model_with_unet._compute_loss(x, cond)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert 'total_mse' in metrics
        assert 'dm_mse' in metrics
        assert 'gas_mse' in metrics
        assert 'stellar_mse' in metrics
    
    def test_training_step(self, dsm_model_with_unet):
        """Test training step with batch."""
        B, H, W = 2, 32, 32
        
        # Create batch in AstroDataset format: (m_dm, large_scale, m_target, conditions)
        m_dm = torch.randn(B, 1, H, W)
        large_scale = torch.randn(B, 3, H, W)
        m_target = torch.randn(B, 3, H, W)
        conditions = torch.randn(B, 35)  # Cosmological parameters
        
        batch = (m_dm, large_scale, m_target, conditions)
        
        loss = dsm_model_with_unet.training_step(batch, batch_idx=0)
        
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_validation_step(self, dsm_model_with_unet):
        """Test validation step with batch."""
        B, H, W = 2, 32, 32
        
        m_dm = torch.randn(B, 1, H, W)
        large_scale = torch.randn(B, 3, H, W)
        m_target = torch.randn(B, 3, H, W)
        conditions = torch.randn(B, 35)
        
        batch = (m_dm, large_scale, m_target, conditions)
        
        loss = dsm_model_with_unet.validation_step(batch, batch_idx=0)
        
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestLightDSMOptimizer:
    """Test LightDSM optimizer configuration."""
    
    @pytest.fixture
    def dsm_model(self):
        """Create DSM model for optimizer tests."""
        from vdm.dsm_model import LightDSM
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                return self.conv(x)
        
        return LightDSM(
            score_model=SimpleModel(),
            learning_rate=1e-4,
            lr_scheduler='cosine',
        )
    
    def test_configure_optimizers_cosine(self, dsm_model):
        """Test optimizer configuration with cosine scheduler."""
        # Create a minimal trainer-like object
        class MockTrainer:
            max_epochs = 100
        
        dsm_model.trainer = MockTrainer()
        
        result = dsm_model.configure_optimizers()
        
        assert 'optimizer' in result
        assert 'lr_scheduler' in result
        
        optimizer = result['optimizer']
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults['lr'] == 1e-4


# ============================================================================
# Test Alias
# ============================================================================

class TestAlias:
    """Test backward compatibility alias."""
    
    def test_alias_exists(self):
        """Test that LightDenoisingScoreMatching alias exists."""
        from vdm.dsm_model import LightDenoisingScoreMatching, LightDSM
        
        assert LightDenoisingScoreMatching is LightDSM
