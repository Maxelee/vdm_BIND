"""
Tests for vdm/ot_flow_model.py module.

Tests cover:
- OT utilities (compute_ot_plan, apply_ot_coupling, sample_ot_pairs)
- OTInterpolant class
- OTVelocityNetWrapper
- LightOTFlow Lightning module
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test OT Utilities
# ============================================================================

class TestOTUtilities:
    """Test OT utility functions."""
    
    def test_compute_ot_plan_exact(self):
        """Test exact OT plan computation."""
        from vdm.ot_flow_model import compute_ot_plan, HAS_POT
        
        if not HAS_POT:
            pytest.skip("POT library not installed")
        
        # Create simple test data
        B, C, H, W = 4, 3, 8, 8
        x0 = torch.randn(B, C, H, W)
        x1 = torch.randn(B, C, H, W)
        
        # Compute OT plan
        pi = compute_ot_plan(x0, x1, method='exact')
        
        # Check shape
        assert pi.shape == (B, B)
        
        # Check it's approximately a permutation matrix (each row/col sums to ~1/B)
        row_sums = pi.sum(dim=1)
        col_sums = pi.sum(dim=0)
        assert torch.allclose(row_sums, torch.ones(B) / B, atol=1e-5)
        assert torch.allclose(col_sums, torch.ones(B) / B, atol=1e-5)
    
    def test_compute_ot_plan_sinkhorn(self):
        """Test Sinkhorn OT plan computation."""
        from vdm.ot_flow_model import compute_ot_plan, HAS_POT
        
        if not HAS_POT:
            pytest.skip("POT library not installed")
        
        B, C, H, W = 4, 3, 8, 8
        x0 = torch.randn(B, C, H, W)
        x1 = torch.randn(B, C, H, W)
        
        # Compute Sinkhorn OT plan
        pi = compute_ot_plan(x0, x1, method='sinkhorn', reg=0.1)
        
        # Check shape
        assert pi.shape == (B, B)
        
        # Check it's approximately a transport plan
        row_sums = pi.sum(dim=1)
        col_sums = pi.sum(dim=0)
        assert torch.allclose(row_sums, torch.ones(B) / B, atol=1e-3)
        assert torch.allclose(col_sums, torch.ones(B) / B, atol=1e-3)
    
    def test_compute_ot_plan_fallback_without_pot(self):
        """Test fallback to identity when POT not available."""
        from vdm.ot_flow_model import compute_ot_plan
        
        B, C, H, W = 4, 3, 8, 8
        x0 = torch.randn(B, C, H, W)
        x1 = torch.randn(B, C, H, W)
        
        # This should work regardless of POT availability
        # (falls back to identity if POT missing)
        pi = compute_ot_plan(x0, x1, method='exact')
        
        assert pi.shape == (B, B)
    
    def test_apply_ot_coupling(self):
        """Test applying OT coupling to reorder samples."""
        from vdm.ot_flow_model import apply_ot_coupling
        
        B = 4
        C, H, W = 3, 8, 8
        x0 = torch.randn(B, C, H, W)
        x1 = torch.arange(B).float().view(B, 1, 1, 1).expand(B, C, H, W)
        
        # Create a simple permutation: reverse order
        pi = torch.zeros(B, B)
        for i in range(B):
            pi[i, B - 1 - i] = 1.0  # Map i -> B-1-i
        
        x0_out, x1_out = apply_ot_coupling(x0, x1, pi)
        
        # x0 should be unchanged
        assert torch.allclose(x0_out, x0)
        
        # x1 should be reversed
        expected = torch.arange(B - 1, -1, -1).float().view(B, 1, 1, 1).expand(B, C, H, W)
        assert torch.allclose(x1_out, expected)
    
    def test_sample_ot_pairs(self):
        """Test sampling OT-paired data."""
        from vdm.ot_flow_model import sample_ot_pairs, HAS_POT
        
        if not HAS_POT:
            pytest.skip("POT library not installed")
        
        B, C, H, W = 4, 3, 8, 8
        x0 = torch.randn(B, C, H, W)
        x1 = torch.randn(B, C, H, W)
        
        x0_paired, x1_paired = sample_ot_pairs(x0, x1, method='exact')
        
        # Check shapes are preserved
        assert x0_paired.shape == x0.shape
        assert x1_paired.shape == x1.shape


# ============================================================================
# Test OTInterpolant
# ============================================================================

class TestOTInterpolant:
    """Test OTInterpolant class."""
    
    @pytest.fixture
    def mock_velocity_model(self):
        """Create a minimal velocity model for testing."""
        class SimpleVelocityModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
            
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                # Just return x for simplicity
                return x
        
        return SimpleVelocityModel()
    
    def test_ot_interpolant_init(self, mock_velocity_model):
        """Test OTInterpolant initialization."""
        from vdm.ot_flow_model import OTInterpolant
        
        interpolant = OTInterpolant(
            velocity_model=mock_velocity_model,
            ot_method='exact',
            ot_reg=0.01,
        )
        
        assert interpolant.ot_method == 'exact'
        assert interpolant.ot_reg == 0.01
        assert interpolant.use_stochastic_interpolant == False
        assert interpolant.sigma == 0.0
    
    def test_get_mu_t(self, mock_velocity_model):
        """Test linear interpolation."""
        from vdm.ot_flow_model import OTInterpolant
        
        interpolant = OTInterpolant(velocity_model=mock_velocity_model)
        
        B, C, H, W = 2, 3, 8, 8
        x0 = torch.zeros(B, C, H, W)
        x1 = torch.ones(B, C, H, W)
        
        # t=0 should give x0
        t0 = torch.zeros(B)
        mu_0 = interpolant.get_mu_t(x0, x1, t0)
        assert torch.allclose(mu_0, x0)
        
        # t=1 should give x1
        t1 = torch.ones(B)
        mu_1 = interpolant.get_mu_t(x0, x1, t1)
        assert torch.allclose(mu_1, x1)
        
        # t=0.5 should give midpoint
        t_mid = torch.full((B,), 0.5)
        mu_mid = interpolant.get_mu_t(x0, x1, t_mid)
        assert torch.allclose(mu_mid, 0.5 * x0 + 0.5 * x1)
    
    def test_get_velocity(self, mock_velocity_model):
        """Test velocity computation."""
        from vdm.ot_flow_model import OTInterpolant
        
        interpolant = OTInterpolant(velocity_model=mock_velocity_model)
        
        B, C, H, W = 2, 3, 8, 8
        x0 = torch.zeros(B, C, H, W)
        x1 = torch.ones(B, C, H, W)
        
        v = interpolant.get_velocity(x0, x1)
        
        # Velocity should be x1 - x0
        assert torch.allclose(v, x1 - x0)
    
    def test_sample_xt_deterministic(self, mock_velocity_model):
        """Test deterministic interpolant sampling."""
        from vdm.ot_flow_model import OTInterpolant
        
        interpolant = OTInterpolant(
            velocity_model=mock_velocity_model,
            use_stochastic_interpolant=False,
        )
        
        B, C, H, W = 2, 3, 8, 8
        x0 = torch.zeros(B, C, H, W)
        x1 = torch.ones(B, C, H, W)
        t = torch.full((B,), 0.5)
        
        x_t = interpolant.sample_xt(x0, x1, t)
        
        # Should be deterministic (just mu_t)
        expected = interpolant.get_mu_t(x0, x1, t)
        assert torch.allclose(x_t, expected)
    
    def test_sample_xt_stochastic(self, mock_velocity_model):
        """Test stochastic interpolant sampling."""
        from vdm.ot_flow_model import OTInterpolant
        
        interpolant = OTInterpolant(
            velocity_model=mock_velocity_model,
            use_stochastic_interpolant=True,
            sigma=0.1,
        )
        
        B, C, H, W = 2, 3, 8, 8
        x0 = torch.zeros(B, C, H, W)
        x1 = torch.ones(B, C, H, W)
        t = torch.full((B,), 0.5)
        
        # Sample twice - should be different due to stochasticity
        torch.manual_seed(42)
        x_t_1 = interpolant.sample_xt(x0, x1, t)
        torch.manual_seed(43)
        x_t_2 = interpolant.sample_xt(x0, x1, t)
        
        # Should be different
        assert not torch.allclose(x_t_1, x_t_2)
    
    def test_compute_loss(self, mock_velocity_model):
        """Test loss computation."""
        from vdm.ot_flow_model import OTInterpolant, HAS_POT
        
        interpolant = OTInterpolant(
            velocity_model=mock_velocity_model,
            ot_method='exact',
        )
        
        B, C, H, W = 4, 3, 8, 8
        x0 = torch.randn(B, C, H, W)
        x1 = torch.randn(B, C, H, W)
        
        # Test with OT disabled
        loss, metrics = interpolant.compute_loss(x0, x1, use_ot=False)
        
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # MSE is non-negative
    
    def test_sample(self, mock_velocity_model):
        """Test ODE sampling."""
        from vdm.ot_flow_model import OTInterpolant
        
        interpolant = OTInterpolant(velocity_model=mock_velocity_model)
        
        B, C, H, W = 2, 3, 8, 8
        x0 = torch.randn(B, C, H, W)
        
        # Sample with 10 steps
        samples = interpolant.sample(x0, n_steps=10)
        
        assert samples.shape == (B, C, H, W)
    
    def test_sample_with_trajectory(self, mock_velocity_model):
        """Test ODE sampling with trajectory."""
        from vdm.ot_flow_model import OTInterpolant
        
        interpolant = OTInterpolant(velocity_model=mock_velocity_model)
        
        B, C, H, W = 2, 3, 8, 8
        x0 = torch.randn(B, C, H, W)
        
        # Sample with trajectory
        trajectory = interpolant.sample(x0, n_steps=10, return_trajectory=True)
        
        assert isinstance(trajectory, list)
        assert len(trajectory) == 11  # Initial + 10 steps


# ============================================================================
# Test OTVelocityNetWrapper
# ============================================================================

class TestOTVelocityNetWrapper:
    """Test OTVelocityNetWrapper class."""
    
    @pytest.fixture
    def mock_unet(self):
        """Create a minimal UNet for testing."""
        from vdm.networks_clean import UNetVDM
        return UNetVDM(
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
    
    def test_wrapper_forward(self, mock_unet):
        """Test wrapper forward pass."""
        from vdm.ot_flow_model import OTVelocityNetWrapper
        
        wrapper = OTVelocityNetWrapper(
            net=mock_unet,
            output_channels=3,
            conditioning_channels=1,
        )
        
        B, C, H, W = 2, 3, 32, 32
        t = torch.rand(B)
        x = torch.randn(B, C, H, W)
        conditioning = torch.randn(B, 1, H, W)
        
        output = wrapper(t, x, conditioning=conditioning)
        
        assert output.shape == (B, C, H, W)
    
    def test_wrapper_with_param_conditioning(self):
        """Test wrapper with parameter conditioning."""
        from vdm.ot_flow_model import OTVelocityNetWrapper
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
            use_param_conditioning=True,
            param_min=np.zeros(5),
            param_max=np.ones(5),
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        wrapper = OTVelocityNetWrapper(
            net=unet,
            output_channels=3,
            conditioning_channels=1,
        )
        
        B, C, H, W = 2, 3, 32, 32
        t = torch.rand(B)
        x = torch.randn(B, C, H, W)
        conditioning = torch.randn(B, 1, H, W)
        params = torch.rand(B, 5)
        
        output = wrapper(t, x, conditioning=conditioning, param_conditioning=params)
        
        assert output.shape == (B, C, H, W)


# ============================================================================
# Test LightOTFlow
# ============================================================================

class TestLightOTFlow:
    """Test LightOTFlow Lightning module."""
    
    @pytest.fixture
    def mock_velocity_model(self):
        """Create a minimal velocity model for testing."""
        from vdm.ot_flow_model import OTVelocityNetWrapper
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
            large_scale_channels=0,
        )
        
        return OTVelocityNetWrapper(
            net=unet,
            output_channels=3,
            conditioning_channels=1,
        )
    
    def test_light_ot_flow_init(self, mock_velocity_model):
        """Test LightOTFlow initialization."""
        from vdm.ot_flow_model import LightOTFlow
        
        model = LightOTFlow(
            velocity_model=mock_velocity_model,
            learning_rate=1e-4,
            n_sampling_steps=50,
            ot_method='exact',
            ot_reg=0.01,
            x0_mode='zeros',
        )
        
        assert model.learning_rate == 1e-4
        assert model.n_sampling_steps == 50
        assert model.x0_mode == 'zeros'
    
    def test_get_x0_zeros(self, mock_velocity_model):
        """Test x0 initialization with zeros."""
        from vdm.ot_flow_model import LightOTFlow
        
        model = LightOTFlow(velocity_model=mock_velocity_model, x0_mode='zeros')
        
        x1 = torch.randn(2, 3, 32, 32)
        x0 = model._get_x0(x1)
        
        assert torch.allclose(x0, torch.zeros_like(x1))
    
    def test_get_x0_noise(self, mock_velocity_model):
        """Test x0 initialization with noise."""
        from vdm.ot_flow_model import LightOTFlow
        
        model = LightOTFlow(velocity_model=mock_velocity_model, x0_mode='noise')
        
        x1 = torch.randn(2, 3, 32, 32)
        
        torch.manual_seed(42)
        x0 = model._get_x0(x1)
        
        # Should be random noise
        assert not torch.allclose(x0, torch.zeros_like(x1))
    
    def test_get_x0_dm_copy(self, mock_velocity_model):
        """Test x0 initialization with dm_copy."""
        from vdm.ot_flow_model import LightOTFlow
        
        model = LightOTFlow(velocity_model=mock_velocity_model, x0_mode='dm_copy')
        
        x1 = torch.randn(2, 3, 32, 32)
        dm_condition = torch.randn(2, 1, 32, 32)
        
        x0 = model._get_x0(x1, dm_condition)
        
        # Should copy DM to all channels
        for c in range(3):
            assert torch.allclose(x0[:, c:c+1], dm_condition)
    
    def test_unpack_batch(self, mock_velocity_model):
        """Test batch unpacking."""
        from vdm.ot_flow_model import LightOTFlow
        
        model = LightOTFlow(
            velocity_model=mock_velocity_model,
            use_param_conditioning=True,
        )
        
        B, H, W = 2, 32, 32
        m_dm = torch.randn(B, 1, H, W)
        large_scale = torch.randn(B, 3, H, W)
        m_target = torch.randn(B, 3, H, W)
        conditions = torch.rand(B, 5)
        
        batch = (m_dm, large_scale, m_target, conditions)
        x1, conditioning, dm_condition, params = model._unpack_batch(batch)
        
        assert x1.shape == (B, 3, H, W)
        assert conditioning.shape == (B, 4, H, W)  # 1 + 3
        assert dm_condition.shape == (B, 1, H, W)
        assert params.shape == (B, 5)
    
    def test_sample(self, mock_velocity_model):
        """Test sampling."""
        from vdm.ot_flow_model import LightOTFlow
        
        model = LightOTFlow(
            velocity_model=mock_velocity_model,
            n_sampling_steps=10,
            x0_mode='zeros',
        )
        
        B, C, H, W = 2, 3, 32, 32
        conditioning = torch.randn(B, 1, H, W)
        
        samples = model.sample(
            shape=(B, C, H, W),
            conditioning=conditioning,
            steps=10,
        )
        
        assert samples.shape == (B, C, H, W)
    
    def test_draw_samples(self, mock_velocity_model):
        """Test BIND-compatible draw_samples interface."""
        from vdm.ot_flow_model import LightOTFlow
        
        model = LightOTFlow(
            velocity_model=mock_velocity_model,
            n_sampling_steps=10,
        )
        
        B, H, W = 2, 32, 32
        conditioning = torch.randn(B, 1, H, W)
        
        samples = model.draw_samples(
            conditioning=conditioning,
            batch_size=B,
            n_sampling_steps=10,
        )
        
        assert samples.shape == (B, 3, H, W)
    
    def test_configure_optimizers_cosine(self, mock_velocity_model):
        """Test optimizer configuration with cosine scheduler."""
        from vdm.ot_flow_model import LightOTFlow
        
        model = LightOTFlow(
            velocity_model=mock_velocity_model,
            lr_scheduler='cosine',
        )
        
        # Mock trainer
        model.trainer = type('Trainer', (), {'max_epochs': 100})()
        
        config = model.configure_optimizers()
        
        assert 'optimizer' in config
        assert 'lr_scheduler' in config
    
    def test_configure_optimizers_plateau(self, mock_velocity_model):
        """Test optimizer configuration with plateau scheduler."""
        from vdm.ot_flow_model import LightOTFlow
        
        model = LightOTFlow(
            velocity_model=mock_velocity_model,
            lr_scheduler='plateau',
        )
        
        config = model.configure_optimizers()
        
        assert 'optimizer' in config
        assert 'lr_scheduler' in config


# ============================================================================
# Test Factory Functions
# ============================================================================

class TestOTFlowFactory:
    """Test factory functions."""
    
    def test_create_ot_flow_model(self):
        """Test create_ot_flow_model factory."""
        from vdm.ot_flow_model import create_ot_flow_model
        
        # This test may fail if UNet import fails
        try:
            model = create_ot_flow_model(
                output_channels=3,
                conditioning_channels=4,
                embedding_dim=32,
                n_blocks=2,
                n_sampling_steps=10,
                ot_method='exact',
            )
            
            assert isinstance(model, LightOTFlow)
        except ImportError:
            pytest.skip("UNet import failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
