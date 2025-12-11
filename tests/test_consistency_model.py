"""
Tests for vdm/consistency_model.py module.

Tests cover:
- ConsistencyNoiseSchedule class
- ConsistencyFunction class (skip connection parameterization)
- ConsistencyModel class
- ConsistencyNetWrapper
- LightConsistency Lightning module
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test ConsistencyNoiseSchedule
# ============================================================================

class TestConsistencyNoiseSchedule:
    """Test ConsistencyNoiseSchedule class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        from vdm.consistency_model import ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule()
        
        assert schedule.sigma_min == 0.002
        assert schedule.sigma_max == 80.0
        assert schedule.rho == 7.0
    
    def test_init_custom(self):
        """Test custom initialization."""
        from vdm.consistency_model import ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule(
            sigma_min=0.01,
            sigma_max=50.0,
            rho=5.0,
        )
        
        assert schedule.sigma_min == 0.01
        assert schedule.sigma_max == 50.0
        assert schedule.rho == 5.0
    
    def test_get_sigma_endpoints(self):
        """Test sigma at t=0 and t=1."""
        from vdm.consistency_model import ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule(sigma_min=0.002, sigma_max=80.0)
        
        # t=0 should give sigma_min
        sigma_0 = schedule.get_sigma(torch.tensor(0.0))
        assert torch.isclose(sigma_0, torch.tensor(0.002), rtol=1e-5)
        
        # t=1 should give sigma_max
        sigma_1 = schedule.get_sigma(torch.tensor(1.0))
        assert torch.isclose(sigma_1, torch.tensor(80.0), rtol=1e-5)
    
    def test_get_sigma_monotonic(self):
        """Test that sigma is monotonically increasing in t."""
        from vdm.consistency_model import ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule()
        
        t = torch.linspace(0, 1, 100)
        sigma = schedule.get_sigma(t)
        
        # Check monotonically increasing
        diffs = sigma[1:] - sigma[:-1]
        assert torch.all(diffs > 0)
    
    def test_get_t_from_sigma_inverse(self):
        """Test that get_t_from_sigma is inverse of get_sigma."""
        from vdm.consistency_model import ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule()
        
        # Test round-trip
        t_original = torch.linspace(0.01, 0.99, 50)  # Avoid exact endpoints
        sigma = schedule.get_sigma(t_original)
        t_recovered = schedule.get_t_from_sigma(sigma)
        
        assert torch.allclose(t_original, t_recovered, rtol=1e-4)
    
    def test_get_discretized_sigmas(self):
        """Test discretized sigma schedule."""
        from vdm.consistency_model import ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule(sigma_min=0.002, sigma_max=80.0)
        
        n_steps = 10
        sigmas = schedule.get_discretized_sigmas(n_steps)
        
        # Should have n_steps + 1 values
        assert sigmas.shape == (n_steps + 1,)
        
        # First should be sigma_max, last should be sigma_min
        assert torch.isclose(sigmas[0], torch.tensor(80.0), rtol=1e-5)
        assert torch.isclose(sigmas[-1], torch.tensor(0.002), rtol=1e-5)
        
        # Should be decreasing
        diffs = sigmas[1:] - sigmas[:-1]
        assert torch.all(diffs < 0)


# ============================================================================
# Test ConsistencyFunction
# ============================================================================

class TestConsistencyFunction:
    """Test ConsistencyFunction class."""
    
    @pytest.fixture
    def mock_net(self):
        """Create a simple network for testing."""
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                return self.conv(x)
        
        return SimpleNet()
    
    def test_init(self, mock_net):
        """Test initialization."""
        from vdm.consistency_model import ConsistencyFunction
        
        fn = ConsistencyFunction(
            net=mock_net,
            sigma_data=0.5,
            sigma_min=0.002,
        )
        
        assert fn.sigma_data == 0.5
        assert fn.sigma_min == 0.002
    
    def test_get_scalings_shape(self, mock_net):
        """Test scaling computation shapes."""
        from vdm.consistency_model import ConsistencyFunction
        
        fn = ConsistencyFunction(net=mock_net)
        
        B = 4
        sigma = torch.rand(B)
        
        c_skip, c_out, c_in = fn.get_scalings(sigma)
        
        # All should be broadcastable to (B, 1, 1, 1)
        assert c_skip.shape == (B, 1, 1, 1)
        assert c_out.shape == (B, 1, 1, 1)
        assert c_in.shape == (B, 1, 1, 1)
    
    def test_get_scalings_boundary(self, mock_net):
        """Test scaling at sigma=sigma_min (boundary condition)."""
        from vdm.consistency_model import ConsistencyFunction
        
        sigma_data = 0.5
        sigma_min = 0.002
        fn = ConsistencyFunction(net=mock_net, sigma_data=sigma_data, sigma_min=sigma_min)
        
        # At sigma close to 0, c_skip should be close to 1
        sigma_small = torch.tensor([sigma_min])
        c_skip, c_out, c_in = fn.get_scalings(sigma_small)
        
        # c_skip = sigma_data^2 / (sigma^2 + sigma_data^2) ≈ 1 when sigma << sigma_data
        expected_c_skip = sigma_data**2 / (sigma_min**2 + sigma_data**2)
        assert torch.isclose(c_skip.squeeze(), torch.tensor(expected_c_skip), rtol=1e-3)
    
    def test_forward(self, mock_net):
        """Test forward pass."""
        from vdm.consistency_model import ConsistencyFunction
        
        fn = ConsistencyFunction(net=mock_net)
        
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        sigma = torch.rand(B) * 10
        
        output = fn(x, sigma)
        
        assert output.shape == (B, C, H, W)
    
    def test_forward_with_conditioning(self, mock_net):
        """Test forward pass with conditioning."""
        from vdm.consistency_model import ConsistencyFunction
        
        fn = ConsistencyFunction(net=mock_net)
        
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        sigma = torch.rand(B) * 10
        conditioning = torch.randn(B, 1, H, W)
        
        output = fn(x, sigma, conditioning=conditioning)
        
        assert output.shape == (B, C, H, W)


# ============================================================================
# Test ConsistencyModel
# ============================================================================

class TestConsistencyModel:
    """Test ConsistencyModel class."""
    
    @pytest.fixture
    def mock_consistency_fn(self):
        """Create a mock consistency function."""
        class MockConsistencyFn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
            
            def forward(self, x, sigma, conditioning=None, param_conditioning=None):
                # Just return x (identity-ish)
                return x * 0.99  # Slight change to avoid exact identity
        
        return MockConsistencyFn()
    
    def test_init(self, mock_consistency_fn):
        """Test initialization."""
        from vdm.consistency_model import ConsistencyModel, ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule()
        model = ConsistencyModel(
            consistency_fn=mock_consistency_fn,
            noise_schedule=schedule,
            sigma_data=0.5,
        )
        
        assert model.sigma_data == 0.5
    
    def test_add_noise(self, mock_consistency_fn):
        """Test noise addition."""
        from vdm.consistency_model import ConsistencyModel, ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule()
        model = ConsistencyModel(
            consistency_fn=mock_consistency_fn,
            noise_schedule=schedule,
        )
        
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        sigma = torch.tensor([1.0, 2.0])
        
        x_noisy = model.add_noise(x, sigma)
        
        # Should be different from original
        assert x_noisy.shape == x.shape
        assert not torch.allclose(x_noisy, x)
    
    def test_compute_denoising_loss(self, mock_consistency_fn):
        """Test denoising loss computation."""
        from vdm.consistency_model import ConsistencyModel, ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule()
        model = ConsistencyModel(
            consistency_fn=mock_consistency_fn,
            noise_schedule=schedule,
        )
        
        B, C, H, W = 4, 3, 32, 32
        x0 = torch.randn(B, C, H, W)
        
        loss = model.compute_denoising_loss(x0)
        
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # MSE is non-negative
    
    def test_compute_ct_loss(self, mock_consistency_fn):
        """Test consistency training loss."""
        from vdm.consistency_model import ConsistencyModel, ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule()
        model = ConsistencyModel(
            consistency_fn=mock_consistency_fn,
            noise_schedule=schedule,
        )
        
        B, C, H, W = 4, 3, 32, 32
        x0 = torch.randn(B, C, H, W)
        
        loss = model.compute_ct_loss(x0, n_steps=10)
        
        assert loss.ndim == 0  # Scalar
        assert loss >= 0
    
    def test_sample_single_step(self, mock_consistency_fn):
        """Test single-step sampling."""
        from vdm.consistency_model import ConsistencyModel, ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule()
        model = ConsistencyModel(
            consistency_fn=mock_consistency_fn,
            noise_schedule=schedule,
        )
        
        B, C, H, W = 2, 3, 32, 32
        x_init = torch.randn(B, C, H, W) * schedule.sigma_max
        
        samples = model.sample(x_init, n_steps=1)
        
        assert samples.shape == (B, C, H, W)
    
    def test_sample_multi_step(self, mock_consistency_fn):
        """Test multi-step sampling."""
        from vdm.consistency_model import ConsistencyModel, ConsistencyNoiseSchedule
        
        schedule = ConsistencyNoiseSchedule()
        model = ConsistencyModel(
            consistency_fn=mock_consistency_fn,
            noise_schedule=schedule,
        )
        
        B, C, H, W = 2, 3, 32, 32
        x_init = torch.randn(B, C, H, W) * schedule.sigma_max
        
        samples = model.sample(x_init, n_steps=5)
        
        assert samples.shape == (B, C, H, W)


# ============================================================================
# Test ConsistencyNetWrapper
# ============================================================================

class TestConsistencyNetWrapper:
    """Test ConsistencyNetWrapper class."""
    
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
        from vdm.consistency_model import ConsistencyNetWrapper
        
        wrapper = ConsistencyNetWrapper(
            net=mock_unet,
            output_channels=3,
            conditioning_channels=1,
        )
        
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        t = torch.rand(B)
        conditioning = torch.randn(B, 1, H, W)
        
        output = wrapper(x, t, conditioning=conditioning)
        
        assert output.shape == (B, C, H, W)
    
    def test_wrapper_with_param_conditioning(self):
        """Test wrapper with parameter conditioning."""
        from vdm.consistency_model import ConsistencyNetWrapper
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
        
        wrapper = ConsistencyNetWrapper(
            net=unet,
            output_channels=3,
            conditioning_channels=1,
        )
        
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        t = torch.rand(B)
        conditioning = torch.randn(B, 1, H, W)
        params = torch.rand(B, 5)
        
        output = wrapper(x, t, conditioning=conditioning, param_conditioning=params)
        
        assert output.shape == (B, C, H, W)


# ============================================================================
# Test LightConsistency
# ============================================================================

class TestLightConsistency:
    """Test LightConsistency Lightning module."""
    
    @pytest.fixture
    def mock_consistency_model(self):
        """Create a mock consistency model for testing."""
        from vdm.consistency_model import (
            ConsistencyModel, ConsistencyFunction, 
            ConsistencyNoiseSchedule, ConsistencyNetWrapper
        )
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
        
        wrapper = ConsistencyNetWrapper(net=unet, output_channels=3, conditioning_channels=1)
        schedule = ConsistencyNoiseSchedule()
        consistency_fn = ConsistencyFunction(net=wrapper)
        
        return ConsistencyModel(
            consistency_fn=consistency_fn,
            noise_schedule=schedule,
        )
    
    def test_init(self, mock_consistency_model):
        """Test initialization."""
        from vdm.consistency_model import LightConsistency
        
        model = LightConsistency(
            consistency_model=mock_consistency_model,
            learning_rate=1e-4,
            n_sampling_steps=1,
        )
        
        assert model.learning_rate == 1e-4
        assert model.n_sampling_steps == 1
    
    def test_unpack_batch(self, mock_consistency_model):
        """Test batch unpacking."""
        from vdm.consistency_model import LightConsistency
        
        model = LightConsistency(
            consistency_model=mock_consistency_model,
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
    
    def test_sample(self, mock_consistency_model):
        """Test sampling."""
        from vdm.consistency_model import LightConsistency
        
        model = LightConsistency(
            consistency_model=mock_consistency_model,
            n_sampling_steps=2,
        )
        
        B, C, H, W = 2, 3, 32, 32
        conditioning = torch.randn(B, 1, H, W)
        
        samples = model.sample(
            shape=(B, C, H, W),
            conditioning=conditioning,
            steps=2,
        )
        
        assert samples.shape == (B, C, H, W)
    
    def test_draw_samples(self, mock_consistency_model):
        """Test BIND-compatible draw_samples interface."""
        from vdm.consistency_model import LightConsistency
        
        model = LightConsistency(
            consistency_model=mock_consistency_model,
            n_sampling_steps=2,
        )
        
        B, H, W = 2, 32, 32
        conditioning = torch.randn(B, 1, H, W)
        
        samples = model.draw_samples(
            conditioning=conditioning,
            batch_size=B,
            n_sampling_steps=2,
        )
        
        assert samples.shape == (B, 3, H, W)
    
    def test_configure_optimizers_cosine(self, mock_consistency_model):
        """Test optimizer configuration with cosine scheduler."""
        from vdm.consistency_model import LightConsistency
        
        model = LightConsistency(
            consistency_model=mock_consistency_model,
            lr_scheduler='cosine',
        )
        
        # Mock trainer
        model.trainer = type('Trainer', (), {'max_epochs': 100})()
        
        config = model.configure_optimizers()
        
        assert 'optimizer' in config
        assert 'lr_scheduler' in config
    
    def test_configure_optimizers_plateau(self, mock_consistency_model):
        """Test optimizer configuration with plateau scheduler."""
        from vdm.consistency_model import LightConsistency
        
        model = LightConsistency(
            consistency_model=mock_consistency_model,
            lr_scheduler='plateau',
        )
        
        config = model.configure_optimizers()
        
        assert 'optimizer' in config
        assert 'lr_scheduler' in config


# ============================================================================
# Test Factory Functions
# ============================================================================

class TestConsistencyFactory:
    """Test factory functions."""
    
    def test_create_consistency_model(self):
        """Test create_consistency_model factory."""
        from vdm.consistency_model import create_consistency_model
        
        try:
            model = create_consistency_model(
                output_channels=3,
                conditioning_channels=4,
                embedding_dim=32,
                n_blocks=2,
                n_sampling_steps=1,
            )
            
            assert hasattr(model, 'draw_samples')
        except ImportError:
            pytest.skip("UNet import failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
