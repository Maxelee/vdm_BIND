"""
Tests for the Interpolant/Flow Matching model.

Tests cover:
- Interpolant core functionality (interpolation, velocity computation)
- LightInterpolant Lightning module
- Model creation factory functions
- Sampling interface compatibility with BIND
"""

import pytest
import torch
import numpy as np

# Skip all tests if interpolant module not available
pytest.importorskip("vdm.interpolant_model")


class TestInterpolant:
    """Test the core Interpolant class."""
    
    def test_get_mu_t_endpoints(self):
        """Test that mu_t equals x0 at t=0 and x1 at t=1."""
        from vdm.interpolant_model import Interpolant
        
        # Create dummy velocity model
        class DummyVelocity(torch.nn.Module):
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return torch.zeros_like(x)
        
        interpolant = Interpolant(velocity_model=DummyVelocity())
        
        # Create test data
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32)
        x1 = torch.randn(batch_size, 3, 32, 32)
        
        # Test t=0
        t0 = torch.zeros(batch_size)
        mu_t0 = interpolant.get_mu_t(x0, x1, t0)
        assert torch.allclose(mu_t0, x0, atol=1e-6), "mu_t should equal x0 at t=0"
        
        # Test t=1
        t1 = torch.ones(batch_size)
        mu_t1 = interpolant.get_mu_t(x0, x1, t1)
        assert torch.allclose(mu_t1, x1, atol=1e-6), "mu_t should equal x1 at t=1"
    
    def test_get_mu_t_midpoint(self):
        """Test that mu_t equals (x0 + x1) / 2 at t=0.5."""
        from vdm.interpolant_model import Interpolant
        
        class DummyVelocity(torch.nn.Module):
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return torch.zeros_like(x)
        
        interpolant = Interpolant(velocity_model=DummyVelocity())
        
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32)
        x1 = torch.randn(batch_size, 3, 32, 32)
        
        t_mid = torch.full((batch_size,), 0.5)
        mu_t_mid = interpolant.get_mu_t(x0, x1, t_mid)
        expected = 0.5 * x1 + 0.5 * x0
        assert torch.allclose(mu_t_mid, expected, atol=1e-6)
    
    def test_velocity_constant(self):
        """Test that true velocity is constant (x1 - x0) for linear interpolant."""
        from vdm.interpolant_model import Interpolant
        
        class DummyVelocity(torch.nn.Module):
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return torch.zeros_like(x)
        
        interpolant = Interpolant(velocity_model=DummyVelocity())
        
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32)
        x1 = torch.randn(batch_size, 3, 32, 32)
        
        v = interpolant.get_velocity(x0, x1)
        expected = x1 - x0
        assert torch.allclose(v, expected, atol=1e-6)
    
    def test_sample_xt_deterministic(self):
        """Test that sample_xt without stochasticity returns mu_t."""
        from vdm.interpolant_model import Interpolant
        
        class DummyVelocity(torch.nn.Module):
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return torch.zeros_like(x)
        
        interpolant = Interpolant(velocity_model=DummyVelocity(), use_stochastic_interpolant=False)
        
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32)
        x1 = torch.randn(batch_size, 3, 32, 32)
        t = torch.rand(batch_size)
        
        x_t, epsilon = interpolant.sample_xt(x0, x1, t)
        mu_t = interpolant.get_mu_t(x0, x1, t)
        
        assert torch.allclose(x_t, mu_t, atol=1e-6)
        assert epsilon is None  # No epsilon for deterministic interpolant
    
    def test_sample_xt_stochastic(self):
        """Test that sample_xt with stochasticity adds noise and returns epsilon."""
        from vdm.interpolant_model import Interpolant
        
        class DummyVelocity(torch.nn.Module):
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return torch.zeros_like(x)
        
        sigma = 0.1
        interpolant = Interpolant(velocity_model=DummyVelocity(), use_stochastic_interpolant=True, sigma=sigma)
        
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32)
        x1 = torch.randn(batch_size, 3, 32, 32)
        t = torch.full((batch_size,), 0.5)  # t=0.5 for maximum noise
        
        x_t, epsilon = interpolant.sample_xt(x0, x1, t)
        mu_t = interpolant.get_mu_t(x0, x1, t)
        sigma_t = interpolant.get_sigma_t(t)
        
        assert epsilon is not None
        # x_t should be mu_t + sigma_t * epsilon
        expected_x_t = mu_t + sigma_t * epsilon
        assert torch.allclose(x_t, expected_x_t, atol=1e-6)
    
    def test_stochastic_loss_includes_noise_term(self):
        """Test that stochastic interpolant loss target includes gamma_dot * z."""
        from vdm.interpolant_model import Interpolant
        
        # Create a velocity model that just returns the input so we can inspect the loss
        class IdentityVelocity(torch.nn.Module):
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return x  # Return x so loss = MSE(x, v_true)
        
        sigma = 0.1
        interp_stoch = Interpolant(velocity_model=IdentityVelocity(), use_stochastic_interpolant=True, sigma=sigma)
        interp_det = Interpolant(velocity_model=IdentityVelocity(), use_stochastic_interpolant=False)
        
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 32, 32)
        x1 = torch.randn(batch_size, 3, 32, 32)
        t = torch.full((batch_size,), 0.5)  # t=0.5 for maximum effect
        
        # Compute losses - they should differ because stochastic includes gamma_dot * z term
        torch.manual_seed(42)
        loss_stoch = interp_stoch.compute_loss(x0, x1, t=t)
        torch.manual_seed(42)
        loss_det = interp_det.compute_loss(x0, x1, t=t)
        
        # Losses should be different due to the noise term
        assert not torch.allclose(loss_stoch, loss_det)


class TestLightInterpolant:
    """Test the LightInterpolant Lightning module."""
    
    def test_initialization(self):
        """Test that LightInterpolant initializes correctly."""
        from vdm.interpolant_model import LightInterpolant
        
        class SimpleVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(7, 3, 1)  # 3 output + 4 conditioning -> 3 output
            
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                if conditioning is not None:
                    x_in = torch.cat([x, conditioning], dim=1)
                else:
                    x_in = x
                return self.conv(x_in)
        
        model = LightInterpolant(
            velocity_model=SimpleVelocity(),
            learning_rate=1e-4,
            n_sampling_steps=20,
            x0_mode='zeros',
        )
        
        assert model.learning_rate == 1e-4
        assert model.n_sampling_steps == 20
        assert model.x0_mode == 'zeros'
    
    def test_x0_modes(self):
        """Test different x0 initialization modes."""
        from vdm.interpolant_model import LightInterpolant
        
        class SimpleVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(7, 3, 1)
            
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                if conditioning is not None:
                    x_in = torch.cat([x, conditioning], dim=1)
                else:
                    x_in = x
                return self.conv(x_in)
        
        x1 = torch.randn(2, 3, 32, 32)
        dm_cond = torch.randn(2, 1, 32, 32)
        
        # Test zeros mode
        model_zeros = LightInterpolant(velocity_model=SimpleVelocity(), x0_mode='zeros')
        x0_zeros = model_zeros._get_x0(x1, dm_cond)
        assert torch.allclose(x0_zeros, torch.zeros_like(x1))
        
        # Test dm_copy mode
        model_dm = LightInterpolant(velocity_model=SimpleVelocity(), x0_mode='dm_copy')
        x0_dm = model_dm._get_x0(x1, dm_cond)
        # Should be dm_cond expanded to 3 channels
        expected = dm_cond.expand(-1, 3, -1, -1)
        assert x0_dm.shape == x1.shape
    
    def test_draw_samples_shape(self):
        """Test that draw_samples returns correct shape."""
        from vdm.interpolant_model import LightInterpolant
        
        class SimpleVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(7, 3, 1)
            
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                if conditioning is not None:
                    x_in = torch.cat([x, conditioning], dim=1)
                else:
                    x_in = x
                return self.conv(x_in)
        
        model = LightInterpolant(
            velocity_model=SimpleVelocity(),
            n_sampling_steps=5,
            x0_mode='zeros',
        )
        model.eval()
        
        batch_size = 2
        conditioning = torch.randn(batch_size, 4, 32, 32)  # 4 conditioning channels
        
        with torch.no_grad():
            samples = model.draw_samples(
                conditioning=conditioning,
                batch_size=batch_size,
                n_sampling_steps=5,
            )
        
        assert samples.shape == (batch_size, 3, 32, 32)
    
    def test_dm_copy_mode_uses_conditioning(self):
        """Test that dm_copy mode correctly initializes x0 from DM condition during sampling."""
        from vdm.interpolant_model import LightInterpolant
        
        class IdentityVelocity(torch.nn.Module):
            """Returns zero velocity, so output equals input."""
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return torch.zeros_like(x)
        
        model = LightInterpolant(
            velocity_model=IdentityVelocity(),
            n_sampling_steps=5,
            x0_mode='dm_copy',
        )
        model.eval()
        
        batch_size = 2
        # Create conditioning where first channel has distinct values
        dm_condition = torch.ones(batch_size, 1, 32, 32) * 5.0  # DM = 5.0
        large_scale = torch.randn(batch_size, 3, 32, 32)  # Random large-scale
        conditioning = torch.cat([dm_condition, large_scale], dim=1)  # (B, 4, H, W)
        
        with torch.no_grad():
            samples = model.sample(
                shape=(batch_size, 3, 32, 32),
                conditioning=conditioning,
            )
        
        # With zero velocity, output should equal x0 = dm_condition expanded to 3 channels
        # So all output channels should be ~5.0
        assert samples.shape == (batch_size, 3, 32, 32)
        assert torch.allclose(samples, torch.ones_like(samples) * 5.0), \
            f"Expected samples to be 5.0 (dm_copy), got mean={samples.mean():.4f}"


class TestVelocityNetWrapper:
    """Test the VelocityNetWrapper class."""
    
    def test_forward_with_conditioning(self):
        """Test forward pass with conditioning."""
        from vdm.interpolant_model import VelocityNetWrapper
        
        # Mock network that matches UNetVDM signature
        # UNetVDM receives x and conditioning separately and handles concatenation internally
        class MockNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)  # Only processes x (3 channels)
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                # UNetVDM internally concatenates and processes, but for mock
                # we just return a transformed x of the same shape
                return self.conv(x)
        
        wrapper = VelocityNetWrapper(
            net=MockNet(),
            output_channels=3,
            conditioning_channels=4,
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        t = torch.rand(batch_size)
        conditioning = torch.randn(batch_size, 4, 32, 32)
        
        output = wrapper(t, x, conditioning)
        
        assert output.shape == (batch_size, 3, 32, 32)
    
    def test_forward_without_conditioning(self):
        """Test forward pass without conditioning."""
        from vdm.interpolant_model import VelocityNetWrapper
        
        # Mock network that handles conditioning=None
        class MockNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)  # Only processes x (3 channels)
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                return self.conv(x)
        
        wrapper = VelocityNetWrapper(
            net=MockNet(),
            output_channels=3,
            conditioning_channels=4,
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        t = torch.rand(batch_size)
        
        # Should work - wrapper passes conditioning=None to network
        output = wrapper(t, x, conditioning=None)
        
        assert output.shape == (batch_size, 3, 32, 32)
    
    def test_forward_handles_tuple_output(self):
        """Test that wrapper correctly handles tuple output from UNetVDM (param_prediction)."""
        from vdm.interpolant_model import VelocityNetWrapper
        
        # Mock network that returns tuple (like UNetVDM with param_prediction=True)
        class MockNetWithParams(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1)
            
            def forward(self, x, t, conditioning=None, param_conditioning=None):
                prediction = self.conv(x)
                predicted_params = torch.randn(x.shape[0], 35)  # 35 params
                return prediction, predicted_params  # Returns tuple!
        
        wrapper = VelocityNetWrapper(
            net=MockNetWithParams(),
            output_channels=3,
            conditioning_channels=4,
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        t = torch.rand(batch_size)
        conditioning = torch.randn(batch_size, 4, 32, 32)
        
        # Should extract only the prediction tensor from the tuple
        output = wrapper(t, x, conditioning)
        
        assert isinstance(output, torch.Tensor), "Output should be tensor, not tuple"
        assert output.shape == (batch_size, 3, 32, 32)


class TestComputeLoss:
    """Test the flow matching loss computation."""
    
    def test_loss_shape(self):
        """Test that loss is a scalar."""
        from vdm.interpolant_model import Interpolant
        
        class PerfectVelocity(torch.nn.Module):
            """Velocity model that predicts exact velocity."""
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return torch.zeros_like(x)  # Will produce non-zero loss
        
        interpolant = Interpolant(velocity_model=PerfectVelocity())
        
        batch_size = 4
        x0 = torch.zeros(batch_size, 3, 32, 32)
        x1 = torch.randn(batch_size, 3, 32, 32)
        conditioning = torch.randn(batch_size, 4, 32, 32)
        
        loss = interpolant.compute_loss(x0, x1, conditioning)
        
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive when velocity prediction is wrong"
    
    def test_loss_zero_when_perfect(self):
        """Test that loss is zero when velocity model is perfect."""
        from vdm.interpolant_model import Interpolant
        
        # Store reference to x0, x1 for perfect prediction
        x0_ref = torch.randn(4, 3, 32, 32)
        x1_ref = torch.randn(4, 3, 32, 32)
        true_velocity = x1_ref - x0_ref
        
        class PerfectVelocity(torch.nn.Module):
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return true_velocity
        
        interpolant = Interpolant(velocity_model=PerfectVelocity())
        
        loss = interpolant.compute_loss(x0_ref, x1_ref)
        
        assert loss.item() < 1e-6, "Loss should be near zero for perfect velocity prediction"


class TestSampling:
    """Test the ODE sampling procedure."""
    
    def test_sample_converges_to_target(self):
        """Test that sampling with true velocity converges to target."""
        from vdm.interpolant_model import Interpolant
        
        batch_size = 2
        x0 = torch.zeros(batch_size, 3, 16, 16)
        x1 = torch.ones(batch_size, 3, 16, 16) * 2.0  # Target is 2.0 everywhere
        
        class ExactVelocity(torch.nn.Module):
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                # True velocity is constant: x1 - x0 = 2.0
                return torch.ones_like(x) * 2.0
        
        interpolant = Interpolant(velocity_model=ExactVelocity())
        
        # Sample with many steps for accuracy
        samples = interpolant.sample(x0, conditioning=None, n_steps=100)
        
        # Should converge to x1
        assert torch.allclose(samples, x1, atol=0.1), f"Samples should converge to target. Got {samples.mean()}, expected 2.0"
    
    def test_sample_trajectory(self):
        """Test that trajectory has correct length."""
        from vdm.interpolant_model import Interpolant
        
        class DummyVelocity(torch.nn.Module):
            def forward(self, t, x, conditioning=None, param_conditioning=None):
                return torch.zeros_like(x)
        
        interpolant = Interpolant(velocity_model=DummyVelocity())
        
        batch_size = 2
        x0 = torch.randn(batch_size, 3, 16, 16)
        n_steps = 10
        
        trajectory = interpolant.sample(x0, n_steps=n_steps, return_trajectory=True)
        
        # Trajectory should have n_steps + 1 elements (including initial)
        assert len(trajectory) == n_steps + 1


class TestFactoryFunction:
    """Test the create_interpolant_model factory function."""
    
    @pytest.mark.skipif(
        not hasattr(pytest.importorskip("vdm.networks_clean"), 'UNet'),
        reason="UNet not available"
    )
    def test_create_interpolant_model(self):
        """Test that factory function creates valid model."""
        from vdm.interpolant_model import create_interpolant_model
        
        model = create_interpolant_model(
            output_channels=3,
            conditioning_channels=4,
            embedding_dim=64,  # Small for testing
            n_blocks=4,
            norm_groups=4,
            n_attention_heads=4,
            learning_rate=1e-4,
            n_sampling_steps=10,
            use_fourier_features=False,  # Disable for simpler test
            add_attention=False,
            x0_mode='zeros',
        )
        
        # Check it's the right type
        from vdm.interpolant_model import LightInterpolant
        assert isinstance(model, LightInterpolant)
        
        # Check sampling works
        model.eval()
        conditioning = torch.randn(1, 4, 32, 32)
        with torch.no_grad():
            samples = model.draw_samples(conditioning, batch_size=1, n_sampling_steps=5)
        assert samples.shape == (1, 3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
