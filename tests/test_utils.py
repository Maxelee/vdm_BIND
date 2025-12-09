"""
Comprehensive tests for vdm/utils.py module.
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKLStdNormal:
    """Test KL divergence to standard normal."""
    
    def test_kl_zero_for_standard_normal(self):
        """KL should be zero when mean=0 and var=1 (standard normal)."""
        from vdm.utils import kl_std_normal
        
        mean_squared = torch.zeros(10)
        var = torch.ones(10)
        kl = kl_std_normal(mean_squared, var)
        
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)
    
    def test_kl_positive_for_non_standard(self):
        """KL should be positive for non-standard distributions."""
        from vdm.utils import kl_std_normal
        
        # Non-zero mean
        mean_squared = torch.ones(10)
        var = torch.ones(10)
        kl = kl_std_normal(mean_squared, var)
        assert torch.all(kl > 0)
        
        # Non-unit variance
        mean_squared = torch.zeros(10)
        var = torch.ones(10) * 2.0
        kl = kl_std_normal(mean_squared, var)
        assert torch.all(kl > 0)
    
    def test_kl_handles_small_variance(self):
        """KL should handle small variance without NaN."""
        from vdm.utils import kl_std_normal
        
        mean_squared = torch.zeros(10)
        var = torch.ones(10) * 1e-10
        kl = kl_std_normal(mean_squared, var)
        
        assert not torch.any(torch.isnan(kl))
        assert not torch.any(torch.isinf(kl))


class TestFixedLinearSchedule:
    """Test fixed linear noise schedule."""
    
    def test_endpoints(self):
        """Test schedule at t=0 and t=1."""
        from vdm.utils import FixedLinearSchedule
        
        schedule = FixedLinearSchedule(gamma_min=-10.0, gamma_max=10.0)
        
        gamma_0 = schedule(torch.tensor([0.0]))
        gamma_1 = schedule(torch.tensor([1.0]))
        
        assert torch.isclose(gamma_0, torch.tensor([-10.0]), atol=1e-6)
        assert torch.isclose(gamma_1, torch.tensor([10.0]), atol=1e-6)
    
    def test_midpoint(self):
        """Test schedule at t=0.5."""
        from vdm.utils import FixedLinearSchedule
        
        schedule = FixedLinearSchedule(gamma_min=-10.0, gamma_max=10.0)
        gamma_half = schedule(torch.tensor([0.5]))
        
        assert torch.isclose(gamma_half, torch.tensor([0.0]), atol=1e-6)
    
    def test_monotonic(self):
        """Schedule should be monotonically increasing."""
        from vdm.utils import FixedLinearSchedule
        
        schedule = FixedLinearSchedule(gamma_min=-10.0, gamma_max=10.0)
        times = torch.linspace(0, 1, 100)
        gammas = schedule(times)
        
        # Check monotonically increasing
        diffs = gammas[1:] - gammas[:-1]
        assert torch.all(diffs > 0)
    
    def test_batch_processing(self):
        """Schedule should handle batched inputs."""
        from vdm.utils import FixedLinearSchedule
        
        schedule = FixedLinearSchedule(gamma_min=-5.0, gamma_max=5.0)
        times = torch.rand(32)
        gammas = schedule(times)
        
        assert gammas.shape == times.shape


class TestLearnedLinearSchedule:
    """Test learned linear noise schedule."""
    
    def test_initial_values(self):
        """Test initial schedule matches linear."""
        from vdm.utils import LearnedLinearSchedule
        
        schedule = LearnedLinearSchedule(gamma_min=-10.0, gamma_max=10.0)
        
        gamma_0 = schedule(torch.tensor([0.0]))
        gamma_1 = schedule(torch.tensor([1.0]))
        
        # Initial values should be close to fixed linear
        assert torch.isclose(gamma_0, torch.tensor([-10.0]), atol=1e-5)
        assert torch.isclose(gamma_1, torch.tensor([10.0]), atol=1e-5)
    
    def test_learnable_parameters(self):
        """Parameters should be learnable."""
        from vdm.utils import LearnedLinearSchedule
        
        schedule = LearnedLinearSchedule(gamma_min=-10.0, gamma_max=10.0)
        
        # Should have 2 parameters: b and w
        params = list(schedule.parameters())
        assert len(params) == 2
        assert all(p.requires_grad for p in params)
    
    def test_monotonic_preserved(self):
        """Schedule should remain monotonic (w is made positive via abs)."""
        from vdm.utils import LearnedLinearSchedule
        
        schedule = LearnedLinearSchedule(gamma_min=-10.0, gamma_max=10.0)
        times = torch.linspace(0, 1, 100)
        gammas = schedule(times)
        
        # Check monotonically increasing
        diffs = gammas[1:] - gammas[:-1]
        assert torch.all(diffs >= 0)
    
    def test_gamma_min_max_clamp(self):
        """Test gamma_min_max clamping works."""
        from vdm.utils import LearnedLinearSchedule
        
        schedule = LearnedLinearSchedule(gamma_min=-10.0, gamma_max=10.0, gamma_min_max=-5.0)
        
        # The b parameter should be clamped to <= -5.0
        gamma_0 = schedule(torch.tensor([0.0]))
        assert gamma_0 <= -5.0


class TestNNSchedule:
    """Test neural network noise schedule."""
    
    def test_approximate_linear(self):
        """Initial NN schedule should be approximately linear."""
        from vdm.utils import NNSchedule
        
        schedule = NNSchedule(gamma_min=-10.0, gamma_max=10.0)
        
        gamma_0 = schedule(torch.tensor([0.0]))
        gamma_1 = schedule(torch.tensor([1.0]))
        
        # Should be close to endpoints (within some tolerance due to NN component)
        assert gamma_0 < -8.0  # Approximately gamma_min
        assert gamma_1 > 8.0   # Approximately gamma_max
    
    def test_learnable_parameters(self):
        """Should have learnable parameters."""
        from vdm.utils import NNSchedule
        
        schedule = NNSchedule(gamma_min=-10.0, gamma_max=10.0)
        params = list(schedule.parameters())
        
        # l1, l2, l3 each have weight and bias
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
    
    def test_batched_input(self):
        """Should handle batched inputs."""
        from vdm.utils import NNSchedule
        
        schedule = NNSchedule(gamma_min=-10.0, gamma_max=10.0)
        
        # 2D input
        times = torch.rand(32, 1)
        gammas = schedule(times)
        assert gammas.shape == times.shape
        
        # 4D input (common in diffusion)
        times_4d = torch.rand(32, 1, 1, 1)
        gammas_4d = schedule(times_4d)
        assert gammas_4d.shape == times_4d.shape


class TestZeroInit:
    """Test zero initialization utility."""
    
    def test_zeros_all_parameters(self):
        """Should zero all parameters."""
        from vdm.utils import zero_init
        
        linear = torch.nn.Linear(10, 5)
        zero_init(linear)
        
        assert torch.all(linear.weight == 0)
        assert torch.all(linear.bias == 0)
    
    def test_returns_module(self):
        """Should return the same module."""
        from vdm.utils import zero_init
        
        linear = torch.nn.Linear(10, 5)
        result = zero_init(linear)
        
        assert result is linear


class TestMonotonicLinear:
    """Test monotonic linear layer."""
    
    def test_forward_shape(self):
        """Output shape should match expected."""
        from vdm.utils import MonotonicLinear
        
        layer = MonotonicLinear(10, 5)
        x = torch.randn(32, 10)
        y = layer(x)
        
        assert y.shape == (32, 5)
    
    def test_positive_weights(self):
        """Forward should use absolute value of weights."""
        from vdm.utils import MonotonicLinear
        
        layer = MonotonicLinear(10, 5)
        
        # Make some weights negative
        with torch.no_grad():
            layer.weight.fill_(-1.0)
        
        x = torch.ones(1, 10)
        y = layer(x)
        
        # Output should still be consistent (using abs weights)
        # With all weights = -1, abs = 1, so output = sum(x) + bias = 10 + bias
        expected = 10 * torch.ones(1, 5) + layer.bias
        assert torch.allclose(y, expected)
    
    def test_extra_repr(self):
        """Should have proper string representation."""
        from vdm.utils import MonotonicLinear
        
        layer = MonotonicLinear(10, 5, bias=True)
        repr_str = layer.extra_repr()
        
        assert 'in_features=10' in repr_str
        assert 'out_features=5' in repr_str
        assert 'bias=True' in repr_str


class TestPowerSpectrum:
    """Test power spectrum computation."""
    
    def test_power_output_shape(self):
        """Power spectrum should return correct shapes."""
        from vdm.utils import power
        
        # Create random field with batch and channel dims
        field = torch.randn(4, 1, 64, 64)
        k, P, N = power(field)
        
        # k bins should be up to kmax = min(64, 64) // 2 = 32
        assert len(k) == 32
        assert len(P) == 32
        assert len(N) == 32
    
    def test_power_positive(self):
        """Power spectrum should be positive."""
        from vdm.utils import power
        
        field = torch.randn(4, 1, 64, 64)
        k, P, N = power(field)
        
        assert torch.all(P >= 0)
    
    def test_power_cross_correlation(self):
        """Cross power spectrum should work."""
        from vdm.utils import power
        
        field1 = torch.randn(4, 1, 64, 64)
        field2 = torch.randn(4, 1, 64, 64)
        
        k, P, N = power(field1, field2)
        
        assert len(k) == 32
        assert len(P) == 32
    
    def test_pk_wrapper(self):
        """pk wrapper function should work."""
        from vdm.utils import pk
        
        fields = [torch.randn(1, 64, 64) for _ in range(3)]
        ks, pks, ns = pk(fields)
        
        assert ks.shape[0] == 3
        assert pks.shape[0] == 3
        assert ns.shape[0] == 3


class TestComputePk:
    """Test compute_pk function."""
    
    def test_compute_pk_shape(self):
        """compute_pk should return correct shapes."""
        from vdm.utils import compute_pk
        
        field = np.random.randn(1, 1, 64, 64).astype(np.float32)
        k, pk = compute_pk(field)
        
        assert len(k) == 32
        assert len(pk) == 32
    
    def test_compute_pk_cross(self):
        """compute_pk cross correlation should work."""
        from vdm.utils import compute_pk
        
        field_a = np.random.randn(1, 1, 64, 64).astype(np.float32) + 1.0
        field_b = np.random.randn(1, 1, 64, 64).astype(np.float32) + 1.0
        
        k, pk = compute_pk(field_a, field_b)
        
        assert len(k) == 32
        assert len(pk) == 32
    
    def test_compute_pk_requires_4d(self):
        """compute_pk should require 4D input."""
        from vdm.utils import compute_pk
        
        field_3d = np.random.randn(1, 64, 64).astype(np.float32)
        
        with pytest.raises(AssertionError):
            compute_pk(field_3d)
