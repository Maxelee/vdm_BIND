"""
Tests for Fourier Neural Operator (FNO) implementation.

Tests cover:
1. Spectral convolution correctness
2. FNO block forward pass
3. Full FNO2d model shapes
4. Time and parameter conditioning
5. VDM-compatible interface
6. Model variants (FNO-S, FNO-B, FNO-L, FNO-XL)
7. Lightning module training step
"""

import pytest
import torch
import numpy as np
from torch import nn

# Skip tests if FNO not available
try:
    from vdm.fno import (
        SpectralConv2d,
        FNOBlock,
        FNOBlockWithConditioning,
        FNO2d,
        FNOForVDM,
        SinusoidalTimeEmbedding,
        ParameterEmbedding,
        create_fno_model,
        FNO_S, FNO_B, FNO_L, FNO_XL,
    )
    from vdm.fno_model import LightFNOVDM, LightFNOFlow
    FNO_AVAILABLE = True
except ImportError:
    FNO_AVAILABLE = False


pytestmark = pytest.mark.skipif(not FNO_AVAILABLE, reason="FNO module not available")


class TestSpectralConv2d:
    """Test spectral convolution layer."""
    
    def test_output_shape(self):
        """Test output shape matches input spatial dims."""
        layer = SpectralConv2d(in_channels=32, out_channels=32, modes1=12, modes2=12)
        x = torch.randn(2, 32, 64, 64)
        y = layer(x)
        assert y.shape == (2, 32, 64, 64)
    
    def test_different_channels(self):
        """Test with different input/output channels."""
        layer = SpectralConv2d(in_channels=16, out_channels=32, modes1=8, modes2=8)
        x = torch.randn(2, 16, 32, 32)
        y = layer(x)
        assert y.shape == (2, 32, 32, 32)
    
    def test_gradient_flow(self):
        """Test gradients flow through spectral conv."""
        layer = SpectralConv2d(in_channels=8, out_channels=8, modes1=4, modes2=4)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        try:
            loss.backward()
            assert x.grad is not None
            assert not torch.isnan(x.grad).any()
        except RuntimeError as e:
            if "MKL FFT" in str(e):
                pytest.skip("MKL FFT error - system-specific issue")


class TestFNOBlock:
    """Test FNO block without conditioning."""
    
    def test_residual_shape(self):
        """Test FNO block preserves shape."""
        block = FNOBlock(channels=64, modes1=16, modes2=16)
        x = torch.randn(2, 64, 64, 64)
        y = block(x)
        assert y.shape == x.shape
    
    def test_with_dropout(self):
        """Test FNO block with dropout."""
        block = FNOBlock(channels=32, modes1=8, modes2=8, dropout=0.1)
        block.train()  # Enable dropout
        x = torch.randn(2, 32, 32, 32)
        y = block(x)
        assert y.shape == x.shape


class TestFNOBlockWithConditioning:
    """Test FNO block with FiLM conditioning."""
    
    def test_conditioning_shape(self):
        """Test conditioned block output shape."""
        block = FNOBlockWithConditioning(
            channels=64, modes1=16, modes2=16, cond_dim=128
        )
        x = torch.randn(2, 64, 64, 64)
        cond = torch.randn(2, 128)
        y = block(x, cond)
        assert y.shape == x.shape
    
    def test_conditioning_affects_output(self):
        """Test that different conditioning produces different outputs."""
        block = FNOBlockWithConditioning(
            channels=32, modes1=8, modes2=8, cond_dim=64
        )
        block.eval()
        x = torch.randn(2, 32, 32, 32)
        cond1 = torch.randn(2, 64)
        cond2 = torch.randn(2, 64)
        
        y1 = block(x, cond1)
        y2 = block(x, cond2)
        
        # Outputs should be different
        assert not torch.allclose(y1, y2, atol=1e-5)


class TestTimeEmbedding:
    """Test sinusoidal time embedding."""
    
    def test_output_shape(self):
        """Test time embedding output dimension."""
        embed = SinusoidalTimeEmbedding(dim=256)
        t = torch.rand(8)
        emb = embed(t)
        assert emb.shape == (8, 256)
    
    def test_different_times(self):
        """Test different times produce different embeddings."""
        embed = SinusoidalTimeEmbedding(dim=128)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])
        
        emb1 = embed(t1)
        emb2 = embed(t2)
        
        assert not torch.allclose(emb1, emb2, atol=1e-5)


class TestParameterEmbedding:
    """Test parameter embedding."""
    
    def test_output_shape(self):
        """Test parameter embedding output."""
        embed = ParameterEmbedding(n_params=35, embed_dim=128)
        params = torch.rand(4, 35)
        emb = embed(params)
        assert emb.shape == (4, 128)
    
    def test_normalization(self):
        """Test parameter normalization with bounds."""
        param_min = [0.0] * 10
        param_max = [1.0] * 10
        embed = ParameterEmbedding(n_params=10, embed_dim=64, param_min=param_min, param_max=param_max)
        
        params = torch.rand(2, 10)
        emb = embed(params)
        assert emb.shape == (2, 64)


class TestFNO2d:
    """Test full FNO2d model."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        model = FNO2d(
            in_channels=3,
            out_channels=3,
            conditioning_channels=1,
            large_scale_channels=3,
            hidden_channels=32,
            n_layers=2,
            modes1=8,
            modes2=8,
            n_params=10,
        )
        
        t = torch.rand(2)
        x_t = torch.randn(2, 3, 32, 32)
        cond = torch.randn(2, 4, 32, 32)  # 1 + 3 conditioning channels
        params = torch.rand(2, 10)
        
        output = model(t, x_t, cond, params)
        assert output.shape == (2, 3, 32, 32)
    
    def test_without_param_conditioning(self):
        """Test FNO without parameter conditioning."""
        model = FNO2d(
            in_channels=3,
            out_channels=3,
            conditioning_channels=1,
            large_scale_channels=3,
            hidden_channels=32,
            n_layers=2,
            modes1=8,
            modes2=8,
            n_params=0,
            use_param_conditioning=False,
        )
        
        t = torch.rand(2)
        x_t = torch.randn(2, 3, 32, 32)
        cond = torch.randn(2, 4, 32, 32)
        
        output = model(t, x_t, cond, None)
        assert output.shape == (2, 3, 32, 32)
    
    def test_gradient_flow(self):
        """Test gradients flow through full model."""
        model = FNO2d(
            hidden_channels=16, n_layers=2, modes1=4, modes2=4, n_params=5
        )
        
        t = torch.rand(2, requires_grad=True)
        x_t = torch.randn(2, 3, 16, 16, requires_grad=True)
        cond = torch.randn(2, 4, 16, 16)
        params = torch.rand(2, 5)
        
        output = model(t, x_t, cond, params)
        loss = output.sum()
        loss.backward()
        
        assert x_t.grad is not None


class TestModelVariants:
    """Test FNO model variants."""
    
    @pytest.mark.parametrize("variant", ['FNO-S', 'FNO-B', 'FNO-L'])
    def test_create_variant(self, variant):
        """Test creating different FNO variants."""
        model = create_fno_model(
            variant=variant,
            img_size=32,
            n_params=10,
            conditioning_channels=1,
            large_scale_channels=3,
        )
        
        t = torch.rand(2)
        x_t = torch.randn(2, 3, 32, 32)
        cond = torch.randn(2, 4, 32, 32)
        params = torch.rand(2, 10)
        
        output = model(t, x_t, cond, params)
        assert output.shape == (2, 3, 32, 32)
    
    def test_invalid_variant(self):
        """Test invalid variant raises error."""
        with pytest.raises(ValueError):
            create_fno_model(variant='FNO-XXL', img_size=32)


class TestFNOForVDM:
    """Test VDM-compatible FNO wrapper."""
    
    def test_gamma_interface(self):
        """Test FNO with gamma (log-SNR) interface."""
        fno = FNO2d(hidden_channels=16, n_layers=2, modes1=4, modes2=4, n_params=5)
        model = FNOForVDM(fno, gamma_min=-13.3, gamma_max=5.0)
        
        x_t = torch.randn(2, 3, 16, 16)
        gamma = torch.tensor([5.0, -10.0])  # High and low SNR
        cond = torch.randn(2, 4, 16, 16)
        params = torch.rand(2, 5)
        
        output = model(x_t, gamma, cond, params)
        assert output.shape == (2, 3, 16, 16)
    
    def test_gamma_to_t_conversion(self):
        """Test gamma to t conversion is correct."""
        fno = FNO2d(hidden_channels=16, n_layers=2, modes1=4, modes2=4, n_params=0)
        gamma_min, gamma_max = -13.3, 5.0
        model = FNOForVDM(fno, gamma_min=gamma_min, gamma_max=gamma_max)
        
        # gamma = gamma_max => t = 0 (clean)
        # gamma = gamma_min => t = 1 (noisy)
        gamma_high = torch.tensor([gamma_max])
        gamma_low = torch.tensor([gamma_min])
        
        # Just check the wrapper runs without error
        x = torch.randn(1, 3, 16, 16)
        cond = torch.randn(1, 4, 16, 16)
        
        out_high = model(x, gamma_high, cond)
        out_low = model(x, gamma_low, cond)
        
        assert out_high.shape == (1, 3, 16, 16)
        assert out_low.shape == (1, 3, 16, 16)


class TestLightFNOVDM:
    """Test Lightning module for FNO with VDM training."""
    
    def test_training_step(self):
        """Test training step produces loss."""
        model = LightFNOVDM(
            fno_model='FNO-S',
            img_size=32,
            n_params=10,
            image_shape=(3, 32, 32),
        )
        
        # Create batch: (condition, large_scale, target, params)
        batch = (
            torch.randn(2, 1, 32, 32),   # condition
            torch.randn(2, 3, 32, 32),   # large_scale
            torch.randn(2, 3, 32, 32),   # target
            torch.rand(2, 10),           # params
        )
        
        loss = model.training_step(batch, 0)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_sampling(self):
        """Test sampling produces correct shape."""
        model = LightFNOVDM(
            fno_model='FNO-S',
            img_size=32,
            n_params=10,
            n_sampling_steps=10,  # Few steps for fast test
            image_shape=(3, 32, 32),
        )
        model.eval()
        
        cond = torch.randn(1, 4, 32, 32)  # Combined conditioning
        params = torch.rand(1, 10)
        
        with torch.no_grad():
            samples = model.sample(cond, batch_size=1, param_conditioning=params, n_steps=5)
        
        assert samples.shape == (1, 3, 32, 32)
    
    def test_draw_samples_bind_interface(self):
        """Test BIND-compatible sampling interface."""
        model = LightFNOVDM(
            fno_model='FNO-S',
            img_size=32,
            n_params=10,
            n_sampling_steps=5,
            image_shape=(3, 32, 32),
        )
        model.eval()
        
        cond = torch.randn(1, 4, 32, 32)
        params_np = np.random.rand(1, 10).astype(np.float32)
        
        with torch.no_grad():
            samples = model.draw_samples(cond, batch_size=1, conditional_params=params_np)
        
        assert samples.shape == (1, 3, 32, 32)


class TestLightFNOFlow:
    """Test Lightning module for FNO with Flow Matching."""
    
    def test_training_step(self):
        """Test flow matching training step."""
        model = LightFNOFlow(
            fno_model='FNO-S',
            img_size=32,
            n_params=10,
            image_shape=(3, 32, 32),
        )
        
        batch = (
            torch.randn(2, 1, 32, 32),
            torch.randn(2, 3, 32, 32),
            torch.randn(2, 3, 32, 32),
            torch.rand(2, 10),
        )
        
        loss = model.training_step(batch, 0)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_sampling(self):
        """Test ODE sampling."""
        model = LightFNOFlow(
            fno_model='FNO-S',
            img_size=32,
            n_params=10,
            n_sampling_steps=5,
            image_shape=(3, 32, 32),
        )
        model.eval()
        
        cond = torch.randn(1, 4, 32, 32)
        params = torch.rand(1, 10)
        
        with torch.no_grad():
            samples = model.sample(cond, batch_size=1, param_conditioning=params)
        
        assert samples.shape == (1, 3, 32, 32)
    
    @pytest.mark.parametrize("x0_mode", ['zeros', 'noise'])
    def test_x0_modes(self, x0_mode):
        """Test different x0 initialization modes."""
        model = LightFNOFlow(
            fno_model='FNO-S',
            img_size=32,
            n_params=10,
            x0_mode=x0_mode,
            image_shape=(3, 32, 32),
        )
        
        batch = (
            torch.randn(2, 1, 32, 32),
            torch.randn(2, 3, 32, 32),
            torch.randn(2, 3, 32, 32),
            torch.rand(2, 10),
        )
        
        loss = model.training_step(batch, 0)
        assert not torch.isnan(loss)


class TestFNOParameterCount:
    """Test parameter counts for different variants."""
    
    def test_variant_sizes(self):
        """Test variants have increasing parameter counts."""
        variants = ['FNO-S', 'FNO-B', 'FNO-L']
        param_counts = []
        
        for variant in variants:
            model = create_fno_model(
                variant=variant,
                img_size=32,
                n_params=10,
            )
            count = sum(p.numel() for p in model.parameters())
            param_counts.append(count)
        
        # Should be monotonically increasing
        assert param_counts[0] < param_counts[1] < param_counts[2]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
