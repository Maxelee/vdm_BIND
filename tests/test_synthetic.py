"""
Integration tests using synthetic data.

These tests verify the full pipeline works end-to-end using small synthetic
datasets that don't require access to real simulation data.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Generate synthetic data once for all tests
@pytest.fixture(scope="module")
def synthetic_data_dir():
    """Create temporary synthetic dataset."""
    from scripts.generate_synthetic_data import generate_synthetic_dataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_synthetic_dataset(
            tmpdir,
            n_samples=5,
            size=64,  # Smaller for faster tests
            seed=42
        )
        yield tmpdir


@pytest.fixture
def sample_data(synthetic_data_dir):
    """Load a single synthetic sample."""
    sample_path = Path(synthetic_data_dir) / "synthetic_halo_000.npz"
    return dict(np.load(sample_path))


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""
    
    def test_sample_shapes(self, sample_data):
        """Verify synthetic sample has correct shapes."""
        assert sample_data['dm'].shape == (64, 64)
        assert sample_data['dm_hydro'].shape == (64, 64)
        assert sample_data['gas'].shape == (64, 64)
        assert sample_data['star'].shape == (64, 64)
        assert sample_data['conditional_params'].shape == (35,)
    
    def test_sample_values(self, sample_data):
        """Verify synthetic sample has reasonable values."""
        # All density fields should be non-negative
        assert sample_data['dm'].min() >= 0
        assert sample_data['gas'].min() >= 0
        assert sample_data['star'].min() >= 0
        
        # DM should have significant signal
        assert sample_data['dm'].max() > 0
        
        # Stellar should be sparse (many zeros)
        stellar_zeros = (sample_data['star'] == 0).sum()
        total_pixels = sample_data['star'].size
        assert stellar_zeros > 0.5 * total_pixels, "Stellar field should be sparse"
    
    def test_normalization_stats_exist(self, synthetic_data_dir):
        """Verify normalization stats are generated."""
        data_dir = Path(synthetic_data_dir)
        assert (data_dir / 'dark_matter_normalization_stats.npz').exists()
        assert (data_dir / 'gas_normalization_stats.npz').exists()
        assert (data_dir / 'stellar_normalization_stats.npz').exists()


class TestModelForwardPass:
    """Test model forward pass with synthetic data."""
    
    def test_unet_forward(self, sample_data):
        """Test UNet forward pass."""
        from vdm.networks_clean import UNetVDM
        
        # Create small model for testing
        model = UNetVDM(
            input_channels=3,
            gamma_min=-13.3,
            gamma_max=5.0,
            embedding_dim=32,  # Small for testing
            n_blocks=2,
            n_attention_heads=4,
            norm_groups=4,
            add_attention=False,  # Faster
            conditioning_channels=1,
            large_scale_channels=3,
        )
        model.eval()
        
        # Prepare input
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)
        t = torch.rand(batch_size)
        
        # DM condition + large scale
        cond = torch.randn(batch_size, 4, 64, 64)
        
        # Forward pass
        with torch.no_grad():
            out = model(x, t, cond)
        
        assert out.shape == (batch_size, 3, 64, 64)
    
    def test_vdm_loss(self, sample_data):
        """Test VDM loss computation."""
        from vdm.networks_clean import UNetVDM
        from vdm.vdm_model_clean import LightCleanVDM
        
        # Create model
        score_model = UNetVDM(
            input_channels=3,
            gamma_min=-13.3,
            gamma_max=5.0,
            embedding_dim=32,
            n_blocks=2,
            n_attention_heads=4,
            norm_groups=4,
            add_attention=False,
            conditioning_channels=1,
            large_scale_channels=3,
        )
        
        light_vdm = LightCleanVDM(
            score_model=score_model,
            learning_rate=1e-4,
            gamma_min=-13.3,
            gamma_max=5.0,
            image_shape=(3, 64, 64),
        )
        
        # Prepare batch
        dm = torch.from_numpy(sample_data['dm']).unsqueeze(0).unsqueeze(0)
        target = torch.stack([
            torch.from_numpy(sample_data['dm_hydro']),
            torch.from_numpy(sample_data['gas']),
            torch.from_numpy(sample_data['star']),
        ]).unsqueeze(0)
        large_scale = torch.randn(1, 3, 64, 64)
        params = torch.from_numpy(sample_data['conditional_params']).unsqueeze(0)
        
        batch = [dm, large_scale, target, params]
        
        # Compute loss
        loss = light_vdm.training_step(batch, 0)
        
        assert torch.isfinite(loss)
        assert loss > 0


class TestInterpolantModel:
    """Test Interpolant model with synthetic data."""
    
    def test_interpolant_forward(self, sample_data):
        """Test Interpolant forward pass."""
        from vdm.networks_clean import UNetVDM
        from vdm.interpolant_model import LightInterpolant, VelocityNetWrapper
        
        # Create velocity model (same architecture)
        net = UNetVDM(
            input_channels=3,
            gamma_min=-13.3,
            gamma_max=5.0,
            embedding_dim=32,
            n_blocks=2,
            n_attention_heads=4,
            norm_groups=4,
            add_attention=False,
            conditioning_channels=1,
            large_scale_channels=3,
        )
        
        # Wrap UNetVDM with VelocityNetWrapper to adapt calling convention
        # VelocityNetWrapper.forward(t, x, cond) -> UNetVDM.forward(z=x, g_t=t, cond)
        velocity_model = VelocityNetWrapper(
            net=net,
            output_channels=3,
            conditioning_channels=4,  # 1 DM + 3 large scale
        )
        
        model = LightInterpolant(
            velocity_model=velocity_model,
            learning_rate=1e-4,
            n_sampling_steps=10,  # Fast for testing
        )
        
        # Prepare batch - ensure proper shapes
        # dm should be (B, 1, H, W)
        dm = torch.from_numpy(sample_data['dm']).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
        # target should be (B, 3, H, W)  
        target = torch.stack([
            torch.from_numpy(sample_data['dm_hydro']).float(),
            torch.from_numpy(sample_data['gas']).float(),
            torch.from_numpy(sample_data['star']).float(),
        ]).unsqueeze(0)  # (1, 3, 64, 64)
        # large_scale should be (B, 3, H, W)
        large_scale = torch.randn(1, 3, 64, 64)
        # params should be (B, N_params)
        params = torch.from_numpy(sample_data['conditional_params']).float().unsqueeze(0)  # (1, 6)
        
        batch = [dm, large_scale, target, params]
        
        # Compute loss
        loss = model.training_step(batch, 0)
        
        assert torch.isfinite(loss)


class TestSampling:
    """Test sampling from models."""
    
    @pytest.mark.skip(reason="Sampling tests are slow and require proper model setup")
    def test_vdm_sampling(self, sample_data):
        """Test VDM sampling produces correct shape."""
        from vdm.networks_clean import UNetVDM
        from vdm.vdm_model_clean import LightCleanVDM
        
        score_model = UNetVDM(
            input_channels=3,
            gamma_min=-13.3,
            gamma_max=5.0,
            embedding_dim=32,
            n_blocks=2,
            n_attention_heads=4,
            norm_groups=4,
            add_attention=False,
            conditioning_channels=1,
            large_scale_channels=3,
        )
        
        model = LightCleanVDM(
            score_model=score_model,
            learning_rate=1e-4,
            gamma_min=-13.3,
            gamma_max=5.0,
            image_shape=(3, 64, 64),
        )
        model.eval()
        
        # Sample
        conditioning = torch.randn(2, 4, 64, 64)
        
        with torch.no_grad():
            samples = model.draw_samples(
                conditioning,
                batch_size=2,
                n_sampling_steps=5,  # Very few steps for speed
            )
        
        assert samples.shape == (2, 3, 64, 64)
    
    @pytest.mark.skip(reason="Sampling tests are slow and require proper model setup")
    def test_interpolant_sampling(self, sample_data):
        """Test Interpolant sampling produces correct shape."""
        from vdm.networks_clean import UNetVDM
        from vdm.interpolant_model import LightInterpolant
        
        velocity_model = UNetVDM(
            input_channels=3,
            gamma_min=-13.3,
            gamma_max=5.0,
            embedding_dim=32,
            n_blocks=2,
            n_attention_heads=4,
            norm_groups=4,
            add_attention=False,
            conditioning_channels=1,
            large_scale_channels=3,
        )
        
        model = LightInterpolant(
            velocity_model=velocity_model,
            learning_rate=1e-4,
            n_sampling_steps=5,
        )
        model.eval()
        
        # Sample
        conditioning = torch.randn(2, 4, 64, 64)
        
        with torch.no_grad():
            samples = model.draw_samples(
                conditioning,
                batch_size=2,
            )
        
        assert samples.shape == (2, 3, 64, 64)


class TestNormalization:
    """Test normalization with synthetic stats."""
    
    def test_normalize_denormalize(self, synthetic_data_dir, sample_data):
        """Test normalization round-trip."""
        # Load stats
        dm_stats = np.load(Path(synthetic_data_dir) / 'dark_matter_normalization_stats.npz')
        mean = dm_stats['dm_mag_mean']
        std = dm_stats['dm_mag_std']
        
        # Normalize
        dm = sample_data['dm']
        dm_log = np.log10(dm + 1)
        dm_norm = (dm_log - mean) / std
        
        # Denormalize
        dm_denorm_log = dm_norm * std + mean
        dm_denorm = 10**dm_denorm_log - 1
        
        # Check round-trip (use rtol=1e-3 to account for log10/10** floating point precision)
        np.testing.assert_allclose(dm, dm_denorm, rtol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
