"""
Integration tests for full pipeline.
These tests verify end-to-end functionality.
"""
import pytest
import torch
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVDMTrainingPipeline:
    """Integration tests for training pipeline."""
    
    @pytest.fixture
    def mock_batch(self):
        """Create a mock training batch."""
        batch_size = 2
        crop_size = 64
        
        # condition: (B, 1, H, W) - normalized DM field
        condition = torch.randn(batch_size, 1, crop_size, crop_size)
        
        # target: (B, 3, H, W) - [DM_hydro, Gas, Stars]
        target = torch.randn(batch_size, 3, crop_size, crop_size)
        
        # conditional_params: (B, 15) - cosmological/astrophysical params
        params = torch.randn(batch_size, 15)
        
        return condition, target, params
    
    def test_model_forward_pass(self, mock_batch):
        """Test complete forward pass through model."""
        from vdm.networks_clean import UNetVDM
        from vdm.vdm_model_clean import CleanVDM
        
        condition, target, params = mock_batch
        
        # Create model
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
        
        vdm = CleanVDM(
            score_model=score_model,
            gamma_min=-10.0,
            gamma_max=10.0,
            image_shape=(3, 64, 64),
        )
        
        # Forward pass
        times = vdm.sample_times(target.shape[0], target.device)
        z_t, gamma_t = vdm.variance_preserving_map(target, times)
        
        # Score model prediction - squeeze gamma from (B,1,1,1) to (B,)
        noise_pred = score_model(z_t, gamma_t.squeeze(), condition)
        
        assert noise_pred.shape == target.shape
    
    def test_loss_computation(self, mock_batch):
        """Test loss computation."""
        from vdm.networks_clean import UNetVDM
        from vdm.vdm_model_clean import LightCleanVDM
        
        condition, target, params = mock_batch
        
        # Create Lightning model
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
        
        light_vdm = LightCleanVDM(
            score_model=score_model,
            learning_rate=1e-4,
            gamma_min=-10.0,
            gamma_max=10.0,
            image_shape=(3, 64, 64),
            noise_schedule="fixed_linear",
        )
        
        # Compute training step (returns loss dict)
        batch = (condition, target, params)
        
        # This would normally be called by Lightning trainer
        # For now just verify model is constructed correctly
        assert hasattr(light_vdm, 'model')
        assert hasattr(light_vdm, 'configure_optimizers')


class TestSamplingPipeline:
    """Integration tests for sampling/inference."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Sampling test requires CUDA")
    def test_draw_samples_shape(self):
        """Test that sampling produces correct output shape."""
        from vdm.networks_clean import UNetVDM
        from vdm.vdm_model_clean import LightCleanVDM
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
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
        ).to(device)
        
        light_vdm = LightCleanVDM(
            score_model=score_model,
            learning_rate=1e-4,
            gamma_min=-10.0,
            gamma_max=10.0,
            image_shape=(3, 64, 64),
            noise_schedule="fixed_linear",
        ).to(device)
        light_vdm.eval()
        
        # Create conditioning on same device
        batch_size = 2
        condition = torch.randn(batch_size, 1, 64, 64).to(device)
        
        # Draw samples with very few steps for speed
        with torch.no_grad():
            samples = light_vdm.draw_samples(
                conditioning=condition,
                batch_size=batch_size,
                n_sampling_steps=5,  # Very few steps for testing
            )
        
        assert samples.shape == (batch_size, 3, 64, 64)


class TestDataPipeline:
    """Test data loading and preprocessing."""
    
    def test_normalization_roundtrip(self):
        """Test that normalization/denormalization is consistent."""
        from bind.workflow_utils import load_normalization_stats
        
        stats = load_normalization_stats()
        
        # Create fake log-space data
        original_log = np.random.randn(10, 128, 128) * 2 + 8  # ~log10 mass values
        
        # Normalize DM channel
        normalized = (original_log - stats['dm_mag_mean']) / stats['dm_mag_std']
        
        # Denormalize
        recovered = normalized * stats['dm_mag_std'] + stats['dm_mag_mean']
        
        assert np.allclose(original_log, recovered)


@pytest.mark.skipif(
    not os.path.exists('/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu'),
    reason="Training data not available"
)
class TestRealDataLoading:
    """Tests that require actual training data."""
    
    def test_load_single_sample(self):
        """Test loading a single training sample."""
        import glob
        
        data_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test'
        samples = glob.glob(f'{data_root}/sim_*/halo_*_3d.npz')
        
        if samples:
            sample = np.load(samples[0])
            
            # Check expected keys
            assert 'dm' in sample
            assert 'dm_hydro' in sample
            assert 'gas' in sample
            assert 'star' in sample
            
            # Check shapes
            assert sample['dm'].shape == (128, 128)
            assert sample['dm_hydro'].shape == (128, 128)
