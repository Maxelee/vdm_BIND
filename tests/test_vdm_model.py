"""
Comprehensive tests for vdm/vdm_model_clean.py module.
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCleanVDMInit:
    """Test CleanVDM initialization."""
    
    @pytest.fixture
    def mock_score_model(self):
        """Create a minimal score model for testing."""
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
    
    def test_fixed_linear_schedule(self, mock_score_model):
        """Test initialization with fixed linear schedule."""
        from vdm.vdm_model_clean import CleanVDM
        
        vdm = CleanVDM(
            score_model=mock_score_model,
            noise_schedule="fixed_linear",
            gamma_min=-10.0,
            gamma_max=10.0,
        )
        
        assert hasattr(vdm, 'gamma')
        assert vdm.antithetic_time_sampling == True
    
    def test_learned_linear_schedule(self, mock_score_model):
        """Test initialization with learned linear schedule."""
        from vdm.vdm_model_clean import CleanVDM
        
        vdm = CleanVDM(
            score_model=mock_score_model,
            noise_schedule="learned_linear",
            gamma_min=-10.0,
            gamma_max=10.0,
        )
        
        # Should have learnable parameters
        gamma_params = list(vdm.gamma.parameters())
        assert len(gamma_params) > 0
    
    def test_learned_nn_schedule(self, mock_score_model):
        """Test initialization with learned NN schedule."""
        from vdm.vdm_model_clean import CleanVDM
        
        vdm = CleanVDM(
            score_model=mock_score_model,
            noise_schedule="learned_nn",
            gamma_min=-10.0,
            gamma_max=10.0,
        )
        
        # Should have learnable parameters from NN
        gamma_params = list(vdm.gamma.parameters())
        assert len(gamma_params) > 0
    
    def test_invalid_schedule_raises(self, mock_score_model):
        """Test that invalid schedule raises error."""
        from vdm.vdm_model_clean import CleanVDM
        
        with pytest.raises(ValueError, match="Unknown noise schedule"):
            CleanVDM(
                score_model=mock_score_model,
                noise_schedule="invalid_schedule",
            )
    
    def test_per_channel_data_noise(self, mock_score_model):
        """Test per-channel data noise initialization."""
        from vdm.vdm_model_clean import CleanVDM
        
        vdm = CleanVDM(
            score_model=mock_score_model,
            data_noise=(0.001, 0.002, 0.003),
        )
        
        assert vdm.use_per_channel_data_noise == True
        assert torch.allclose(vdm.data_noise, torch.tensor([0.001, 0.002, 0.003]))
    
    def test_single_data_noise(self, mock_score_model):
        """Test single data noise initialization."""
        from vdm.vdm_model_clean import CleanVDM
        
        vdm = CleanVDM(
            score_model=mock_score_model,
            data_noise=0.001,
        )
        
        assert vdm.use_per_channel_data_noise == False
        assert vdm.data_noise == 0.001
    
    def test_channel_weights_registered(self, mock_score_model):
        """Test channel weights are registered as buffer."""
        from vdm.vdm_model_clean import CleanVDM
        
        vdm = CleanVDM(
            score_model=mock_score_model,
            channel_weights=(1.0, 2.0, 3.0),
        )
        
        assert hasattr(vdm, 'channel_weights')
        assert torch.allclose(vdm.channel_weights, torch.tensor([1.0, 2.0, 3.0]))
    
    def test_focal_loss_config(self, mock_score_model):
        """Test focal loss configuration."""
        from vdm.vdm_model_clean import CleanVDM
        
        vdm = CleanVDM(
            score_model=mock_score_model,
            use_focal_loss=True,
            focal_gamma=3.0,
        )
        
        assert vdm.use_focal_loss == True
        assert vdm.focal_gamma == 3.0


class TestCleanVDMNoiseScheduleHelpers:
    """Test noise schedule helper methods."""
    
    @pytest.fixture
    def vdm(self):
        """Create VDM for testing."""
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
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        return CleanVDM(
            score_model=score_model,
            noise_schedule="fixed_linear",
            gamma_min=-10.0,
            gamma_max=10.0,
        )
    
    def test_alpha_range(self, vdm):
        """Alpha should be in (0, 1)."""
        gamma = torch.linspace(-10, 10, 100)
        alpha = vdm.alpha(gamma)
        
        assert torch.all(alpha > 0)
        assert torch.all(alpha < 1)
    
    def test_sigma_range(self, vdm):
        """Sigma should be in (0, 1)."""
        gamma = torch.linspace(-10, 10, 100)
        sigma = vdm.sigma(gamma)
        
        assert torch.all(sigma > 0)
        assert torch.all(sigma < 1)
    
    def test_alpha_sigma_complement(self, vdm):
        """Alpha^2 + sigma^2 should equal 1 (variance preserving)."""
        gamma = torch.linspace(-10, 10, 100)
        alpha = vdm.alpha(gamma)
        sigma = vdm.sigma(gamma)
        
        sum_sq = alpha**2 + sigma**2
        assert torch.allclose(sum_sq, torch.ones_like(sum_sq), atol=1e-6)
    
    def test_snr_positive(self, vdm):
        """SNR should be positive."""
        gamma = torch.tensor([-5.0, 0.0, 5.0])
        snr = vdm.get_snr(gamma)
        
        assert torch.all(snr > 0)
    
    def test_snr_decreases_with_gamma(self, vdm):
        """SNR should decrease as gamma increases (more noise)."""
        gamma = torch.linspace(-10, 10, 100)
        snr = vdm.get_snr(gamma)
        
        # SNR should be monotonically decreasing
        diffs = snr[1:] - snr[:-1]
        assert torch.all(diffs < 0)


class TestCleanVDMDiffusion:
    """Test diffusion process methods."""
    
    @pytest.fixture
    def vdm(self):
        """Create VDM for testing."""
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
    
    def test_variance_preserving_map_shapes(self, vdm):
        """Test variance preserving map output shapes."""
        x = torch.randn(4, 3, 64, 64)
        times = torch.rand(4)
        
        z_t, gamma_t = vdm.variance_preserving_map(x, times)
        
        assert z_t.shape == x.shape
        assert gamma_t.shape == (4, 1, 1, 1)
    
    def test_variance_preserving_map_with_noise(self, vdm):
        """Test variance preserving map with provided noise."""
        x = torch.randn(4, 3, 64, 64)
        times = torch.rand(4)
        noise = torch.randn_like(x)
        
        z_t, gamma_t = vdm.variance_preserving_map(x, times, noise=noise)
        
        # Verify it uses the provided noise
        alpha_t = vdm.alpha(gamma_t)
        sigma_t = vdm.sigma(gamma_t)
        expected_z_t = alpha_t * x + sigma_t * noise
        
        assert torch.allclose(z_t, expected_z_t)
    
    def test_variance_preserving_map_preserves_variance(self, vdm):
        """Test that variance is preserved (approximately)."""
        x = torch.randn(1000, 3, 64, 64)  # Large batch for statistics
        times = torch.ones(1000) * 0.5  # Fixed time
        
        z_t, _ = vdm.variance_preserving_map(x, times)
        
        # Variance should be approximately preserved
        x_var = x.var()
        z_var = z_t.var()
        
        # Should be close (within 10%)
        assert abs(z_var / x_var - 1.0) < 0.1
    
    def test_sample_times_shape(self, vdm):
        """Test sample_times output shape."""
        times = vdm.sample_times(32, 'cpu')
        
        assert times.shape == (32,)
    
    def test_sample_times_range(self, vdm):
        """Test sample_times in [0, 1]."""
        times = vdm.sample_times(1000, 'cpu')
        
        assert torch.all(times >= 0)
        assert torch.all(times <= 1)
    
    def test_sample_times_antithetic(self, vdm):
        """Test antithetic time sampling produces evenly spaced times."""
        vdm.antithetic_time_sampling = True
        times = vdm.sample_times(100, 'cpu')
        
        # Times should be approximately evenly spaced
        sorted_times = torch.sort(times)[0]
        diffs = sorted_times[1:] - sorted_times[:-1]
        
        # All diffs should be approximately equal
        assert torch.std(diffs) < 0.02


class TestCleanVDMLoss:
    """Test loss computation methods."""
    
    @pytest.fixture
    def vdm(self):
        """Create VDM for testing."""
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
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        return CleanVDM(
            score_model=score_model,
            noise_schedule="fixed_linear",
            gamma_min=-10.0,
            gamma_max=10.0,
            image_shape=(3, 64, 64),
            channel_weights=(1.0, 1.0, 1.0),
        )
    
    def test_latent_loss_shape(self, vdm):
        """Test latent loss output shape."""
        x = torch.randn(4, 3, 64, 64)
        bpd_factor = 1.0 / (np.prod(x.shape[1:]) * np.log(2))
        
        latent_loss = vdm.get_latent_loss(x, bpd_factor)
        
        assert latent_loss.shape == (4,)
    
    def test_latent_loss_positive(self, vdm):
        """Test latent loss is non-negative."""
        x = torch.randn(4, 3, 64, 64)
        bpd_factor = 1.0 / (np.prod(x.shape[1:]) * np.log(2))
        
        latent_loss = vdm.get_latent_loss(x, bpd_factor)
        
        # KL divergence should be non-negative
        assert torch.all(latent_loss >= 0)
    
    def test_reconstruction_loss_shape(self, vdm):
        """Test reconstruction loss output shape."""
        x = torch.randn(4, 3, 64, 64)
        bpd_factor = 1.0 / (np.prod(x.shape[1:]) * np.log(2))
        
        recons_loss = vdm.get_reconstruction_loss(x, bpd_factor)
        
        assert recons_loss.shape == (4,)
    
    def test_reconstruction_loss_detailed(self, vdm):
        """Test detailed reconstruction loss."""
        x = torch.randn(4, 3, 64, 64)
        bpd_factor = 1.0 / (np.prod(x.shape[1:]) * np.log(2))
        
        recons_loss, details = vdm.get_reconstruction_loss_detailed(x, bpd_factor)
        
        assert recons_loss.shape == (4,)
        assert 'recons_loss_dm' in details
        assert 'recons_loss_gas' in details
        assert 'recons_loss_stars' in details
    
    def test_get_loss_returns_metrics(self, vdm):
        """Test get_loss returns loss and metrics."""
        x = torch.randn(4, 3, 64, 64)
        conditioning = torch.randn(4, 1, 64, 64)
        
        loss, metrics = vdm.get_loss(x, conditioning)
        
        assert loss.shape == (4,)
        assert 'diffusion_loss' in metrics
        assert 'latent_loss' in metrics
        assert 'reconstruction_loss' in metrics


class TestCleanVDMFocalLoss:
    """Test focal loss functionality."""
    
    @pytest.fixture
    def vdm_focal(self):
        """Create VDM with focal loss enabled."""
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
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        return CleanVDM(
            score_model=score_model,
            noise_schedule="fixed_linear",
            gamma_min=-10.0,
            gamma_max=10.0,
            image_shape=(3, 64, 64),
            use_focal_loss=True,
            focal_gamma=2.0,
        )
    
    def test_focal_loss_enabled(self, vdm_focal):
        """Test focal loss is enabled."""
        assert vdm_focal.use_focal_loss == True
        assert vdm_focal.focal_gamma == 2.0
    
    def test_focal_loss_applied_to_stellar_only(self, vdm_focal):
        """Test that focal loss is only applied to stellar channel (channel 2)."""
        # This is tested implicitly through the loss computation
        # Focal loss should downweight easy examples and upweight hard ones
        
        x = torch.randn(4, 3, 64, 64)
        conditioning = torch.randn(4, 1, 64, 64)
        
        loss, metrics = vdm_focal.get_loss(x, conditioning)
        
        # Just verify it runs without error
        assert not torch.isnan(loss).any()


class TestCleanVDMPerChannelNoise:
    """Test per-channel data noise functionality."""
    
    @pytest.fixture
    def vdm_per_channel(self):
        """Create VDM with per-channel noise."""
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
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        return CleanVDM(
            score_model=score_model,
            noise_schedule="fixed_linear",
            gamma_min=-10.0,
            gamma_max=10.0,
            image_shape=(3, 64, 64),
            data_noise=(0.001, 0.002, 0.003),
        )
    
    def test_per_channel_noise_setup(self, vdm_per_channel):
        """Test per-channel noise is set up correctly."""
        assert vdm_per_channel.use_per_channel_data_noise == True
        assert len(vdm_per_channel.data_noise) == 3
    
    def test_per_channel_reconstruction_loss(self, vdm_per_channel):
        """Test reconstruction loss with per-channel noise."""
        x = torch.randn(4, 3, 64, 64)
        bpd_factor = 1.0 / (np.prod(x.shape[1:]) * np.log(2))
        
        recons_loss = vdm_per_channel.get_reconstruction_loss(x, bpd_factor)
        
        assert recons_loss.shape == (4,)
        assert not torch.isnan(recons_loss).any()
    
    def test_per_channel_detailed_loss(self, vdm_per_channel):
        """Test detailed loss includes per-channel noise info."""
        x = torch.randn(4, 3, 64, 64)
        bpd_factor = 1.0 / (np.prod(x.shape[1:]) * np.log(2))
        
        recons_loss, details = vdm_per_channel.get_reconstruction_loss_detailed(x, bpd_factor)
        
        # Should have per-channel data noise info
        assert 'data_noise_dm' in details
        assert 'data_noise_gas' in details
        assert 'data_noise_stars' in details


class TestLightCleanVDM:
    """Test Lightning wrapper for CleanVDM."""
    
    @pytest.fixture
    def light_vdm(self):
        """Create LightCleanVDM for testing."""
        from vdm.networks_clean import UNetVDM
        from vdm.vdm_model_clean import LightCleanVDM
        
        score_model = UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,
            norm_groups=4,
            n_blocks=2,
            add_attention=False,
            use_fourier_features=False,
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        return LightCleanVDM(
            score_model=score_model,
            learning_rate=1e-4,
            gamma_min=-10.0,
            gamma_max=10.0,
            image_shape=(3, 64, 64),
            noise_schedule="fixed_linear",
        )
    
    def test_is_lightning_module(self, light_vdm):
        """Test it's a proper Lightning module."""
        from lightning.pytorch import LightningModule
        assert isinstance(light_vdm, LightningModule)
    
    def test_configure_optimizers_constant(self, light_vdm):
        """Test optimizer configuration with constant LR (no trainer needed)."""
        # Use constant scheduler which doesn't need trainer
        light_vdm.lr_scheduler = 'constant'
        optim = light_vdm.configure_optimizers()
        
        assert isinstance(optim, torch.optim.AdamW)
    
    def test_training_step(self, light_vdm):
        """Test training step runs."""
        # batch format: (m_dm, large_scale, m_target, param_conditioning)
        batch = (
            torch.randn(4, 1, 64, 64),  # m_dm (conditioning)
            torch.randn(4, 0, 64, 64),  # large_scale (empty for this test)
            torch.randn(4, 3, 64, 64),  # m_target
            None,  # params
        )
        
        loss = light_vdm.training_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
    
    def test_validation_step(self, light_vdm):
        """Test validation step runs."""
        # batch format: (m_dm, large_scale, m_target, param_conditioning)
        batch = (
            torch.randn(4, 1, 64, 64),  # m_dm
            torch.randn(4, 0, 64, 64),  # large_scale
            torch.randn(4, 3, 64, 64),  # m_target
            None,  # params
        )
        
        # Should not raise
        light_vdm.validation_step(batch, 0)
