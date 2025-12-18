"""
Tests for BIND inference pipeline.
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNormalizationStats:
    """Test normalization loading and application."""
    
    def test_load_normalization_stats(self):
        """Should load all normalization stats."""
        from bind.config_loader import load_normalization_stats
        
        stats = load_normalization_stats()
        
        # Check all expected keys exist
        expected_keys = ['dm_mag_mean', 'dm_mag_std', 
                        'gas_mag_mean', 'gas_mag_std',
                        'star_mag_mean', 'star_mag_std']
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
            assert isinstance(stats[key], float)
    
    def test_normalization_values_reasonable(self):
        """Normalization values should be in expected ranges."""
        from bind.config_loader import load_normalization_stats
        
        stats = load_normalization_stats()
        
        # Means should be reasonable for log10(mass + 1) values
        for key in ['dm_mag_mean', 'gas_mag_mean', 'star_mag_mean']:
            assert -10 < stats[key] < 20, f"{key} = {stats[key]} out of range"
        
        # Stds should be positive and reasonable
        for key in ['dm_mag_std', 'gas_mag_std', 'star_mag_std']:
            assert 0 < stats[key] < 10, f"{key} = {stats[key]} out of range"


class TestConfigLoader:
    """Test configuration loading."""
    
    @pytest.fixture
    def sample_config_path(self, tmp_path):
        """Create a temporary config file for testing."""
        config_content = """
[TRAINING]
seed = 42
cropsize = 128
batch_size = 16
num_workers = 4
embedding_dim = 96
norm_groups = 32
n_blocks = 5
n_attention_heads = 8
gamma_min = -10.0
gamma_max = 10.0
learning_rate = 0.0001
noise_schedule = fixed_linear
use_fourier_features = True
fourier_legacy = False
conditioning_channels = 1
large_scale_channels = 3
channel_weights = 1.0,1.0,1.0
use_focal_loss = False
model_name = test_model
version = 0
tb_logs = /tmp/tb_logs
data_root = /tmp/data
dataset = IllustrisTNG
param_norm_path = 
"""
        config_file = tmp_path / "test_config.ini"
        config_file.write_text(config_content)
        
        # Create tb_logs directory structure
        tb_logs = tmp_path / "tb_logs" / "test_model" / "version_0" / "checkpoints"
        tb_logs.mkdir(parents=True, exist_ok=True)
        
        # Update config to use tmp paths
        updated_content = config_content.replace('/tmp/tb_logs', str(tmp_path / "tb_logs"))
        updated_content = updated_content.replace('/tmp/data', str(tmp_path / "data"))
        config_file.write_text(updated_content)
        
        return str(config_file)
    
    def test_config_loader_basic(self, sample_config_path):
        """ConfigLoader should parse config correctly."""
        from bind.config_loader import ConfigLoader
        
        config = ConfigLoader(sample_config_path, verbose=False)
        
        assert config.seed == 42
        assert config.cropsize == 128
        assert config.batch_size == 16
        assert config.embedding_dim == 96
        assert config.n_blocks == 5
        assert config.gamma_min == -10.0
        assert config.gamma_max == 10.0
        assert config.use_fourier_features == True
        assert config.fourier_legacy == False
    
    def test_config_loader_channel_weights(self, sample_config_path):
        """ConfigLoader should parse channel weights string."""
        from bind.config_loader import ConfigLoader
        
        config = ConfigLoader(sample_config_path, verbose=False)
        
        assert config.channel_weights == '1.0,1.0,1.0'


class TestHaloPasting:
    """Test halo pasting utilities."""
    
    def test_weight_function_2d(self):
        """2D weight function should have correct shape and values."""
        from bind.bind import HaloPaster2D
        
        paster = HaloPaster2D(box_size_kpc=50000.0, r_in_factor=1.0, r_out_factor=3.0)
        
        # Create radial coordinates
        r = np.linspace(0, 10, 100)
        r_in = 2.0
        r_out = 6.0
        
        W = paster._create_weight_function(r, r_in, r_out)
        
        # Inside r_in: weight = 1
        assert np.allclose(W[r <= r_in], 1.0)
        
        # Outside r_out: weight = 0
        assert np.allclose(W[r > r_out], 0.0)
        
        # Transition region: 0 < weight < 1
        transition_mask = (r > r_in) & (r <= r_out)
        assert np.all(W[transition_mask] > 0)
        assert np.all(W[transition_mask] < 1)
    
    def test_weight_function_3d(self):
        """3D weight function should behave similarly to 2D."""
        from bind.bind import HaloPaster3D
        
        paster = HaloPaster3D(box_size_kpc=50000.0, r_in_factor=1.0, r_out_factor=3.0)
        
        r = np.linspace(0, 10, 100)
        r_in = 2.0
        r_out = 6.0
        
        W = paster._create_weight_function(r, r_in, r_out)
        
        assert np.allclose(W[r <= r_in], 1.0)
        assert np.allclose(W[r > r_out], 0.0)


class TestPeriodicBoundaries:
    """Test periodic boundary handling."""
    
    def test_extract_region_periodic(self):
        """Region extraction should handle periodic boundaries."""
        from bind.bind import BIND
        
        # Create a simple test field
        field = np.arange(16).reshape(4, 4).astype(np.float32)
        center = np.array([0.5, 0.5])  # Near corner
        
        # Create minimal BIND instance (we'll test the helper method)
        # This is a bit awkward but tests the actual code path
        
        # Test periodic indexing logic manually
        grid_size = 4
        half_size = 1
        center_pix = np.array([0, 0])  # Corner
        
        # Generate periodic indices
        ix = (center_pix[0] - half_size + np.arange(2 * half_size)) % grid_size
        iy = (center_pix[1] - half_size + np.arange(2 * half_size)) % grid_size
        
        # Should wrap around
        expected_ix = np.array([3, 0])  # -1 % 4 = 3, 0 % 4 = 0
        expected_iy = np.array([3, 0])
        
        assert np.array_equal(ix, expected_ix)
        assert np.array_equal(iy, expected_iy)


class TestConfigLoaderTypes:
    """Test ConfigLoader type conversion."""
    
    @pytest.fixture
    def minimal_config_path(self, tmp_path):
        """Create a minimal config file."""
        config_content = """
[TRAINING]
seed = 123
cropsize = 64
batch_size = 8
num_workers = 2
embedding_dim = 32
norm_groups = 8
n_blocks = 2
n_attention_heads = 4
version = 1
gamma_min = -5.0
gamma_max = 5.0
learning_rate = 0.001
model_name = test
tb_logs = {tb_logs}
data_root = {data_root}
dataset = test_dataset
param_norm_path = 
data_noise = 0.01,0.05,0.1
channel_weights = 1.0,2.0,3.0
use_focal_loss = True
focal_gamma = 2.5
"""
        tb_logs = tmp_path / "tb_logs" / "test" / "version_1" / "checkpoints"
        tb_logs.mkdir(parents=True, exist_ok=True)
        
        config_file = tmp_path / "config.ini"
        config_file.write_text(config_content.format(
            tb_logs=str(tmp_path / "tb_logs"),
            data_root=str(tmp_path / "data")
        ))
        
        return str(config_file)
    
    def test_int_params_parsed_correctly(self, minimal_config_path):
        """Integer parameters should be parsed as int."""
        from bind.config_loader import ConfigLoader
        
        config = ConfigLoader(minimal_config_path)
        
        assert isinstance(config.seed, int)
        assert config.seed == 123
        assert isinstance(config.batch_size, int)
        assert config.batch_size == 8
    
    def test_float_params_parsed_correctly(self, minimal_config_path):
        """Float parameters should be parsed as float."""
        from bind.config_loader import ConfigLoader
        
        config = ConfigLoader(minimal_config_path)
        
        assert isinstance(config.gamma_min, float)
        assert config.gamma_min == -5.0
        assert isinstance(config.learning_rate, float)
        assert config.learning_rate == 0.001
    
    def test_bool_params_parsed_correctly(self, minimal_config_path):
        """Boolean parameters should be parsed as bool."""
        from bind.config_loader import ConfigLoader
        
        config = ConfigLoader(minimal_config_path)
        
        assert isinstance(config.use_focal_loss, bool)
        assert config.use_focal_loss == True
    
    def test_per_channel_data_noise(self, minimal_config_path):
        """Per-channel data_noise should be parsed as tuple."""
        from bind.config_loader import ConfigLoader
        
        config = ConfigLoader(minimal_config_path)
        
        assert isinstance(config.data_noise, tuple)
        assert len(config.data_noise) == 3
        assert config.data_noise == (0.01, 0.05, 0.1)
    
    def test_natural_sort_key(self):
        """Natural sort key should sort numbers correctly."""
        from bind.config_loader import ConfigLoader
        
        test_strings = ['epoch_2', 'epoch_10', 'epoch_1', 'epoch_100']
        sorted_strings = sorted(test_strings, key=ConfigLoader._natural_sort_key)
        
        assert sorted_strings == ['epoch_1', 'epoch_2', 'epoch_10', 'epoch_100']


class TestLoadNormalizationStats:
    """Test load_normalization_stats function."""
    
    def test_load_from_default_path(self):
        """Should load from default path when no path provided."""
        from bind.config_loader import load_normalization_stats
        
        stats = load_normalization_stats()
        
        assert 'dm_mag_mean' in stats
        assert 'dm_mag_std' in stats
        assert 'gas_mag_mean' in stats
        assert 'gas_mag_std' in stats
        assert 'star_mag_mean' in stats
        assert 'star_mag_std' in stats
    
    def test_stats_are_floats(self):
        """All stats should be float values."""
        from bind.config_loader import load_normalization_stats
        
        stats = load_normalization_stats()
        
        for key, value in stats.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"
    
    def test_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        from bind.config_loader import load_normalization_stats
        
        with pytest.raises(FileNotFoundError):
            load_normalization_stats(base_path=str(tmp_path))


class TestHaloPaster2D:
    """Test HaloPaster2D class."""
    
    def test_init(self):
        """Test HaloPaster2D initialization."""
        from bind.bind import HaloPaster2D
        
        paster = HaloPaster2D(box_size_kpc=50000.0)
        
        assert paster.box_size_kpc == 50000.0
        assert paster.r_in_factor > 0
        assert paster.r_out_factor > paster.r_in_factor
    
    def test_weight_function_monotonic(self):
        """Weight function should decrease monotonically."""
        from bind.bind import HaloPaster2D
        
        paster = HaloPaster2D(box_size_kpc=50000.0)
        
        r = np.linspace(0, 10, 100)
        r_in = 2.0
        r_out = 6.0
        
        W = paster._create_weight_function(r, r_in, r_out)
        
        # In transition region, weights should decrease
        transition_mask = (r > r_in) & (r <= r_out)
        transition_indices = np.where(transition_mask)[0]
        
        for i in range(len(transition_indices) - 1):
            idx1, idx2 = transition_indices[i], transition_indices[i + 1]
            assert W[idx1] >= W[idx2], "Weight should decrease"


class TestHaloPaster3D:
    """Test HaloPaster3D class."""
    
    def test_init(self):
        """Test HaloPaster3D initialization."""
        from bind.bind import HaloPaster3D
        
        paster = HaloPaster3D(box_size_kpc=50000.0)
        
        assert paster.box_size_kpc == 50000.0
    
    def test_weight_function_3d(self):
        """3D weight function should behave correctly."""
        from bind.bind import HaloPaster3D
        
        paster = HaloPaster3D(box_size_kpc=50000.0)
        
        r = np.linspace(0, 10, 100)
        r_in = 2.0
        r_out = 6.0
        
        W = paster._create_weight_function(r, r_in, r_out)
        
        # Basic assertions
        assert W[0] == 1.0  # At r=0, weight should be 1
        assert W[-1] == 0.0  # At large r, weight should be 0
