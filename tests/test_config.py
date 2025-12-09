"""
Configuration and path validation tests.
"""
import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestConfigPaths:
    """Test configuration path setup."""
    
    def test_project_root_exists(self):
        """Project root should exist."""
        assert os.path.isdir(config.PROJECT_ROOT)
    
    def test_data_dir_exists(self):
        """Data directory should exist."""
        assert os.path.isdir(config.DATA_DIR)
    
    def test_normalization_stats_exist(self):
        """All normalization stats files should exist."""
        assert os.path.isfile(config.DM_NORM_STATS), f"Missing: {config.DM_NORM_STATS}"
        assert os.path.isfile(config.GAS_NORM_STATS), f"Missing: {config.GAS_NORM_STATS}"
        assert os.path.isfile(config.STELLAR_NORM_STATS), f"Missing: {config.STELLAR_NORM_STATS}"
    
    def test_validate_paths_required(self):
        """Required paths should all exist."""
        results = config.validate_paths(required_only=True)
        for name, (path, exists) in results.items():
            assert exists, f"Required path missing: {name} = {path}"
    
    def test_get_config_paths(self):
        """get_config_paths should return expected keys."""
        paths = config.get_config_paths()
        expected_keys = ['PROJECT_ROOT', 'DATA_DIR', 'TRAIN_DATA_ROOT', 
                        'CAMELS_ROOT', 'BIND_OUTPUT_ROOT', 'TB_LOGS_ROOT']
        for key in expected_keys:
            assert key in paths


class TestNormalizationStats:
    """Test normalization statistics files."""
    
    def test_dm_stats_content(self):
        """DM stats should have expected keys."""
        import numpy as np
        stats = np.load(config.DM_NORM_STATS)
        assert 'dm_mag_mean' in stats
        assert 'dm_mag_std' in stats
        assert stats['dm_mag_std'] > 0  # std should be positive
    
    def test_gas_stats_content(self):
        """Gas stats should have expected keys."""
        import numpy as np
        stats = np.load(config.GAS_NORM_STATS)
        assert 'gas_mag_mean' in stats
        assert 'gas_mag_std' in stats
        assert stats['gas_mag_std'] > 0
    
    def test_stellar_stats_content(self):
        """Stellar stats should have expected keys."""
        import numpy as np
        stats = np.load(config.STELLAR_NORM_STATS)
        assert 'star_mag_mean' in stats
        assert 'star_mag_std' in stats
        assert stats['star_mag_std'] > 0


class TestEnvironmentOverrides:
    """Test that environment variables override defaults."""
    
    def test_train_data_root_override(self):
        """TRAIN_DATA_ROOT env var should override default."""
        import importlib
        original = os.environ.get('TRAIN_DATA_ROOT')
        try:
            os.environ['TRAIN_DATA_ROOT'] = '/custom/path'
            # Reload config module
            importlib.reload(config)
            assert config.TRAIN_DATA_ROOT == '/custom/path'
        finally:
            # Restore original
            if original:
                os.environ['TRAIN_DATA_ROOT'] = original
            else:
                os.environ.pop('TRAIN_DATA_ROOT', None)
            importlib.reload(config)
