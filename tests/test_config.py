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


class TestParamNormalizationLoading:
    """Test flexible parameter normalization loading from train_unified."""
    
    def test_unconditional_none_path(self):
        """Test loading with None path (unconditional)."""
        from train_unified import load_param_normalization
        
        use_cond, min_vals, max_vals, n_params = load_param_normalization(None)
        
        assert use_cond is False
        assert min_vals is None
        assert max_vals is None
        assert n_params == 0
    
    def test_unconditional_none_string(self):
        """Test loading with 'none' string (unconditional)."""
        from train_unified import load_param_normalization
        
        use_cond, min_vals, max_vals, n_params = load_param_normalization('none')
        
        assert use_cond is False
        assert min_vals is None
        assert max_vals is None
        assert n_params == 0
    
    def test_unconditional_empty_string(self):
        """Test loading with empty string (unconditional)."""
        from train_unified import load_param_normalization
        
        use_cond, min_vals, max_vals, n_params = load_param_normalization('')
        
        assert use_cond is False
        assert n_params == 0
    
    def test_inline_params(self):
        """Test loading with inline param_min/max."""
        from train_unified import load_param_normalization
        
        use_cond, min_vals, max_vals, n_params = load_param_normalization(
            param_norm_path=None,
            param_min_inline='0.0,0.0,0.0',
            param_max_inline='1.0,1.0,1.0'
        )
        
        assert use_cond is True
        assert n_params == 3
        assert len(min_vals) == 3
        assert len(max_vals) == 3
        assert min_vals[0] == 0.0
        assert max_vals[0] == 1.0
    
    def test_inline_params_with_spaces(self):
        """Test inline params with spaces around commas."""
        from train_unified import load_param_normalization
        
        use_cond, min_vals, max_vals, n_params = load_param_normalization(
            param_norm_path=None,
            param_min_inline='0.0, 0.5, 1.0',
            param_max_inline='1.0, 1.5, 2.0'
        )
        
        assert use_cond is True
        assert n_params == 3
        assert min_vals[1] == 0.5
        assert max_vals[2] == 2.0
    
    def test_explicit_n_params_zero(self):
        """Test explicit n_params=0 for unconditional."""
        from train_unified import load_param_normalization
        
        use_cond, min_vals, max_vals, n_params = load_param_normalization(
            param_norm_path=None,
            n_params_inline='0'
        )
        
        assert use_cond is False
        assert n_params == 0
    
    def test_json_file_loading(self, tmp_path):
        """Test loading parameters from JSON file."""
        import json
        from train_unified import load_param_normalization
        
        # Create temp JSON file
        json_path = tmp_path / "params.json"
        json_data = {
            'param_min': [0.1, 0.5, 0.0],
            'param_max': [0.5, 1.0, 1.0],
            'param_names': ['omega_m', 'sigma_8', 'agn_eff']
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        
        use_cond, min_vals, max_vals, n_params = load_param_normalization(str(json_path))
        
        assert use_cond is True
        assert n_params == 3
        assert min_vals[0] == 0.1
        assert max_vals[1] == 1.0
    
    def test_csv_file_loading(self, tmp_path):
        """Test loading parameters from CSV file."""
        import pandas as pd
        from train_unified import load_param_normalization
        
        # Create temp CSV file
        csv_path = tmp_path / "params.csv"
        df = pd.DataFrame({
            'Parameter': ['omega_m', 'sigma_8'],
            'MinVal': [0.1, 0.5],
            'MaxVal': [0.5, 1.0]
        })
        df.to_csv(csv_path, index=False)
        
        use_cond, min_vals, max_vals, n_params = load_param_normalization(str(csv_path))
        
        assert use_cond is True
        assert n_params == 2
        assert min_vals[0] == 0.1
        assert max_vals[1] == 1.0
    
    def test_csv_alternative_columns(self, tmp_path):
        """Test loading CSV with 'min'/'max' columns instead of 'MinVal'/'MaxVal'."""
        import pandas as pd
        from train_unified import load_param_normalization
        
        csv_path = tmp_path / "params_alt.csv"
        df = pd.DataFrame({
            'name': ['p1', 'p2'],
            'min': [0.0, -1.0],
            'max': [1.0, 1.0]
        })
        df.to_csv(csv_path, index=False)
        
        use_cond, min_vals, max_vals, n_params = load_param_normalization(str(csv_path))
        
        assert use_cond is True
        assert n_params == 2
        assert min_vals[1] == -1.0
    
    def test_file_not_found_error(self):
        """Test that missing file raises appropriate error."""
        from train_unified import load_param_normalization
        
        with pytest.raises(FileNotFoundError):
            load_param_normalization('/nonexistent/path/params.csv')
    
    def test_inline_params_mismatched_lengths(self):
        """Test that mismatched inline param lengths raise error."""
        from train_unified import load_param_normalization
        
        with pytest.raises(ValueError, match="same length"):
            load_param_normalization(
                param_norm_path=None,
                param_min_inline='0.0,0.0',
                param_max_inline='1.0,1.0,1.0'
            )
