"""
VDM-BIND Path Configuration

This module defines all configurable paths used throughout the project.
Paths can be overridden via environment variables or by editing the defaults below.

Environment Variables:
    VDM_BIND_ROOT       - Project root directory (auto-detected)
    VDM_BIND_DATA       - Normalization stats and model artifacts
    TRAIN_DATA_ROOT     - Training data directory (halo cutouts)
    CAMELS_SIMS_ROOT    - CAMELS simulation base directory
    BIND_OUTPUT_ROOT    - BIND output directory
    TB_LOGS_ROOT        - TensorBoard logs directory
    
Example:
    export TRAIN_DATA_ROOT=/path/to/train_data_rotated2_128_cpu
    export BIND_OUTPUT_ROOT=/path/to/bind_outputs
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT ROOT (auto-detected)
# =============================================================================
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = os.environ.get('VDM_BIND_ROOT', str(_THIS_FILE.parent))

# =============================================================================
# DATA PATHS
# =============================================================================

# Project data directory (normalization stats, quantile transformers)
DATA_DIR = os.environ.get('VDM_BIND_DATA', os.path.join(PROJECT_ROOT, 'data'))

# Alias for backward compatibility
NORMALIZATION_STATS_DIR = DATA_DIR

# Normalization statistics files
DM_NORM_STATS = os.path.join(DATA_DIR, 'dark_matter_normalization_stats.npz')
GAS_NORM_STATS = os.path.join(DATA_DIR, 'gas_normalization_stats.npz')
STELLAR_NORM_STATS = os.path.join(DATA_DIR, 'stellar_normalization_stats.npz')
QUANTILE_TRANSFORMER = os.path.join(DATA_DIR, 'quantile_normalizer_stellar.pkl')

# =============================================================================
# TRAINING DATA PATHS
# =============================================================================

# Base directory for training data (halo cutouts with rotations)
TRAIN_DATA_ROOT = os.environ.get(
    'TRAIN_DATA_ROOT', 
    '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu'
)
TRAIN_DATA_TRAIN = os.path.join(TRAIN_DATA_ROOT, 'train')
TRAIN_DATA_TEST = os.path.join(TRAIN_DATA_ROOT, 'test')

# Full-resolution projected images (1024^2)
PROJECTED_IMAGES_ROOT = os.environ.get(
    'PROJECTED_IMAGES_ROOT',
    '/mnt/home/mlee1/ceph/train_data_1024'
)
PROJECTED_IMAGES_SB35 = os.path.join(PROJECTED_IMAGES_ROOT, 'projected_images')
PROJECTED_IMAGES_1P = os.path.join(PROJECTED_IMAGES_ROOT, 'projected_images_1P')

# =============================================================================
# SIMULATION PATHS (CAMELS)
# =============================================================================

CAMELS_ROOT = os.environ.get('CAMELS_SIMS_ROOT', '/mnt/ceph/users/camels')

# IllustrisTNG simulations
CAMELS_TNG_HYDRO = os.path.join(CAMELS_ROOT, 'Sims/IllustrisTNG/L50n512')
CAMELS_TNG_DM = os.path.join(CAMELS_ROOT, 'Sims/IllustrisTNG_DM/L50n512')
CAMELS_FOF = os.path.join(CAMELS_ROOT, 'FOF_Subfind/IllustrisTNG_DM/L50n512')

# Local simulation metadata (user's home)
LOCAL_SIMS_ROOT = os.environ.get('LOCAL_SIMS_ROOT', '/mnt/home/mlee1/Sims')
PARAM_FILE_1P = os.path.join(LOCAL_SIMS_ROOT, 'IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt')
PARAM_MINMAX_SB35 = os.path.join(LOCAL_SIMS_ROOT, 'IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv')
PARAM_DF_SB35 = '/mnt/home/mlee1/50Mpc_boxes/data/param_df.csv'

# =============================================================================
# OUTPUT PATHS
# =============================================================================

# BIND output directory
BIND_OUTPUT_ROOT = os.environ.get('BIND_OUTPUT_ROOT', '/mnt/home/mlee1/ceph/BIND2d')

# TensorBoard logs
TB_LOGS_ROOT = os.environ.get('TB_LOGS_ROOT', '/mnt/home/mlee1/ceph/tb_logs')

# Analysis output
ANALYSIS_OUTPUT_ROOT = os.environ.get('ANALYSIS_OUTPUT_ROOT', os.path.join(PROJECT_ROOT, 'outputs'))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config_paths():
    """Return dictionary of all configured paths for logging/debugging."""
    return {
        'PROJECT_ROOT': PROJECT_ROOT,
        'DATA_DIR': DATA_DIR,
        'TRAIN_DATA_ROOT': TRAIN_DATA_ROOT,
        'CAMELS_ROOT': CAMELS_ROOT,
        'BIND_OUTPUT_ROOT': BIND_OUTPUT_ROOT,
        'TB_LOGS_ROOT': TB_LOGS_ROOT,
    }


def validate_paths(required_only=True):
    """
    Validate that required paths exist.
    
    Args:
        required_only: If True, only check paths required for basic operation
        
    Returns:
        dict: {path_name: (path, exists)}
    """
    required = {
        'DATA_DIR': DATA_DIR,
        'DM_NORM_STATS': DM_NORM_STATS,
        'GAS_NORM_STATS': GAS_NORM_STATS,
        'STELLAR_NORM_STATS': STELLAR_NORM_STATS,
    }
    
    optional = {
        'TRAIN_DATA_ROOT': TRAIN_DATA_ROOT,
        'BIND_OUTPUT_ROOT': BIND_OUTPUT_ROOT,
        'QUANTILE_TRANSFORMER': QUANTILE_TRANSFORMER,
    }
    
    paths_to_check = required if required_only else {**required, **optional}
    
    results = {}
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        results[name] = (path, exists)
        
    return results


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("VDM-BIND Configuration")
    print("=" * 60)
    for name, path in get_config_paths().items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {name}: {path}")
    print("=" * 60)


if __name__ == '__main__':
    print_config()
