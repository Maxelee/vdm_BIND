#!/usr/bin/env python
"""
VDM-BIND Test Runner

This script validates the codebase configuration and runs tests.

Usage:
    # Quick validation (check paths only)
    python run_tests.py --validate
    
    # Run all tests
    python run_tests.py
    
    # Run specific test file
    python run_tests.py tests/test_config.py
    
    # Run with verbose output
    python run_tests.py -v
    
    # Run with coverage
    python run_tests.py --cov
"""

import argparse
import subprocess
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def validate_configuration():
    """Validate that all required paths and files exist."""
    import config
    
    print("=" * 60)
    print("VDM-BIND Configuration Validation")
    print("=" * 60)
    
    all_ok = True
    
    # Check required paths
    required = config.validate_paths(required_only=True)
    print("\nRequired paths:")
    for name, (path, exists) in required.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    # Check optional paths
    print("\nOptional paths (training/inference):")
    optional_paths = {
        'TRAIN_DATA_ROOT': config.TRAIN_DATA_ROOT,
        'QUANTILE_TRANSFORMER': config.QUANTILE_TRANSFORMER,
        'BIND_OUTPUT_ROOT': config.BIND_OUTPUT_ROOT,
        'TB_LOGS_ROOT': config.TB_LOGS_ROOT,
    }
    for name, path in optional_paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "○"  # Circle for optional
        print(f"  {status} {name}: {path}")
    
    # Check Python imports
    print("\nPython imports:")
    imports_to_check = [
        ('torch', 'PyTorch'),
        ('lightning', 'PyTorch Lightning'),
        ('numpy', 'NumPy'),
        ('h5py', 'HDF5'),
        ('pandas', 'Pandas'),
    ]
    
    for module, name in imports_to_check:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_ok = False
    
    # Check optional imports
    print("\nOptional imports:")
    optional_imports = [
        ('MAS_library', 'Pylians MAS'),
        ('Pk_library', 'Pylians Pk'),
        ('joblib', 'Joblib'),
        ('pytest', 'Pytest'),
    ]
    
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ○ {name} - not installed (optional)")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All required components validated successfully!")
    else:
        print("✗ Some required components are missing. See above.")
    print("=" * 60)
    
    return all_ok


def run_tests(args):
    """Run pytest with specified arguments."""
    cmd = ['python', '-m', 'pytest']
    
    if args.verbose:
        cmd.append('-v')
    
    if args.cov:
        cmd.extend(['--cov=vdm', '--cov=bind', '--cov-report=term-missing'])
    
    if args.test_path:
        cmd.append(args.test_path)
    else:
        cmd.append('tests/')
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='VDM-BIND test runner and validator'
    )
    parser.add_argument(
        '--validate', '-V',
        action='store_true',
        help='Only validate configuration (no tests)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose test output'
    )
    parser.add_argument(
        '--cov',
        action='store_true',
        help='Run with coverage reporting'
    )
    parser.add_argument(
        'test_path',
        nargs='?',
        default=None,
        help='Specific test file or directory to run'
    )
    
    args = parser.parse_args()
    
    # Always validate first
    ok = validate_configuration()
    
    if args.validate:
        sys.exit(0 if ok else 1)
    
    if not ok:
        print("\nWarning: Some required components are missing.")
        print("Tests may fail. Continue anyway? [y/N]")
        response = input().strip().lower()
        if response != 'y':
            sys.exit(1)
    
    print("\n")
    sys.exit(run_tests(args))


if __name__ == '__main__':
    main()
