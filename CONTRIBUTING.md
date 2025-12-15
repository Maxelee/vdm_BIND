# Contributing to VDM-BIND

Thank you for your interest in contributing to VDM-BIND! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Adding New Models](#adding-new-models)
7. [Documentation](#documentation)

---

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/vdm_BIND.git
cd vdm_BIND
git remote add upstream https://github.com/Maxelee/vdm_BIND.git
```

### Branch Naming

Use the following prefixes for branches:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/dit-backbone` |
| `fix/` | Bug fixes | `fix/normalization-bug` |
| `docs/` | Documentation | `docs/api-reference` |
| `test/` | Testing | `test/synthetic-data` |
| `experiment/` | Research experiments | `experiment/new-loss` |

---

## Development Setup

```bash
# Create development environment
conda create -n vdm_bind_dev python=3.10
conda activate vdm_bind_dev

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install dev dependencies separately
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

---

## Code Style

We follow PEP 8 with some modifications:

### Formatting

```bash
# Format code with black
black vdm/ bind/ tests/

# Check linting
flake8 vdm/ bind/ --max-line-length=100
```

### Docstrings

Use Google-style docstrings:

```python
def sample(model, conditioning, batch_size=1, n_steps=50):
    """Generate samples from a trained model.
    
    Args:
        model: Trained generative model (VDM, Interpolant, etc.)
        conditioning: Conditioning tensor of shape (B, C, H, W)
        batch_size: Number of samples to generate
        n_steps: Number of sampling steps
    
    Returns:
        Generated samples of shape (B, 3, H, W)
    
    Example:
        >>> samples = sample(model, condition, batch_size=4)
        >>> print(samples.shape)
        torch.Size([4, 3, 128, 128])
    """
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, Tuple, Dict
import torch
from torch import Tensor

def load_checkpoint(
    path: str,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict[str, any]]:
    ...
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=vdm --cov=bind --cov-report=html

# Run specific test file
pytest tests/test_vdm_model.py -v

# Run tests matching a pattern
pytest tests/ -k "test_forward" -v
```

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_my_feature.py
import pytest
import torch
from vdm.my_module import MyClass

class TestMyClass:
    """Tests for MyClass."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return MyClass(dim=64)
    
    def test_forward_shape(self, model):
        """Test that forward pass produces correct shape."""
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 3, 64, 64)
    
    def test_invalid_input(self, model):
        """Test that invalid input raises error."""
        with pytest.raises(ValueError):
            model(torch.randn(2, 5, 64, 64))  # Wrong channels
```

---

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/my-feature
   ```

2. **Make your changes** and commit:
   ```bash
   git add -A
   git commit -m "feat: add new feature X"
   ```
   
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation
   - `test:` Tests
   - `refactor:` Code refactoring

3. **Run tests** before pushing:
   ```bash
   pytest tests/ -v
   black vdm/ bind/ --check
   ```

4. **Push and create PR**:
   ```bash
   git push origin feature/my-feature
   ```
   Then open a Pull Request on GitHub.

5. **PR Checklist**:
   - [ ] Tests pass
   - [ ] Code is formatted
   - [ ] Documentation updated
   - [ ] CHANGELOG updated (if applicable)

---

## Adding New Models

To add a new generative model type:

### 1. Create Model File

```python
# vdm/my_model.py
from lightning.pytorch import LightningModule
from vdm.networks_clean import UNetVDM

class LightMyModel(LightningModule):
    """My new generative model."""
    
    def __init__(self, score_model, learning_rate=1e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['score_model'])
        self.score_model = score_model
        self.learning_rate = learning_rate
    
    def training_step(self, batch, batch_idx):
        # Implement training logic
        ...
    
    def draw_samples(self, conditioning, batch_size=1, n_steps=50):
        # Implement sampling logic
        ...
```

### 2. Add Config File

```ini
# configs/my_model.ini
[TRAINING]
seed = 8
dataset = IllustrisTNG
data_root = /path/to/data

# Model-specific parameters
my_param = value
```

### 3. Register in train_unified.py

```python
# In train_unified.py
MODEL_TYPES = ['vdm', 'triple', 'ddpm', 'dsm', 'interpolant', 
               'ot_flow', 'consistency', 'my_model']  # Add here

DEFAULT_CONFIGS = {
    ...
    'my_model': 'configs/my_model.ini',
}
```

### 4. Add Tests

```python
# tests/test_my_model.py
def test_my_model_forward():
    ...

def test_my_model_sampling():
    ...
```

### 5. Update Documentation

- Add to `vdm/MODEL_COMPARISON.md`
- Update main `README.md`
- Add docstrings

---

## Documentation

### Building Docs (Future)

```bash
# Install docs dependencies
pip install sphinx sphinx-rtd-theme

# Build docs
cd docs
make html
```

### Documentation Standards

- All public functions need docstrings
- Include usage examples
- Keep README.md up to date
- Update CHANGELOG.md for releases

---

## Questions?

- Open an issue for bugs or feature requests
- Reach out to maintainers for guidance
- Check existing issues before creating new ones

Thank you for contributing! 🌟
