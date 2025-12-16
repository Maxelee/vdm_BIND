"""
Tests for backbone abstraction layer.

Tests the unified backbone interface and registry system.
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vdm.backbones import (
    BackboneBase,
    BackboneRegistry,
    UNetBackbone,
    DiTBackbone,
    FNOBackbone,
    create_backbone,
    list_backbones,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def img_size():
    return 32  # Small for fast tests

@pytest.fixture
def input_channels():
    return 3

@pytest.fixture
def conditioning_channels():
    return 1

@pytest.fixture
def large_scale_channels():
    return 3

@pytest.fixture
def param_dim():
    return 6


# =============================================================================
# Registry Tests
# =============================================================================

class TestBackboneRegistry:
    """Tests for backbone registry."""
    
    def test_available_backbones(self):
        """Test listing available backbones."""
        available = list_backbones()
        assert "unet" in available
        assert "dit" in available
        assert "fno" in available
        # Check presets
        assert "dit-s" in available
        assert "fno-b" in available
        assert "unet-b" in available
    
    def test_get_class(self):
        """Test getting backbone class."""
        cls = BackboneRegistry.get_class("unet")
        assert cls == UNetBackbone
        
        cls = BackboneRegistry.get_class("dit")
        assert cls == DiTBackbone
        
        cls = BackboneRegistry.get_class("fno")
        assert cls == FNOBackbone
    
    def test_unknown_backbone(self):
        """Test error for unknown backbone."""
        with pytest.raises(KeyError):
            create_backbone("unknown_backbone")
    
    def test_register_custom_backbone(self):
        """Test registering a custom backbone."""
        @BackboneRegistry.register("test_custom")
        class TestBackbone(BackboneBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.linear = torch.nn.Linear(10, 10)
            
            def forward(self, x_t, t, conditioning=None, param_conditioning=None):
                return x_t
        
        assert "test_custom" in list_backbones()
        backbone = create_backbone("test_custom", img_size=32)
        assert isinstance(backbone, TestBackbone)
        
        # Clean up
        del BackboneRegistry._registry["test_custom"]


# =============================================================================
# UNet Backbone Tests
# =============================================================================

class TestUNetBackbone:
    """Tests for UNet backbone wrapper."""
    
    def test_unet_creation(self, img_size):
        """Test UNet backbone creation."""
        backbone = create_backbone("unet", img_size=img_size)
        assert isinstance(backbone, UNetBackbone)
        assert backbone.img_size == img_size
    
    def test_unet_forward(self, batch_size, img_size, input_channels, 
                          conditioning_channels, large_scale_channels):
        """Test UNet forward pass."""
        backbone = create_backbone(
            "unet",
            img_size=img_size,
            input_channels=input_channels,
            conditioning_channels=conditioning_channels,
            large_scale_channels=large_scale_channels,
            embedding_dim=32,  # Small for tests
            n_blocks=2,
        )
        
        x_t = torch.randn(batch_size, input_channels, img_size, img_size)
        t = torch.rand(batch_size)
        conditioning = torch.randn(
            batch_size, 
            conditioning_channels + large_scale_channels, 
            img_size, 
            img_size
        )
        
        output = backbone(x_t, t, conditioning)
        
        assert output.shape == (batch_size, input_channels, img_size, img_size)
    
    def test_unet_with_params(self, batch_size, img_size, param_dim):
        """Test UNet with parameter conditioning."""
        backbone = create_backbone(
            "unet",
            img_size=img_size,
            param_dim=param_dim,
            embedding_dim=32,
            n_blocks=2,
        )
        
        x_t = torch.randn(batch_size, 3, img_size, img_size)
        t = torch.rand(batch_size)
        conditioning = torch.randn(batch_size, 4, img_size, img_size)
        params = torch.randn(batch_size, param_dim)
        
        output = backbone(x_t, t, conditioning, params)
        assert output.shape == (batch_size, 3, img_size, img_size)
    
    def test_unet_gamma_conversion(self, batch_size):
        """Test t to gamma conversion."""
        backbone = create_backbone("unet", img_size=32, gamma_min=-13.3, gamma_max=5.0)
        
        # t=0 should give gamma_max (clean)
        t_clean = torch.zeros(batch_size)
        gamma_clean = backbone.t_to_gamma(t_clean)
        assert torch.allclose(gamma_clean, torch.tensor(5.0))
        
        # t=1 should give gamma_min (noisy)
        t_noisy = torch.ones(batch_size)
        gamma_noisy = backbone.t_to_gamma(t_noisy)
        assert torch.allclose(gamma_noisy, torch.tensor(-13.3))
    
    def test_unet_preset_configs(self, img_size):
        """Test UNet preset configurations."""
        # UNet-S
        unet_s = create_backbone("unet-s", img_size=img_size)
        assert unet_s.net.embedding_dim == 64
        
        # UNet-B
        unet_b = create_backbone("unet-b", img_size=img_size)
        assert unet_b.net.embedding_dim == 128


# =============================================================================
# DiT Backbone Tests
# =============================================================================

class TestDiTBackbone:
    """Tests for DiT backbone wrapper."""
    
    def test_dit_creation(self, img_size):
        """Test DiT backbone creation."""
        backbone = create_backbone("dit", img_size=img_size, patch_size=4)
        assert isinstance(backbone, DiTBackbone)
        assert backbone.img_size == img_size
    
    def test_dit_forward(self, batch_size, img_size, input_channels,
                         conditioning_channels, large_scale_channels):
        """Test DiT forward pass."""
        backbone = create_backbone(
            "dit",
            img_size=img_size,
            input_channels=input_channels,
            conditioning_channels=conditioning_channels,
            large_scale_channels=large_scale_channels,
            hidden_size=64,  # Small for tests
            depth=2,
            num_heads=2,
            patch_size=4,
        )
        
        x_t = torch.randn(batch_size, input_channels, img_size, img_size)
        t = torch.rand(batch_size)
        conditioning = torch.randn(
            batch_size,
            conditioning_channels + large_scale_channels,
            img_size,
            img_size
        )
        
        output = backbone(x_t, t, conditioning)
        assert output.shape == (batch_size, input_channels, img_size, img_size)
    
    def test_dit_with_params(self, batch_size, img_size, param_dim):
        """Test DiT with parameter conditioning."""
        backbone = create_backbone(
            "dit",
            img_size=img_size,
            param_dim=param_dim,
            hidden_size=64,
            depth=2,
            num_heads=2,
            patch_size=4,
        )
        
        x_t = torch.randn(batch_size, 3, img_size, img_size)
        t = torch.rand(batch_size)
        conditioning = torch.randn(batch_size, 4, img_size, img_size)
        params = torch.randn(batch_size, param_dim)
        
        output = backbone(x_t, t, conditioning, params)
        assert output.shape == (batch_size, 3, img_size, img_size)
    
    def test_dit_preset_configs(self, img_size):
        """Test DiT preset configurations."""
        # DiT-S
        dit_s = create_backbone("dit-s", img_size=img_size)
        assert dit_s.net.hidden_size == 384
        assert dit_s.net.depth == 12


# =============================================================================
# FNO Backbone Tests
# =============================================================================

class TestFNOBackbone:
    """Tests for FNO backbone wrapper."""
    
    def test_fno_creation(self, img_size):
        """Test FNO backbone creation."""
        backbone = create_backbone("fno", img_size=img_size)
        assert isinstance(backbone, FNOBackbone)
        assert backbone.img_size == img_size
    
    def test_fno_forward(self, batch_size, img_size, input_channels,
                         conditioning_channels, large_scale_channels):
        """Test FNO forward pass."""
        backbone = create_backbone(
            "fno",
            img_size=img_size,
            input_channels=input_channels,
            conditioning_channels=conditioning_channels,
            large_scale_channels=large_scale_channels,
            hidden_channels=32,  # Small for tests
            n_layers=2,
            modes=8,
        )
        
        x_t = torch.randn(batch_size, input_channels, img_size, img_size)
        t = torch.rand(batch_size)
        conditioning = torch.randn(
            batch_size,
            conditioning_channels + large_scale_channels,
            img_size,
            img_size
        )
        
        output = backbone(x_t, t, conditioning)
        assert output.shape == (batch_size, input_channels, img_size, img_size)
    
    def test_fno_with_params(self, batch_size, img_size, param_dim):
        """Test FNO with parameter conditioning."""
        backbone = create_backbone(
            "fno",
            img_size=img_size,
            param_dim=param_dim,
            hidden_channels=32,
            n_layers=2,
            modes=8,
        )
        
        x_t = torch.randn(batch_size, 3, img_size, img_size)
        t = torch.rand(batch_size)
        conditioning = torch.randn(batch_size, 4, img_size, img_size)
        params = torch.randn(batch_size, param_dim)
        
        output = backbone(x_t, t, conditioning, params)
        assert output.shape == (batch_size, 3, img_size, img_size)
    
    def test_fno_preset_configs(self, img_size):
        """Test FNO preset configurations."""
        # FNO-S
        fno_s = create_backbone("fno-s", img_size=img_size)
        assert fno_s.net.hidden_channels == 32
        
        # FNO-B
        fno_b = create_backbone("fno-b", img_size=img_size)
        assert fno_b.net.hidden_channels == 64


# =============================================================================
# Interface Consistency Tests
# =============================================================================

class TestInterfaceConsistency:
    """Test that all backbones follow the same interface."""
    
    @pytest.mark.parametrize("backbone_type", ["unet", "dit", "fno"])
    def test_standard_interface(self, backbone_type, batch_size, img_size):
        """Test that all backbones accept the same inputs."""
        # Create each backbone with minimal config
        if backbone_type == "unet":
            backbone = create_backbone(
                backbone_type, img_size=img_size, embedding_dim=32, n_blocks=2
            )
        elif backbone_type == "dit":
            backbone = create_backbone(
                backbone_type, img_size=img_size, hidden_size=64, depth=2,
                num_heads=2, patch_size=4
            )
        else:
            backbone = create_backbone(
                backbone_type, img_size=img_size, hidden_channels=32,
                n_layers=2, modes=8
            )
        
        # Standard inputs
        x_t = torch.randn(batch_size, 3, img_size, img_size)
        t = torch.rand(batch_size)
        conditioning = torch.randn(batch_size, 4, img_size, img_size)
        params = torch.randn(batch_size, 6)
        
        # All backbones should accept these inputs
        output = backbone(x_t, t, conditioning)
        assert output.shape == x_t.shape
    
    @pytest.mark.parametrize("backbone_type", ["unet", "dit", "fno"])
    def test_without_conditioning(self, backbone_type, batch_size, img_size):
        """Test backbones work without conditioning."""
        # Create backbone with no conditioning channels
        if backbone_type == "unet":
            backbone = create_backbone(
                backbone_type, img_size=img_size, 
                conditioning_channels=0, large_scale_channels=0,
                embedding_dim=32, n_blocks=2, use_fourier_features=False
            )
        elif backbone_type == "dit":
            backbone = create_backbone(
                backbone_type, img_size=img_size,
                conditioning_channels=0, large_scale_channels=0,
                hidden_size=64, depth=2, num_heads=2, patch_size=4
            )
        else:
            backbone = create_backbone(
                backbone_type, img_size=img_size,
                conditioning_channels=0, large_scale_channels=0,
                hidden_channels=32, n_layers=2, modes=8
            )
        
        x_t = torch.randn(batch_size, 3, img_size, img_size)
        t = torch.rand(batch_size)
        
        output = backbone(x_t, t)
        assert output.shape == x_t.shape
    
    @pytest.mark.parametrize("backbone_type", ["unet", "dit", "fno"])
    def test_get_config(self, backbone_type, img_size):
        """Test configuration retrieval."""
        if backbone_type == "unet":
            backbone = create_backbone(
                backbone_type, img_size=img_size, embedding_dim=32, n_blocks=2
            )
        elif backbone_type == "dit":
            backbone = create_backbone(
                backbone_type, img_size=img_size, hidden_size=64, depth=2,
                num_heads=2, patch_size=4
            )
        else:
            backbone = create_backbone(
                backbone_type, img_size=img_size, hidden_channels=32,
                n_layers=2, modes=8
            )
        
        config = backbone.get_config()
        
        # All configs should have these keys
        assert "input_channels" in config
        assert "output_channels" in config
        assert "conditioning_channels" in config
        assert "large_scale_channels" in config
        assert "param_dim" in config
        assert "img_size" in config
        assert "backbone_type" in config


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestGradientFlow:
    """Test that gradients flow through backbones."""
    
    @pytest.mark.parametrize("backbone_type", ["unet", "dit", "fno"])
    def test_gradients_flow(self, backbone_type, batch_size, img_size):
        """Test that gradients flow through the backbone."""
        if backbone_type == "unet":
            backbone = create_backbone(
                backbone_type, img_size=img_size, embedding_dim=32, n_blocks=2
            )
        elif backbone_type == "dit":
            backbone = create_backbone(
                backbone_type, img_size=img_size, hidden_size=64, depth=2,
                num_heads=2, patch_size=4
            )
        else:
            backbone = create_backbone(
                backbone_type, img_size=img_size, hidden_channels=32,
                n_layers=2, modes=8
            )
        
        x_t = torch.randn(batch_size, 3, img_size, img_size, requires_grad=True)
        t = torch.rand(batch_size)
        conditioning = torch.randn(batch_size, 4, img_size, img_size)
        
        output = backbone(x_t, t, conditioning)
        loss = output.sum()
        loss.backward()
        
        assert x_t.grad is not None
        assert x_t.grad.shape == x_t.shape


# =============================================================================
# Time Convention Tests
# =============================================================================

class TestTimeConvention:
    """Test time convention consistency."""
    
    def test_t_boundaries(self, batch_size, img_size):
        """Test that t=0 and t=1 work correctly."""
        backbone = create_backbone(
            "unet", img_size=img_size, embedding_dim=32, n_blocks=2
        )
        
        x_t = torch.randn(batch_size, 3, img_size, img_size)
        conditioning = torch.randn(batch_size, 4, img_size, img_size)
        
        # t=0 (clean)
        t_clean = torch.zeros(batch_size)
        output_clean = backbone(x_t, t_clean, conditioning)
        assert output_clean.shape == x_t.shape
        assert torch.isfinite(output_clean).all()
        
        # t=1 (noisy)
        t_noisy = torch.ones(batch_size)
        output_noisy = backbone(x_t, t_noisy, conditioning)
        assert output_noisy.shape == x_t.shape
        assert torch.isfinite(output_noisy).all()
    
    def test_t_interpolation(self, batch_size, img_size):
        """Test that intermediate t values work."""
        backbone = create_backbone(
            "fno", img_size=img_size, hidden_channels=32, n_layers=2, modes=8
        )
        
        x_t = torch.randn(batch_size, 3, img_size, img_size)
        conditioning = torch.randn(batch_size, 4, img_size, img_size)
        
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.full((batch_size,), t_val)
            output = backbone(x_t, t, conditioning)
            assert torch.isfinite(output).all(), f"NaN/Inf at t={t_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
