"""
Tests for Method Abstraction Layer.

Tests the MethodRegistry and concrete method implementations (VDM, Flow, Consistency).
"""

import pytest
import torch
import numpy as np

from vdm.methods import (
    BaseMethod,
    MethodRegistry,
    VDMMethod,
    FlowMatchingMethod,
    ConsistencyMethod,
    create_method,
    list_methods,
)
from vdm.backbones import create_backbone, BackboneRegistry


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def small_img_size():
    """Small image size for fast tests."""
    return 32


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def unet_backbone(small_img_size):
    """Create a small UNet backbone for testing."""
    return create_backbone(
        "unet",
        img_size=small_img_size,
        input_channels=3,
        output_channels=3,
        conditioning_channels=1,
        large_scale_channels=3,
        param_dim=0,
        embedding_dim=32,
        n_blocks=2,
    )


@pytest.fixture
def unet_backbone_with_params(small_img_size):
    """Create a UNet backbone with parameter conditioning."""
    return create_backbone(
        "unet",
        img_size=small_img_size,
        input_channels=3,
        output_channels=3,
        conditioning_channels=1,
        large_scale_channels=3,
        param_dim=6,
        embedding_dim=32,
        n_blocks=2,
        param_min=[0.0] * 6,
        param_max=[1.0] * 6,
    )


@pytest.fixture
def sample_batch(batch_size, small_img_size):
    """Create a sample batch in AstroDataset format."""
    m_dm = torch.randn(batch_size, 1, small_img_size, small_img_size)
    large_scale = torch.randn(batch_size, 3, small_img_size, small_img_size)
    m_target = torch.randn(batch_size, 3, small_img_size, small_img_size)
    conditions = torch.rand(batch_size, 6)
    return (m_dm, large_scale, m_target, conditions)


# =============================================================================
# Test Method Registry
# =============================================================================

class TestMethodRegistry:
    """Tests for MethodRegistry."""
    
    def test_available_methods(self):
        """Test listing available methods."""
        methods = list_methods()
        assert isinstance(methods, list)
        assert len(methods) >= 3  # At least vdm, flow, consistency
        assert "vdm" in methods
        assert "flow" in methods
        assert "consistency" in methods
    
    def test_get_class(self):
        """Test getting method class."""
        vdm_cls = MethodRegistry.get_class("vdm")
        assert vdm_cls == VDMMethod
        
        flow_cls = MethodRegistry.get_class("flow")
        assert flow_cls == FlowMatchingMethod
    
    def test_unknown_method(self):
        """Test error on unknown method."""
        with pytest.raises(KeyError):
            MethodRegistry.get_class("unknown_method")
    
    def test_create_with_backbone(self, unet_backbone, small_img_size):
        """Test creating method with pre-made backbone."""
        method = MethodRegistry.create(
            "vdm",
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        assert isinstance(method, VDMMethod)
        assert method.backbone is unet_backbone
    
    def test_create_with_backbone_type(self, small_img_size):
        """Test creating method with backbone_type."""
        method = MethodRegistry.create(
            "flow",
            backbone_type="unet",
            img_size=small_img_size,
            image_shape=(3, small_img_size, small_img_size),
        )
        assert isinstance(method, FlowMatchingMethod)
        assert method.backbone is not None
    
    def test_create_requires_backbone(self):
        """Test that create requires backbone or backbone_type."""
        with pytest.raises(ValueError, match="Either 'backbone' or 'backbone_type' must be provided"):
            MethodRegistry.create("vdm")
    
    def test_early_stopping_metric(self):
        """Test getting early stopping metric."""
        assert MethodRegistry.get_early_stopping_metric("vdm") == "val/elbo"
        assert MethodRegistry.get_early_stopping_metric("flow") == "val/loss"
        assert MethodRegistry.get_early_stopping_metric("consistency") == "val/loss"


# =============================================================================
# Test VDM Method
# =============================================================================

class TestVDMMethod:
    """Tests for VDM Method."""
    
    def test_creation(self, unet_backbone, small_img_size):
        """Test VDM method creation."""
        method = VDMMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        assert isinstance(method, BaseMethod)
        assert method.method_name == "vdm"
    
    def test_compute_loss(self, unet_backbone, sample_batch, small_img_size):
        """Test loss computation."""
        method = VDMMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        
        x, conditioning, params = method._unpack_batch(sample_batch)
        loss, metrics = method.compute_loss(x, conditioning, None)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert "elbo" in metrics
        assert "diffusion_loss" in metrics
    
    def test_training_step(self, unet_backbone, sample_batch, small_img_size):
        """Test training step."""
        method = VDMMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        
        loss = method.training_step(sample_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_sample(self, unet_backbone, batch_size, small_img_size):
        """Test sampling."""
        method = VDMMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
            n_sampling_steps=5,  # Few steps for fast test
        )
        
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        samples = method.sample(
            conditioning=conditioning,
            n_samples=batch_size,
            n_steps=3,
            device='cpu',
        )
        
        assert samples.shape == (batch_size, 3, small_img_size, small_img_size)
    
    def test_sample_with_trajectory(self, unet_backbone, batch_size, small_img_size):
        """Test sampling with trajectory."""
        method = VDMMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        samples, trajectory = method.sample(
            conditioning=conditioning,
            n_samples=batch_size,
            n_steps=3,
            return_trajectory=True,
            device='cpu',
        )
        
        assert samples.shape == (batch_size, 3, small_img_size, small_img_size)
        assert len(trajectory) == 4  # Initial + 3 steps
    
    def test_bind_interface(self, unet_backbone, batch_size, small_img_size):
        """Test BIND-compatible draw_samples interface."""
        method = VDMMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        samples = method.draw_samples(
            conditioning=conditioning,
            batch_size=batch_size,
            n_sampling_steps=3,
        )
        
        assert samples.shape == (batch_size, 3, small_img_size, small_img_size)


# =============================================================================
# Test Flow Matching Method
# =============================================================================

class TestFlowMatchingMethod:
    """Tests for Flow Matching Method."""
    
    def test_creation(self, unet_backbone, small_img_size):
        """Test Flow method creation."""
        method = FlowMatchingMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        assert isinstance(method, BaseMethod)
        assert method.method_name == "flow"
    
    def test_compute_loss(self, unet_backbone, sample_batch, small_img_size):
        """Test loss computation."""
        method = FlowMatchingMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        
        x, conditioning, params = method._unpack_batch(sample_batch)
        loss, metrics = method.compute_loss(x, conditioning, None)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in metrics
        assert "mse" in metrics
    
    @pytest.mark.parametrize("x0_mode", ["zeros", "noise"])
    def test_x0_modes(self, unet_backbone, sample_batch, small_img_size, x0_mode):
        """Test different x0 initialization modes."""
        method = FlowMatchingMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
            x0_mode=x0_mode,
        )
        
        x, conditioning, params = method._unpack_batch(sample_batch)
        loss, metrics = method.compute_loss(x, conditioning, None)
        
        assert isinstance(loss, torch.Tensor)
    
    def test_stochastic_interpolant(self, unet_backbone, sample_batch, small_img_size):
        """Test stochastic interpolant mode."""
        method = FlowMatchingMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
            use_stochastic_interpolant=True,
            sigma=0.1,
        )
        
        x, conditioning, params = method._unpack_batch(sample_batch)
        loss, metrics = method.compute_loss(x, conditioning, None)
        
        assert isinstance(loss, torch.Tensor)
    
    def test_sample(self, unet_backbone, batch_size, small_img_size):
        """Test sampling."""
        method = FlowMatchingMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
            n_sampling_steps=5,
        )
        
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        samples = method.sample(
            conditioning=conditioning,
            n_samples=batch_size,
            n_steps=3,
            device='cpu',
        )
        
        assert samples.shape == (batch_size, 3, small_img_size, small_img_size)
    
    def test_stochastic_sampling(self, unet_backbone, batch_size, small_img_size):
        """Test SDE sampling."""
        method = FlowMatchingMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        samples = method.sample(
            conditioning=conditioning,
            n_samples=batch_size,
            n_steps=3,
            stochastic=True,
            device='cpu',
        )
        
        assert samples.shape == (batch_size, 3, small_img_size, small_img_size)


# =============================================================================
# Test Consistency Method
# =============================================================================

class TestConsistencyMethod:
    """Tests for Consistency Method."""
    
    def test_creation(self, unet_backbone, small_img_size):
        """Test Consistency method creation."""
        method = ConsistencyMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
        )
        assert isinstance(method, BaseMethod)
        assert method.method_name == "consistency"
    
    def test_compute_loss_denoising(self, unet_backbone, sample_batch, small_img_size):
        """Test loss computation in denoising phase."""
        method = ConsistencyMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
            use_denoising_pretraining=True,
        )
        
        x, conditioning, params = method._unpack_batch(sample_batch)
        loss, metrics = method.compute_loss(x, conditioning, None)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "denoising_loss" in metrics
    
    def test_compute_loss_consistency(self, unet_backbone, sample_batch, small_img_size):
        """Test loss computation in consistency phase."""
        method = ConsistencyMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
            use_denoising_pretraining=False,
            ct_n_steps=5,
        )
        
        x, conditioning, params = method._unpack_batch(sample_batch)
        loss, metrics = method.compute_loss(x, conditioning, None)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "consistency_loss" in metrics
    
    def test_single_step_sample(self, unet_backbone, batch_size, small_img_size):
        """Test single-step sampling (key feature of consistency models)."""
        method = ConsistencyMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
            n_sampling_steps=1,  # Single step!
        )
        
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        samples = method.sample(
            conditioning=conditioning,
            n_samples=batch_size,
            n_steps=1,
            device='cpu',
        )
        
        assert samples.shape == (batch_size, 3, small_img_size, small_img_size)
    
    def test_multi_step_sample(self, unet_backbone, batch_size, small_img_size):
        """Test multi-step sampling."""
        method = ConsistencyMethod(
            backbone=unet_backbone,
            image_shape=(3, small_img_size, small_img_size),
            ct_n_steps=5,
        )
        
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        samples = method.sample(
            conditioning=conditioning,
            n_samples=batch_size,
            n_steps=3,
            device='cpu',
        )
        
        assert samples.shape == (batch_size, 3, small_img_size, small_img_size)


# =============================================================================
# Test Cross-Backbone Compatibility
# =============================================================================

class TestCrossBackboneCompatibility:
    """Test that methods work with different backbones."""
    
    @pytest.mark.parametrize("method_type", ["vdm", "flow", "consistency"])
    def test_method_with_unet(self, method_type, small_img_size, batch_size):
        """Test each method with UNet backbone."""
        method = create_method(
            method_type,
            backbone_type="unet",
            img_size=small_img_size,
            image_shape=(3, small_img_size, small_img_size),
            n_sampling_steps=3,
            ct_n_steps=5,  # For consistency
        )
        
        # Test that it can compute loss
        x = torch.randn(batch_size, 3, small_img_size, small_img_size)
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        
        loss, metrics = method.compute_loss(x, conditioning, None)
        assert isinstance(loss, torch.Tensor)
    
    def test_method_with_param_conditioning(self, unet_backbone_with_params, small_img_size, batch_size):
        """Test method with parameter conditioning."""
        method = VDMMethod(
            backbone=unet_backbone_with_params,
            image_shape=(3, small_img_size, small_img_size),
            use_param_conditioning=True,
        )
        
        x = torch.randn(batch_size, 3, small_img_size, small_img_size)
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        params = torch.rand(batch_size, 6)
        
        loss, metrics = method.compute_loss(x, conditioning, params)
        assert isinstance(loss, torch.Tensor)


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_method(self, small_img_size):
        """Test create_method helper."""
        method = create_method(
            "vdm",
            backbone_type="unet",
            img_size=small_img_size,
            image_shape=(3, small_img_size, small_img_size),
        )
        assert isinstance(method, VDMMethod)
    
    def test_list_methods(self):
        """Test list_methods helper."""
        methods = list_methods()
        assert "vdm" in methods
        assert "flow" in methods
        assert "consistency" in methods


# =============================================================================
# Test Gradients
# =============================================================================

class TestGradients:
    """Test that gradients flow correctly."""
    
    @pytest.mark.parametrize("method_type", ["vdm", "flow", "consistency"])
    def test_gradients_flow(self, method_type, small_img_size, batch_size):
        """Test that gradients flow through methods."""
        method = create_method(
            method_type,
            backbone_type="unet",
            img_size=small_img_size,
            image_shape=(3, small_img_size, small_img_size),
            n_sampling_steps=3,
            ct_n_steps=5,
            use_denoising_pretraining=False,  # For consistency
        )
        
        x = torch.randn(batch_size, 3, small_img_size, small_img_size, requires_grad=True)
        conditioning = torch.randn(batch_size, 4, small_img_size, small_img_size)
        
        loss, _ = method.compute_loss(x, conditioning, None)
        loss.backward()
        
        # Check backbone has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in method.backbone.parameters())
        assert has_grad, f"No gradients in backbone for {method_type}"
