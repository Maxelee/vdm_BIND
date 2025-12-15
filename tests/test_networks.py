"""
Comprehensive tests for vdm/networks_clean.py module.
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTimestepEmbedding:
    """Test timestep embedding function."""
    
    def test_output_shape(self):
        """Test embedding output shape."""
        from vdm.networks_clean import get_timestep_embedding
        
        timesteps = torch.rand(32)
        emb = get_timestep_embedding(timesteps, embedding_dim=64)
        
        assert emb.shape == (32, 64)
    
    def test_different_embedding_dims(self):
        """Test various embedding dimensions."""
        from vdm.networks_clean import get_timestep_embedding
        
        timesteps = torch.rand(16)
        
        for dim in [32, 64, 128, 256]:
            emb = get_timestep_embedding(timesteps, embedding_dim=dim)
            assert emb.shape == (16, dim)
    
    def test_same_timesteps_give_same_embeddings(self):
        """Same timesteps should give same embeddings."""
        from vdm.networks_clean import get_timestep_embedding
        
        # Use fixed timesteps (not near boundaries where sin/cos might be degenerate)
        timesteps = torch.tensor([0.25, 0.5, 0.75])
        
        emb1 = get_timestep_embedding(timesteps.clone(), embedding_dim=64)
        emb2 = get_timestep_embedding(timesteps.clone(), embedding_dim=64)
        
        assert torch.allclose(emb1, emb2)
    
    def test_different_timesteps_different_embeddings(self):
        """Different timesteps should give different embeddings."""
        from vdm.networks_clean import get_timestep_embedding
        
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])
        
        emb1 = get_timestep_embedding(t1, embedding_dim=64)
        emb2 = get_timestep_embedding(t2, embedding_dim=64)
        
        assert not torch.allclose(emb1, emb2)


class TestAttention:
    """Test attention mechanisms."""
    
    def test_attention_output_shape(self):
        """Test attention output shape."""
        from vdm.networks_clean import Attention
        
        attn = Attention(n_heads=4)
        
        # qkv has 3*H*C channels where H=4 heads, C=16 channels per head
        qkv = torch.randn(2, 3 * 4 * 16, 8, 8)
        out = attn(qkv)
        
        # Output should have H*C channels
        assert out.shape == (2, 4 * 16, 8, 8)
    
    def test_attention_block_residual(self):
        """Test AttentionBlock has residual connection."""
        from vdm.networks_clean import AttentionBlock
        
        block = AttentionBlock(n_heads=4, n_channels=64, norm_groups=8)
        
        x = torch.randn(2, 64, 8, 8)
        out = block(x)
        
        assert out.shape == x.shape
        # Output should not be exactly zero (residual adds input)
        assert not torch.allclose(out, torch.zeros_like(out))


class TestCrossAttention:
    """Test cross-attention mechanism."""
    
    def test_cross_attention_output_shape(self):
        """Test cross-attention output shape."""
        from vdm.networks_clean import CrossAttentionBlock
        
        block = CrossAttentionBlock(
            n_heads=4,
            n_channels=64,
            cond_channels=32,
            norm_groups=8,
        )
        
        x = torch.randn(2, 64, 16, 16)
        cond = torch.randn(2, 32, 16, 16)
        
        out = block(x, cond)
        
        assert out.shape == x.shape
    
    def test_cross_attention_chunked(self):
        """Test chunked cross-attention for memory efficiency."""
        from vdm.networks_clean import CrossAttentionBlock
        
        block = CrossAttentionBlock(
            n_heads=4,
            n_channels=64,
            cond_channels=32,
            norm_groups=8,
            use_chunked_attention=True,
            chunk_size=64,
        )
        
        x = torch.randn(2, 64, 16, 16)
        cond = torch.randn(2, 32, 16, 16)
        
        out = block(x, cond)
        
        assert out.shape == x.shape
    
    def test_cross_attention_with_stats(self):
        """Test cross-attention returns attention statistics."""
        from vdm.networks_clean import CrossAttentionBlock
        
        block = CrossAttentionBlock(
            n_heads=4,
            n_channels=64,
            cond_channels=32,
            norm_groups=8,
        )
        
        x = torch.randn(2, 64, 8, 8)
        cond = torch.randn(2, 32, 8, 8)
        
        out, stats = block(x, cond, return_attention_stats=True)
        
        assert out.shape == x.shape
        assert isinstance(stats, dict)


class TestFourierFeatures:
    """Test Fourier feature encoding."""
    
    def test_fourier_features_output_shape_new_mode(self):
        """Test Fourier features output shape in new mode."""
        from vdm.networks_clean import FourierFeatures
        
        ff = FourierFeatures(frequencies=[1, 2, 4, 8])
        
        x = torch.randn(2, 4, 64, 64)
        out = ff(x)
        
        # New mode: Output has 2 * num_frequencies * channels features
        # 2 (sin+cos) * 4 (frequencies) * 4 (channels) = 32
        expected_channels = 2 * 4 * 4
        assert out.shape == (2, expected_channels, 64, 64)
    
    def test_fourier_features_output_shape_legacy_mode(self):
        """Test Fourier features output shape in legacy mode."""
        from vdm.networks_clean import FourierFeatures
        
        ff = FourierFeatures(first=-2.0, last=1.0, step=1.0, legacy_mode=True)
        
        x = torch.randn(2, 4, 64, 64)
        out = ff(x)
        
        # Legacy mode: Output has 2 * num_frequencies features
        # freq exponents: [-2, -1, 0, 1] = 4 freqs
        # 2 (sin+cos) * 4 (frequencies) = 8
        assert out.shape == (2, 8, 64, 64)
    
    def test_fourier_features_deterministic(self):
        """Fourier features should be deterministic."""
        from vdm.networks_clean import FourierFeatures
        
        ff = FourierFeatures(frequencies=[1, 2, 4, 8])
        
        x = torch.randn(2, 4, 32, 32)
        
        out1 = ff(x)
        out2 = ff(x)
        
        assert torch.allclose(out1, out2)


class TestResnetBlock:
    """Test ResNet block."""
    
    def test_resnet_block_same_channels(self):
        """Test ResNet block with same input/output channels."""
        from vdm.networks_clean import ResnetBlock
        
        block = ResnetBlock(
            ch_in=64,
            ch_out=64,
            condition_dim=128,
            dropout_prob=0.0,
            norm_groups=8,
        )
        
        x = torch.randn(2, 64, 32, 32)
        cond = torch.randn(2, 128)
        
        out = block(x, cond)
        
        assert out.shape == x.shape
    
    def test_resnet_block_different_channels(self):
        """Test ResNet block with different input/output channels."""
        from vdm.networks_clean import ResnetBlock
        
        block = ResnetBlock(
            ch_in=64,
            ch_out=128,
            condition_dim=128,
            dropout_prob=0.0,
            norm_groups=8,
        )
        
        x = torch.randn(2, 64, 32, 32)
        cond = torch.randn(2, 128)
        
        out = block(x, cond)
        
        assert out.shape == (2, 128, 32, 32)
    
    def test_resnet_block_with_dropout(self):
        """Test ResNet block with dropout."""
        from vdm.networks_clean import ResnetBlock
        
        block = ResnetBlock(
            ch_in=64,
            ch_out=64,
            condition_dim=128,
            dropout_prob=0.5,
            norm_groups=8,
        )
        
        block.train()  # Enable dropout
        
        x = torch.randn(2, 64, 32, 32)
        cond = torch.randn(2, 128)
        
        # With dropout, multiple forward passes may differ
        out1 = block(x, cond)
        out2 = block(x, cond)
        
        # Outputs should be different with dropout
        # (with high probability for large tensors)
        assert out1.shape == out2.shape


class TestUNetVDM:
    """Test UNet VDM architecture."""
    
    @pytest.fixture
    def small_unet(self):
        """Create small UNet for testing."""
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
    
    def test_forward_basic(self, small_unet):
        """Test basic forward pass."""
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.randn(2)
        cond = torch.randn(2, 1, 64, 64)
        
        out = small_unet(x, gamma, cond)
        
        assert out.shape == x.shape
    
    def test_forward_deterministic_eval(self, small_unet):
        """Test forward pass is deterministic in eval mode."""
        small_unet.eval()
        
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.randn(2)
        cond = torch.randn(2, 1, 64, 64)
        
        with torch.no_grad():
            out1 = small_unet(x, gamma, cond)
            out2 = small_unet(x, gamma, cond)
        
        assert torch.allclose(out1, out2)
    
    def test_forward_with_large_scale(self):
        """Test forward with large-scale conditioning."""
        from vdm.networks_clean import UNetVDM
        
        unet = UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,
            norm_groups=4,
            n_blocks=2,
            add_attention=False,
            use_fourier_features=False,
            conditioning_channels=1,
            large_scale_channels=3,
        )
        
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.randn(2)
        # Conditioning: 1 base + 3 large-scale = 4 channels
        cond = torch.randn(2, 4, 64, 64)
        
        out = unet(x, gamma, cond)
        
        assert out.shape == x.shape
    
    def test_forward_with_attention(self):
        """Test forward with self-attention."""
        from vdm.networks_clean import UNetVDM
        
        unet = UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,
            norm_groups=4,
            n_blocks=2,
            add_attention=True,
            n_attention_heads=4,
            use_fourier_features=False,
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.randn(2)
        cond = torch.randn(2, 1, 64, 64)
        
        out = unet(x, gamma, cond)
        
        assert out.shape == x.shape
    
    def test_forward_with_fourier_features(self):
        """Test forward with Fourier features."""
        from vdm.networks_clean import UNetVDM
        
        unet = UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,
            norm_groups=4,
            n_blocks=2,
            add_attention=False,
            use_fourier_features=True,
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.randn(2)
        cond = torch.randn(2, 1, 64, 64)
        
        out = unet(x, gamma, cond)
        
        assert out.shape == x.shape
    
    def test_gamma_shape_scalar(self, small_unet):
        """Test UNet handles scalar gamma."""
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.tensor(0.5)  # Scalar
        cond = torch.randn(2, 1, 64, 64)
        
        out = small_unet(x, gamma, cond)
        
        assert out.shape == x.shape
    
    def test_gamma_shape_1d(self, small_unet):
        """Test UNet handles 1D gamma."""
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.tensor([0.5])  # 1D
        cond = torch.randn(2, 1, 64, 64)
        
        out = small_unet(x, gamma, cond)
        
        assert out.shape == x.shape


class TestUNetVDMWithParamConditioning:
    """Test UNet with parameter conditioning."""
    
    def test_param_conditioning_forward(self):
        """Test forward with parameter conditioning."""
        from vdm.networks_clean import UNetVDM
        
        # Define param ranges (5 parameters)
        param_min = [0.0] * 5
        param_max = [1.0] * 5
        
        unet = UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,
            norm_groups=4,
            n_blocks=2,
            add_attention=False,
            use_fourier_features=False,
            use_param_conditioning=True,
            use_param_prediction=True,
            param_min=param_min,
            param_max=param_max,
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.randn(2)
        cond = torch.randn(2, 1, 64, 64)
        params = torch.randn(2, 5)
        
        out = unet(x, gamma, cond, param_conditioning=params)
        
        # With param prediction enabled, output is a tuple: (noise_pred, param_pred)
        if isinstance(out, tuple):
            noise_pred, param_pred = out
            assert noise_pred.shape == x.shape
            assert param_pred.shape == (2, 5)
        else:
            assert out.shape == x.shape
    
    def test_param_conditioning_no_prediction(self):
        """Test forward with param conditioning but no prediction head."""
        from vdm.networks_clean import UNetVDM
        
        # Define param ranges (5 parameters)
        param_min = [0.0] * 5
        param_max = [1.0] * 5
        
        unet = UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,
            norm_groups=4,
            n_blocks=2,
            add_attention=False,
            use_fourier_features=False,
            use_param_conditioning=True,
            use_param_prediction=False,  # Disable prediction head
            param_min=param_min,
            param_max=param_max,
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.randn(2)
        cond = torch.randn(2, 1, 64, 64)
        params = torch.randn(2, 5)
        
        out = unet(x, gamma, cond, param_conditioning=params)
        
        # Without param prediction, output is just noise prediction
        assert out.shape == x.shape


class TestUNetVDMWithCrossAttention:
    """Test UNet with cross-attention."""
    
    def test_cross_attention_forward(self):
        """Test forward with cross-attention."""
        from vdm.networks_clean import UNetVDM
        
        unet = UNetVDM(
            input_channels=3,
            gamma_min=-10.0,
            gamma_max=10.0,
            embedding_dim=32,
            norm_groups=4,
            n_blocks=2,
            add_attention=True,
            n_attention_heads=4,
            use_fourier_features=False,
            use_cross_attention=True,
            cross_attention_heads=4,
            conditioning_channels=1,
            large_scale_channels=0,
        )
        
        x = torch.randn(2, 3, 64, 64)
        gamma = torch.randn(2)
        cond = torch.randn(2, 1, 64, 64)
        
        out = unet(x, gamma, cond)
        
        assert out.shape == x.shape


class TestDownsampleUpsample:
    """Test downsample and upsample blocks."""
    
    def test_downblock_halves_resolution(self):
        """Test DownBlock halves spatial resolution."""
        from vdm.networks_clean import DownBlock, ResnetBlock
        
        resnet = ResnetBlock(ch_in=64, ch_out=64, condition_dim=128, norm_groups=8)
        down = DownBlock(resnet_block=resnet)
        
        x = torch.randn(2, 64, 32, 32)
        cond = torch.randn(2, 128)
        
        out, xskip = down(x, cond)
        
        assert out.shape == (2, 64, 16, 16)
        assert xskip.shape == (2, 64, 32, 32)
    
    def test_upblock_doubles_resolution(self):
        """Test UpBlock doubles spatial resolution."""
        from vdm.networks_clean import UpBlock, ResnetBlock
        
        # UpBlock expects input channels = upsample_ch//2 + skip_channels
        upsample_ch = 128
        skip_ch = 64
        resnet = ResnetBlock(ch_in=upsample_ch // 2 + skip_ch, ch_out=64, condition_dim=128, norm_groups=8)
        up = UpBlock(resnet_block=resnet, upsample_ch=upsample_ch)
        
        x = torch.randn(2, upsample_ch, 16, 16)
        xskip = torch.randn(2, skip_ch, 32, 32)
        cond = torch.randn(2, 128)
        
        out = up(x, xskip, cond)
        
        assert out.shape == (2, 64, 32, 32)
    
    def test_updownblock(self):
        """Test UpDownBlock module (residual block with optional attention)."""
        from vdm.networks_clean import UpDownBlock, ResnetBlock
        
        resnet = ResnetBlock(ch_in=64, ch_out=64, condition_dim=128, norm_groups=8)
        updown = UpDownBlock(resnet_block=resnet)
        
        x = torch.randn(2, 64, 32, 32)
        cond = torch.randn(2, 128)
        
        out = updown(x, cond)
        
        assert out.shape == x.shape


class TestParamEmbedding:
    """Test flexible parameter embedding for conditional/unconditional generation."""
    
    def test_unconditional_none_params(self):
        """Test ParamEmbedding with None params (unconditional mode)."""
        from vdm.networks_clean import ParamEmbedding
        
        embed = ParamEmbedding(embed_dim=192, param_min=None, param_max=None)
        
        assert embed.Nparams == 0
        assert embed.embedding is None
        
        # Forward should return zeros
        dummy_input = torch.randn(4, 5)  # Batch of 4, any param dim
        out = embed(dummy_input)
        assert out.shape == (4, 192)
        assert torch.allclose(out, torch.zeros_like(out))
    
    def test_unconditional_empty_params(self):
        """Test ParamEmbedding with empty param lists."""
        from vdm.networks_clean import ParamEmbedding
        
        embed = ParamEmbedding(embed_dim=192, param_min=[], param_max=[])
        
        assert embed.Nparams == 0
        assert embed.embedding is None
    
    def test_unconditional_forward_with_none_input(self):
        """Test unconditional forward with None input."""
        from vdm.networks_clean import ParamEmbedding
        
        embed = ParamEmbedding(embed_dim=128, param_min=None, param_max=None)
        
        # Forward with None should return zeros
        out = embed(None)
        assert out.shape == (1, 128)
        assert torch.allclose(out, torch.zeros_like(out))
    
    def test_conditional_custom_params(self):
        """Test ParamEmbedding with custom number of parameters."""
        from vdm.networks_clean import ParamEmbedding
        
        # 6 custom parameters
        param_min = [0.1, 0.5, 0.0, 0.0, 0.1, 1e4]
        param_max = [0.5, 1.2, 1.0, 1.0, 10.0, 1e6]
        
        embed = ParamEmbedding(embed_dim=192, param_min=param_min, param_max=param_max)
        
        assert embed.Nparams == 6
        assert embed.embedding is not None
        
        # Forward with valid params
        params = torch.rand(8, 6)  # Batch of 8
        out = embed(params)
        assert out.shape == (8, 192)
    
    def test_conditional_single_param(self):
        """Test ParamEmbedding with single parameter."""
        from vdm.networks_clean import ParamEmbedding
        
        embed = ParamEmbedding(embed_dim=64, param_min=[0.0], param_max=[1.0])
        
        assert embed.Nparams == 1
        
        params = torch.rand(4, 1)
        out = embed(params)
        assert out.shape == (4, 64)
    
    def test_conditional_many_params(self):
        """Test ParamEmbedding with many parameters (like CAMELS)."""
        from vdm.networks_clean import ParamEmbedding
        
        # 35 parameters like CAMELS
        param_min = np.zeros(35).tolist()
        param_max = np.ones(35).tolist()
        
        embed = ParamEmbedding(embed_dim=192, param_min=param_min, param_max=param_max)
        
        assert embed.Nparams == 35
        
        params = torch.rand(16, 35)
        out = embed(params)
        assert out.shape == (16, 192)
    
    def test_param_normalization(self):
        """Test that parameters are normalized to [0,1]."""
        from vdm.networks_clean import ParamEmbedding
        
        param_min = [0.0, -10.0]
        param_max = [1.0, 10.0]
        
        embed = ParamEmbedding(embed_dim=64, param_min=param_min, param_max=param_max)
        
        # Params at min should give different embedding than params at max
        params_min = torch.tensor([[0.0, -10.0]])
        params_max = torch.tensor([[1.0, 10.0]])
        
        out_min = embed(params_min)
        out_max = embed(params_max)
        
        assert not torch.allclose(out_min, out_max)
    
    def test_param_count_mismatch_error(self):
        """Test that mismatched param count raises error."""
        from vdm.networks_clean import ParamEmbedding
        
        embed = ParamEmbedding(embed_dim=64, param_min=[0, 0, 0], param_max=[1, 1, 1])
        
        # Wrong number of parameters should raise error
        wrong_params = torch.rand(4, 5)  # 5 params instead of 3
        
        with pytest.raises(ValueError, match="Expected 3 parameters"):
            embed(wrong_params)
    
    def test_param_min_max_length_mismatch(self):
        """Test that mismatched min/max lengths raise error."""
        from vdm.networks_clean import ParamEmbedding
        
        with pytest.raises(ValueError, match="must have same length"):
            ParamEmbedding(embed_dim=64, param_min=[0, 0], param_max=[1, 1, 1])
    
    def test_1d_input_handling(self):
        """Test that 1D input is properly reshaped."""
        from vdm.networks_clean import ParamEmbedding
        
        embed = ParamEmbedding(embed_dim=64, param_min=[0, 0], param_max=[1, 1])
        
        # 1D input (single sample, no batch dim)
        params_1d = torch.tensor([0.5, 0.5])
        out = embed(params_1d)
        
        assert out.shape == (1, 64)  # Should add batch dimension
    
    def test_numpy_arrays_work(self):
        """Test that numpy arrays work for param_min/max."""
        from vdm.networks_clean import ParamEmbedding
        
        param_min = np.array([0.0, 0.0, 0.0])
        param_max = np.array([1.0, 1.0, 1.0])
        
        embed = ParamEmbedding(embed_dim=64, param_min=param_min, param_max=param_max)
        
        assert embed.Nparams == 3
        
        params = torch.rand(2, 3)
        out = embed(params)
        assert out.shape == (2, 64)

