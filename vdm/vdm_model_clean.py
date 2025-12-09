"""
Clean Variational Diffusion Model for Astrophysical Data.

This is a simplified, transparent implementation following the VDM paper (arxiv:2107.00630).

Key features:
- Noise prediction network (predicts noise, not clean data)
- Variance-preserving diffusion process
- Optional parameter prediction auxiliary task
- 3-channel mode: [DM, Gas, Stars] (all in log space, normalized)

Architecture:
- Forward: Add noise to data (variance preserving)
- Training: Predict the noise that was added
- Sampling: Iteratively denoise from pure noise to data

Loss components:
1. Diffusion loss: MSE between predicted and actual noise
2. Latent loss: KL divergence to standard Gaussian prior
3. Reconstruction loss: Negative log-likelihood of data
4. Parameter prediction loss: MSE between predicted and true parameters (optional)
"""

import torch
import numpy as np
from torch import nn, Tensor, autograd
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from torch.special import expm1
from torch.distributions.normal import Normal
from typing import Optional, Tuple, Dict
from tqdm import trange

from lightning.pytorch import LightningModule

from vdm.utils import FixedLinearSchedule, LearnedLinearSchedule, NNSchedule, kl_std_normal


# ===================================================================
# CORE VDM MODEL
# ===================================================================

class CleanVDM(nn.Module):
    """
    Clean Variational Diffusion Model.
    
    Implements the standard VDM formulation with configurable hyperparameters.
    All parameters are explicit - no magic numbers.
    
    Supports:
    - 3-channel mode: [DM, Gas, Stars_log] (default)
    - Focal loss for stellar channel: downweights easy examples, upweights hard examples
    
    Args:
        score_model: Neural network that predicts noise
        noise_schedule: Type of noise schedule ("fixed_linear", "learned_linear", "learned_nn")
        gamma_min: Minimum log SNR (default: -13.3)
        gamma_max: Maximum log SNR (default: 5.0)
        antithetic_time_sampling: Use antithetic sampling for times (default: True)
        image_shape: Shape of input images (C, H, W)
        data_noise: Noise level for reconstruction (default: 1.0)
        lambdas: Loss weights [diffusion, latent, reconstruction] (default: (1.0, 1.0, 1.0))
        
        # Per-channel loss weighting
        channel_weights: Weights for each channel [DM, Gas, Stars] (default: (1.0, 1.0, 1.0))
        
        # Focal loss for stellar channel
        use_focal_loss: Enable focal loss for stellar channel (default: False)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
        
        # Auxiliary tasks
        use_param_prediction: Enable parameter prediction head (default: False)
        param_prediction_weight: Weight for parameter prediction loss (default: 0.01)
    """
    
    def __init__(
        self,
        score_model: nn.Module,
        # Noise schedule
        noise_schedule: str = "fixed_linear",
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        antithetic_time_sampling: bool = True,
        # Data configuration
        image_shape: Tuple[int, int, int] = (3, 128, 128),  # 3-channel mode: [DM, Gas, Stars]
        data_noise: float = 1e-3,  # Can be single float or tuple of floats (one per channel)
        lambdas: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        # Per-channel loss weighting
        channel_weights: Tuple[float, ...] = (1.0, 1.0, 1.0),  # [DM, Gas, Stars]
        # Focal loss for stellar channel
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        # Parameter prediction
        use_param_prediction: bool = False,
        param_prediction_weight: float = 0.01,
    ):
        super().__init__()
        
        print("\n" + "="*80)
        print("INITIALIZING CLEAN VDM MODEL")
        print("="*80)
        
        self.score_model = score_model
        self.image_shape = image_shape
        
        # Support per-channel data_noise: can be float or tuple
        if isinstance(data_noise, (tuple, list)):
            # Per-channel noise levels
            self.data_noise = torch.tensor(data_noise, dtype=torch.float32)
            self.use_per_channel_data_noise = True
        else:
            # Single noise level for all channels
            self.data_noise = data_noise
            self.use_per_channel_data_noise = False
        
        self.lambdas = lambdas
        
        print(f"\n📊 MODEL CONFIGURATION:")
        print(f"  Image shape: {image_shape}")
        print(f"  Noise schedule: {noise_schedule}")
        print(f"  Gamma range: [{gamma_min}, {gamma_max}]")
        if self.use_per_channel_data_noise:
            print(f"  Data noise (per-channel): {data_noise}")
        else:
            print(f"  Data noise: {data_noise}")
        print(f"  Loss weights (diffusion, latent, recons): {lambdas}")
        
        # Initialize noise schedule
        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(gamma_min=gamma_min, gamma_max=gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(gamma_min=gamma_min, gamma_max=gamma_max)
        elif noise_schedule == "learned_nn":
            self.gamma = NNSchedule(gamma_min=gamma_min, gamma_max=gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        
        self.antithetic_time_sampling = antithetic_time_sampling
        
        # Per-channel loss weighting
        self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32))
        print(f"\n🎯 CHANNEL WEIGHTS: {channel_weights}")
        
        # Focal loss for stellar channel
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        if use_focal_loss:
            print(f"\n🔥 FOCAL LOSS enabled for stellar channel (gamma={focal_gamma})")
        
        # Parameter prediction
        self.use_param_prediction = use_param_prediction
        self.param_prediction_weight = param_prediction_weight
        if use_param_prediction:
            print(f"\n✓ Parameter prediction enabled (weight={param_prediction_weight})")
        
        print("\n" + "="*80 + "\n")
    
    # ========== NOISE SCHEDULE HELPERS ==========
    
    def alpha(self, gamma_t: Tensor) -> Tensor:
        """Compute alpha coefficient: sqrt(sigmoid(-gamma))"""
        return torch.sqrt(torch.sigmoid(-gamma_t))
    
    def sigma(self, gamma_t: Tensor) -> Tensor:
        """Compute sigma coefficient: sqrt(sigmoid(gamma))"""
        return torch.sqrt(torch.sigmoid(gamma_t))
    
    def get_snr(self, gamma_t: Tensor) -> Tensor:
        """Compute Signal-to-Noise Ratio: exp(-gamma) = alpha^2 / sigma^2"""
        return torch.exp(-gamma_t.squeeze())
    
    # ========== DIFFUSION PROCESS ==========
    
    def variance_preserving_map(
        self, 
        x: Tensor, 
        times: Tensor, 
        noise: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Add noise to data sample using variance-preserving diffusion.
        
        z_t = alpha_t * x + sigma_t * noise
        
        Args:
            x: Clean data (B, C, H, W)
            times: Diffusion times in [0, 1] (B,)
            noise: Optional pre-generated noise (B, C, H, W)
        
        Returns:
            z_t: Noisy data (B, C, H, W)
            gamma_t: Log SNR values (B, 1, 1, 1)
        """
        with torch.enable_grad():
            # Reshape times for broadcasting: (B,) -> (B, 1, 1, 1)
            times_for_gamma = times.view((times.shape[0],) + (1,) * (x.ndim - 1))
            gamma_t = self.gamma(times_for_gamma)
        
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma(gamma_t)
        
        if noise is None:
            noise = torch.randn_like(x)
        
        z_t = alpha_t * x + sigma_t * noise
        
        return z_t, gamma_t
    
    def sample_times(self, batch_size: int, device: str) -> Tensor:
        """
        Sample diffusion times for batch.
        
        Args:
            batch_size: Number of times to sample
            device: Device to create tensor on
        
        Returns:
            times: Sampled times in [0, 1] (B,)
        """
        if self.antithetic_time_sampling:
            # Antithetic sampling: more stable gradients
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=device)
        else:
            # Uniform random sampling
            times = torch.rand(batch_size, device=device)
        
        return times
    
    # ========== LOSS FUNCTIONS ==========
    
    def get_diffusion_loss(
        self,
        pred_noise: Tensor,
        true_noise: Tensor,
        gamma_t: Tensor,
        times: Tensor,
        bpd_factor: float,
    ) -> Tensor:
        """
        Compute diffusion loss: MSE between predicted and true noise.
        
        This is the main loss component that trains the model to denoise.
        
        Args:
            pred_noise: Model's noise prediction (B, C, H, W)
            true_noise: Actual noise that was added (B, C, H, W)
            gamma_t: Log SNR values (B, 1, 1, 1)
            times: Diffusion times (B,)
            bpd_factor: Conversion factor to bits per dimension
        
        Returns:
            Diffusion loss per sample (B,)
        """
        # Compute gradient of gamma w.r.t. time
        # gamma_grad will have shape (B,) after the gradient computation
        gamma_grad = autograd.grad(
            gamma_t,
            times,
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]  # (B,)
        
        # Compute per-channel MSE
        n_channels = pred_noise.shape[1]
        channel_losses = []
        
        for c in range(n_channels):
            pred_c = pred_noise[:, c:c+1]  # (B, 1, H, W)
            true_c = true_noise[:, c:c+1]
            
            # MSE per channel, summed over spatial dimensions
            mse_c = ((pred_c - true_c) ** 2).flatten(start_dim=1).sum(dim=-1)  # (B,)
            
            # Apply focal loss to stellar channel (channel 2) if enabled
            if self.use_focal_loss and c == 2 and n_channels == 3:
                # ===== FOCAL LOSS FOR STELLAR CHANNEL =====
                # Focal loss adaptively weights examples based on difficulty:
                #   - Easy examples (low error): downweighted (less contribution to loss)
                #   - Hard examples (high error): upweighted (more contribution to loss)
                #
                # Formula: focal_loss = (1 - p_t)^gamma * MSE
                #   where p_t = exp(-MSE) represents prediction confidence
                #
                # How it works:
                #   1. p_t = exp(-mse_c): Convert error to confidence score
                #      - Low MSE → p_t ≈ 1 (high confidence, easy example)
                #      - High MSE → p_t ≈ 0 (low confidence, hard example)
                #
                #   2. focal_weight = (1 - p_t)^gamma: Compute adaptive weight
                #      - Easy examples: (1 - 1)^gamma ≈ 0 → downweighted
                #      - Hard examples: (1 - 0)^gamma ≈ 1 → upweighted
                #      - gamma controls the strength (typical: 2.0)
                #
                # Benefits:
                #   - Focuses training on challenging stellar regions
                #   - Prevents overwhelming contribution from easy regions
                #   - Improves learning of sparse stellar structures
                # ==========================================
                p_t = torch.exp(-mse_c)  # Confidence: high MSE → low p_t (hard)
                focal_weight = (1 - p_t) ** self.focal_gamma  # Adaptive weight
                weighted_mse_c = self.channel_weights[c] * focal_weight * mse_c
            else:
                # Standard weighted MSE for other channels
                weighted_mse_c = self.channel_weights[c] * mse_c
            
            channel_losses.append(weighted_mse_c)
        
        # Sum across channels
        total_loss = sum(channel_losses)  # (B,)
        
        # Apply VDM weighting: bpd_factor * 0.5 * total_loss * gamma_grad
        # This matches Eq. 17 in arxiv:2107.00630
        weighted_loss = bpd_factor * 0.5 * total_loss * gamma_grad
        
        return weighted_loss
    
    def get_param_prediction_loss(
        self,
        predicted_params: Tensor,
        true_params: Tensor,
    ) -> Tensor:
        """
        Compute parameter prediction loss: MSE between predicted and true parameters.
        
        Parameters are normalized to [0, 1] before comparison.
        
        Args:
            predicted_params: Predicted parameters (B, N_params)
            true_params: True parameters (B, N_params)
        
        Returns:
            MSE loss (scalar)
        """
        # Normalize true parameters to [0, 1]
        param_min = self.score_model.param_conditioning_embedding.min.to(true_params.device)
        param_max = self.score_model.param_conditioning_embedding.max.to(true_params.device)
        
        true_params_normalized = (true_params - param_min) / (param_max - param_min + 1e-8)
        true_params_normalized = torch.clamp(true_params_normalized, 0.0, 1.0)
        
        # MSE loss
        loss = mse_loss(predicted_params, true_params_normalized, reduction='mean')
        
        return loss
    
    def get_latent_loss(self, x: Tensor, bpd_factor: float) -> Tensor:
        """
        Compute latent loss: KL divergence to ensure prior is standard Gaussian.
        
        This ensures the final noisy distribution matches a standard Gaussian.
        
        Args:
            x: Clean data (B, C, H, W)
            bpd_factor: Conversion factor to bits per dimension
        
        Returns:
            KL loss per sample (B,)
        """
        gamma_1 = self.gamma(torch.tensor([1.0], device=x.device))
        sigma_1_sq = torch.sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x**2
        
        kl_loss = kl_std_normal(mean_sq, sigma_1_sq).flatten(start_dim=1).sum(dim=-1)
        
        return bpd_factor * kl_loss
    
    def get_reconstruction_loss(self, x: Tensor, bpd_factor: float) -> Tensor:
        """
        Compute reconstruction loss: Negative log-likelihood of data.
        
        This ensures the model can reconstruct the data at t=0.
        Supports per-channel data_noise for different uncertainty levels.
        
        Args:
            x: Clean data (B, C, H, W)
            bpd_factor: Conversion factor to bits per dimension
        
        Returns:
            NLL loss per sample (B,)
        """
        noise_0 = torch.randn_like(x)
        times = torch.tensor([0.0], device=x.device)
        z_0, gamma_0 = self.variance_preserving_map(x, times=times, noise=noise_0)
        
        alpha_0 = self.alpha(gamma_0)
        z_0_rescaled = z_0 / alpha_0
        
        if self.use_per_channel_data_noise:
            # Per-channel data noise: compute NLL for each channel separately
            data_noise = self.data_noise.to(x.device)  # (C,)
            nll = torch.zeros_like(x)
            
            for c in range(x.shape[1]):
                # Compute NLL for channel c with its specific noise level
                nll[:, c:c+1] = -Normal(
                    loc=z_0_rescaled[:, c:c+1], 
                    scale=data_noise[c]
                ).log_prob(x[:, c:c+1])
            
            nll = nll.flatten(start_dim=1).sum(dim=-1)
        else:
            # Single data noise for all channels
            nll = -Normal(loc=z_0_rescaled, scale=self.data_noise).log_prob(x)
            nll = nll.flatten(start_dim=1).sum(dim=-1)
        
        return bpd_factor * nll
    
    def get_reconstruction_loss_detailed(
        self,
        x: Tensor,
        bpd_factor: float,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute reconstruction loss with per-channel breakdown.
        
        Args:
            x: Clean data (B, C, H, W)
            bpd_factor: Conversion factor to bits per dimension
        
        Returns:
            Total NLL loss per sample (B,)
            Dictionary with per-channel reconstruction losses
        """
        noise_0 = torch.randn_like(x)
        times = torch.tensor([0.0], device=x.device)
        z_0, gamma_0 = self.variance_preserving_map(x, times=times, noise=noise_0)
        
        alpha_0 = self.alpha(gamma_0)
        z_0_rescaled = z_0 / alpha_0
        
        per_channel_losses = {}
        channel_names = ['dm', 'gas', 'stars']
        
        if self.use_per_channel_data_noise:
            # Per-channel data noise: compute NLL for each channel separately
            data_noise = self.data_noise.to(x.device)  # (C,)
            nll_per_channel = []
            
            for c in range(x.shape[1]):
                # Compute NLL for channel c with its specific noise level
                nll_c = -Normal(
                    loc=z_0_rescaled[:, c:c+1], 
                    scale=data_noise[c]
                ).log_prob(x[:, c:c+1])
                nll_c_sum = nll_c.flatten(start_dim=1).sum(dim=-1)
                nll_per_channel.append(nll_c_sum)
                
                # Store per-channel loss (with data_noise info)
                if c < len(channel_names):
                    per_channel_losses[f'recons_loss_{channel_names[c]}'] = (bpd_factor * nll_c_sum).mean().item()
                    per_channel_losses[f'data_noise_{channel_names[c]}'] = data_noise[c].item()
            
            # Total reconstruction loss
            nll = sum(nll_per_channel)
        else:
            # Single data noise for all channels
            nll = -Normal(loc=z_0_rescaled, scale=self.data_noise).log_prob(x)
            
            # Still compute per-channel breakdown for monitoring
            for c in range(x.shape[1]):
                nll_c = -Normal(loc=z_0_rescaled[:, c:c+1], scale=self.data_noise).log_prob(x[:, c:c+1])
                nll_c_sum = nll_c.flatten(start_dim=1).sum(dim=-1)
                if c < len(channel_names):
                    per_channel_losses[f'recons_loss_{channel_names[c]}'] = (bpd_factor * nll_c_sum).mean().item()
            
            per_channel_losses['data_noise_all'] = self.data_noise
            nll = nll.flatten(start_dim=1).sum(dim=-1)
        
        return bpd_factor * nll, per_channel_losses
    
    def get_loss(
        self,
        x: Tensor,
        conditioning: Tensor,
        param_conditioning: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute total VDM loss.
        
        Args:
            x: Clean data (B, C, H, W)
            conditioning: Conditioning information (B, C_cond, H, W)
            param_conditioning: Optional parameter conditioning (B, N_params)
            noise: Optional pre-generated noise (B, C, H, W)
        
        Returns:
            total_loss: Loss per sample (B,)
            metrics: Dictionary of loss components and metrics
        """
        # Bits per dimension factor
        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))
        
        # Sample diffusion times
        times = self.sample_times(x.shape[0], device=x.device).requires_grad_(True)
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(x)
        
        # Add noise to data
        x_t, gamma_t = self.variance_preserving_map(x=x, times=times, noise=noise)
        
        # Predict noise using score model
        score_output = self.score_model(
            x_t,
            g_t=gamma_t.squeeze(),
            conditioning=conditioning,
            param_conditioning=param_conditioning,
        )
        
        # Parse model output
        # Possible outputs:
        # 1. Just noise prediction: pred_noise
        # 2. Noise + params: (pred_noise, predicted_params)
        
        pred_noise = None
        predicted_params = None
        
        if isinstance(score_output, tuple):
            pred_noise, predicted_params = score_output
        else:
            pred_noise = score_output
        
        # ========== COMPUTE LOSS COMPONENTS ==========
        
        # 1. Diffusion loss (main component)
        diffusion_loss = self.get_diffusion_loss(
            pred_noise=pred_noise,
            true_noise=noise,
            gamma_t=gamma_t,
            times=times,
            bpd_factor=bpd_factor,
        )
        diffusion_loss = self.lambdas[0] * diffusion_loss
        
        # 2. Latent loss
        latent_loss = self.get_latent_loss(x=x, bpd_factor=bpd_factor)
        latent_loss = self.lambdas[1] * latent_loss
        
        # 3. Reconstruction loss (with per-channel breakdown)
        recons_loss, recons_per_channel = self.get_reconstruction_loss_detailed(x=x, bpd_factor=bpd_factor)
        recons_loss = self.lambdas[2] * recons_loss
        
        # 4. Parameter prediction loss (optional)
        param_loss = torch.tensor(0.0, device=x.device)
        if self.use_param_prediction and predicted_params is not None and param_conditioning is not None:
            param_loss = self.get_param_prediction_loss(predicted_params, param_conditioning)
            param_loss = self.param_prediction_weight * param_loss
        
        # Total loss
        total_loss = diffusion_loss + latent_loss + recons_loss + param_loss
        
        # ========== COMPUTE METRICS ==========
        
        metrics = {
            # Core losses
            "elbo": total_loss.mean().item(),
            "diffusion_loss": diffusion_loss.mean().item(),
            "latent_loss": latent_loss.mean().item(),
            "reconstruction_loss": recons_loss.mean().item(),
            
            # Auxiliary losses
            "param_loss": param_loss.item() if isinstance(param_loss, Tensor) else param_loss,
            
            # Per-channel reconstruction losses (with data_noise info)
            **recons_per_channel,
            
            # Channel-wise diffusion losses (for debugging)
            "diffusion_loss_dm": self._compute_channel_loss(pred_noise[:, 0:1], noise[:, 0:1], gamma_t, times, bpd_factor),
            "diffusion_loss_gas": self._compute_channel_loss(pred_noise[:, 1:2], noise[:, 1:2], gamma_t, times, bpd_factor),
        }
        
        # Add stellar channel loss (3-channel mode: [DM, Gas, Stars])
        if pred_noise.shape[1] == 3:
            metrics["diffusion_loss_stars"] = self._compute_channel_loss(
                pred_noise[:, 2:3], noise[:, 2:3], gamma_t, times, bpd_factor
            )
        
        # Monitoring metrics
        metrics["mean_snr"] = self.get_snr(gamma_t).mean().item()
        metrics["mean_gamma"] = gamma_t.mean().item()
        
        return total_loss, metrics
    
    def _compute_channel_loss(
        self,
        pred: Tensor,
        true: Tensor,
        gamma_t: Tensor,
        times: Tensor,
        bpd_factor: float,
    ) -> float:
        """Helper to compute loss for a single channel."""
        gamma_grad = autograd.grad(
            gamma_t,
            times,
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        mse = ((pred - true) ** 2).flatten(start_dim=1).sum(dim=-1)
        loss = bpd_factor * 0.5 * mse * gamma_grad
        
        return loss.mean().item()
    
    # ========== SAMPLING ==========
    
    def sample_zs_given_zt(
        self,
        zt: Tensor,
        conditioning: Tensor,
        t: Tensor,
        s: Tensor,
        param_conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample p(z_s | z_t, conditioning) for ancestral sampling.
        
        This is Eq. 34 in arxiv:2107.00630.
        
        Args:
            zt: Noisy data at time t (B, C, H, W)
            conditioning: Conditioning information (B, C_cond, H, W)
            t: Current time (scalar or (B,))
            s: Next time (scalar or (B,))
            param_conditioning: Optional parameter conditioning (B, N_params)
        
        Returns:
            zs: Noisy data at time s (B, C, H, W)
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        
        alpha_t = self.alpha(gamma_t)
        alpha_s = self.alpha(gamma_s)
        sigma_t = self.sigma(gamma_t)
        sigma_s = self.sigma(gamma_s)
        
        # Predict noise
        score_output = self.score_model(
            zt,
            conditioning=conditioning,
            g_t=gamma_t,
            param_conditioning=param_conditioning,
        )
        
        # Extract noise prediction (ignore auxiliary outputs)
        if isinstance(score_output, tuple):
            pred_noise = score_output[0]
        else:
            pred_noise = score_output
        
        # Compute mean and variance of p(z_s | z_t)
        mean = alpha_s / alpha_t * (zt - c * sigma_t * pred_noise)
        scale = sigma_s * torch.sqrt(c)
        
        # Sample
        zs = mean + scale * torch.randn_like(zt)
        
        return zs
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: int,
        device: str = 'cuda',
        z: Optional[Tensor] = None,
        param_conditioning: Optional[Tensor] = None,
        return_all: bool = False,
        verbose: bool = False
    ) -> Tensor:
        """
        Generate samples through iterative denoising.
        
        Args:
            conditioning: Conditioning information (B, C_cond, H, W)
            batch_size: Number of samples to generate
            n_sampling_steps: Number of denoising steps
            device: Device to generate samples on
            z: Optional initial noise (B, C, H, W)
            param_conditioning: Optional parameter conditioning (B, N_params)
            return_all: Return all intermediate steps
            verbose: Show progress bar
        
        Returns:
            Generated samples (B, C, H, W) or (n_steps, B, C, H, W) if return_all=True
        """
        # Ensure model is in eval mode for sampling
        self.score_model.eval()
        
        # Initialize from noise
        if z is None:
            z = torch.randn((batch_size, *self.image_shape), device=device)
        
        # Create time steps from 1.0 to 0.0
        steps = torch.linspace(1.0, 0.0, n_sampling_steps + 1, device=device)
        
        # Iterative denoising
        if return_all:
            zs = []
        
        iterator = trange(n_sampling_steps, desc="sampling") if verbose else range(n_sampling_steps)
        
        for i in iterator:
            z = self.sample_zs_given_zt(
                zt=z,
                conditioning=conditioning,
                t=steps[i],
                s=steps[i + 1],
                param_conditioning=param_conditioning,
            )
            
            if return_all:
                zs.append(z)
        
        if return_all:
            return torch.stack(zs, dim=0)
        
        return z


# ===================================================================
# LIGHTNING MODULE
# ===================================================================

class LightCleanVDM(LightningModule):
    """
    PyTorch Lightning wrapper for CleanVDM.
    
    Handles training, validation, and optimization.
    """
    
    def __init__(
        self,
        score_model: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 1.0e-5,
        lr_scheduler: str = 'onecycle',
        n_sampling_steps: int = 250,
        draw_figure=None,
        dataset: str = 'illustris',
        **kwargs
    ):
        super().__init__()
        
        self.model = CleanVDM(score_model=score_model, **kwargs)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.n_sampling_steps = n_sampling_steps
        self.draw_figure = draw_figure
        self.dataset = dataset
        
        self.save_hyperparameters(ignore=['score_model', 'draw_figure'])
    
    def forward(self, x: Tensor, conditioning: Tensor, param_conditioning: Optional[Tensor] = None):
        """Forward pass."""
        return self.model.get_loss(x, conditioning, param_conditioning)
    
    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Training step."""
        m_dm, large_scale, m_target, param_conditioning = batch
        
        # Concatenate spatial conditioning
        conditioning = torch.cat([m_dm, large_scale], dim=1)
        
        # Compute loss
        loss, metrics = self.model.get_loss(m_target, conditioning, param_conditioning)
        
        # Log metrics
        prog_bar_metrics = {'elbo', 'diffusion_loss', 'mask_accuracy'}
        
        for key, value in metrics.items():
            self.log(
                f"train/{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(key in prog_bar_metrics),
                logger=True,
                sync_dist=True,
            )
        
        return loss.mean()
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Validation step."""
        m_dm, large_scale, m_target, param_conditioning = batch
        
        # Concatenate spatial conditioning
        conditioning = torch.cat([m_dm, large_scale], dim=1)
        
        # Compute loss
        loss, metrics = self.model.get_loss(m_target, conditioning, param_conditioning)
        
        # Log metrics
        prog_bar_metrics = {'elbo', 'diffusion_loss'}
        
        for key, value in metrics.items():
            self.log(
                f"val/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(key in prog_bar_metrics),
                logger=True,
                sync_dist=True,
            )
        
        # Generate samples for visualization (first batch only)
        if batch_idx == 0 and self.draw_figure is not None:
            samples = self.model.sample(
                conditioning=conditioning[:4],
                batch_size=4,
                n_sampling_steps=self.n_sampling_steps,
                param_conditioning=param_conditioning[:4] if param_conditioning is not None else None,
                verbose=False
            )
            
            fig = self.draw_figure(
                samples[:4].cpu().numpy(),
                m_target[:4].cpu().numpy(),
                conditioning[:4].cpu().numpy(),
                self.dataset
            )
            self.logger.experiment.add_figure("val/samples", fig, self.global_step)
        
        return loss.mean()
    
    def draw_samples(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: int,
        param_conditioning: Optional[Tensor] = None,
        verbose: bool = False,
    ) -> Tensor:
        """Draw samples from the model."""
        return self.model.sample(
            conditioning=conditioning,
            batch_size=batch_size,
            n_sampling_steps=n_sampling_steps,
            param_conditioning=param_conditioning,
            verbose=verbose,
        )
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.lr_scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )

            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        elif self.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=1e-7
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/diffusion_loss',  # Monitor validation diffusion loss
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        elif self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-7
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        elif self.lr_scheduler == 'cosine_warmup':
            # Cosine annealing WITH linear warmup (good for higher LR)
            # Warmup for 5% of training, then cosine decay
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(0.05 * total_steps)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup
                    return float(step) / float(max(1, warmup_steps))
                else:
                    # Cosine decay
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return 0.5 * (1.0 + np.cos(np.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        elif self.lr_scheduler == 'cosine_restart':
            # Cosine annealing with WARM RESTARTS - great for escaping local minima
            # T_0=20: First restart after 20 epochs
            # T_mult=2: Double the period after each restart (20, 40, 80, ...)
            # This allows the model to escape local minima and explore
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=20,  # First cycle: 20 epochs
                T_mult=2,  # Double cycle length after each restart
                eta_min=self.learning_rate * 0.01  # Minimum LR = 1% of initial
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        elif self.lr_scheduler == 'constant':
            # No scheduler, constant learning rate
            return optimizer
        else:
            return optimizer
