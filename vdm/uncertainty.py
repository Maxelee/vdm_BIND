"""
Uncertainty Quantification for VDM-BIND Models.

This module provides tools for estimating prediction uncertainty through:
1. Multi-realization sampling (inherent stochasticity in diffusion models)
2. MC Dropout (approximate Bayesian inference)
3. Ensemble disagreement (combining multiple models)
4. Calibration analysis (reliability diagrams)

Usage:
    from vdm.uncertainty import UncertaintyEstimator
    
    estimator = UncertaintyEstimator(model, n_samples=10)
    mean, std, samples = estimator.predict_with_uncertainty(conditioning, params)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Callable
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class UncertaintyResult:
    """Container for uncertainty estimation results."""
    mean: np.ndarray  # Mean prediction (C, H, W) or (B, C, H, W)
    std: np.ndarray   # Standard deviation (same shape as mean)
    samples: Optional[np.ndarray] = None  # Individual samples (N, C, H, W) or (N, B, C, H, W)
    
    # Per-pixel statistics
    variance: Optional[np.ndarray] = None
    coefficient_of_variation: Optional[np.ndarray] = None  # std / mean
    
    # Calibration metrics (if ground truth provided)
    calibration_error: Optional[float] = None
    coverage_probabilities: Optional[Dict[float, float]] = None


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty for diffusion models.
    
    Supports multiple uncertainty estimation methods:
    - Multi-realization: Sample multiple times from the diffusion model
    - MC Dropout: Enable dropout at inference time
    - Ensemble: Combine predictions from multiple models
    
    Args:
        model: The diffusion model (LightCleanVDM, LightInterpolant, etc.)
        n_samples: Number of samples for uncertainty estimation
        use_mc_dropout: Whether to enable dropout at inference time
        mc_dropout_rate: Dropout rate for MC Dropout (if different from training)
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        use_mc_dropout: bool = False,
        mc_dropout_rate: Optional[float] = None,
    ):
        self.model = model
        self.n_samples = n_samples
        self.use_mc_dropout = use_mc_dropout
        self.mc_dropout_rate = mc_dropout_rate
        self.device = next(model.parameters()).device
        
    def _enable_mc_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        def apply_dropout(module):
            if isinstance(module, nn.Dropout):
                module.train()
                if self.mc_dropout_rate is not None:
                    module.p = self.mc_dropout_rate
        self.model.apply(apply_dropout)
        
    def _disable_mc_dropout(self):
        """Disable dropout layers (restore eval mode)."""
        def disable_dropout(module):
            if isinstance(module, nn.Dropout):
                module.eval()
        self.model.apply(disable_dropout)
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        conditioning: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        n_sampling_steps: Optional[int] = None,
        return_samples: bool = True,
        show_progress: bool = True,
    ) -> UncertaintyResult:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            conditioning: Input conditioning tensor (B, C_cond, H, W)
            params: Optional parameter conditioning (B, N_params)
            n_sampling_steps: Override default sampling steps
            return_samples: Whether to return individual samples
            show_progress: Show progress bar
            
        Returns:
            UncertaintyResult with mean, std, and optionally individual samples
        """
        self.model.eval()
        
        if self.use_mc_dropout:
            self._enable_mc_dropout()
        
        # Move to device
        conditioning = conditioning.to(self.device)
        if params is not None:
            params = params.to(self.device)
        
        batch_size = conditioning.shape[0]
        samples_list = []
        
        # Generate multiple samples
        iterator = range(self.n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling for uncertainty")
        
        for _ in iterator:
            # Use model's draw_samples method if available
            if hasattr(self.model, 'draw_samples'):
                kwargs = {}
                if n_sampling_steps is not None:
                    kwargs['n_sampling_steps'] = n_sampling_steps
                if params is not None:
                    kwargs['param_conditioning'] = params
                    
                sample = self.model.draw_samples(
                    conditioning,
                    batch_size=batch_size,
                    **kwargs
                )
            else:
                # Fallback for models without draw_samples
                raise NotImplementedError(
                    f"Model {type(self.model).__name__} does not have draw_samples method"
                )
            
            samples_list.append(sample.cpu().numpy())
        
        if self.use_mc_dropout:
            self._disable_mc_dropout()
        
        # Stack samples: (N, B, C, H, W)
        samples = np.stack(samples_list, axis=0)
        
        # Compute statistics along sample dimension
        mean = np.mean(samples, axis=0)  # (B, C, H, W)
        std = np.std(samples, axis=0)    # (B, C, H, W)
        variance = np.var(samples, axis=0)
        
        # Coefficient of variation (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = np.where(np.abs(mean) > 1e-8, std / np.abs(mean), 0.0)
        
        return UncertaintyResult(
            mean=mean,
            std=std,
            samples=samples if return_samples else None,
            variance=variance,
            coefficient_of_variation=cv,
        )
    
    def compute_calibration(
        self,
        predictions_mean: np.ndarray,
        predictions_std: np.ndarray,
        ground_truth: np.ndarray,
        confidence_levels: List[float] = [0.5, 0.68, 0.9, 0.95, 0.99],
    ) -> Tuple[Dict[float, float], float]:
        """
        Compute calibration metrics.
        
        For well-calibrated uncertainty, the fraction of ground truth values
        falling within confidence intervals should match the confidence level.
        
        Args:
            predictions_mean: Mean predictions
            predictions_std: Standard deviation predictions
            ground_truth: True values
            confidence_levels: Confidence levels to evaluate
            
        Returns:
            Tuple of (coverage_dict, expected_calibration_error)
        """
        from scipy import stats
        
        # Flatten for analysis
        mean_flat = predictions_mean.flatten()
        std_flat = predictions_std.flatten()
        gt_flat = ground_truth.flatten()
        
        # Compute coverage for each confidence level
        coverage = {}
        for conf in confidence_levels:
            # For Gaussian, find z-score for this confidence level
            z = stats.norm.ppf((1 + conf) / 2)
            
            lower = mean_flat - z * std_flat
            upper = mean_flat + z * std_flat
            
            # Fraction of GT within interval
            in_interval = (gt_flat >= lower) & (gt_flat <= upper)
            coverage[conf] = float(np.mean(in_interval))
        
        # Expected Calibration Error (ECE)
        ece = np.mean([abs(coverage[c] - c) for c in confidence_levels])
        
        return coverage, ece
    
    def reliability_diagram_data(
        self,
        predictions_mean: np.ndarray,
        predictions_std: np.ndarray,
        ground_truth: np.ndarray,
        n_bins: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute data for reliability diagram.
        
        Args:
            predictions_mean: Mean predictions
            predictions_std: Standard deviation predictions
            ground_truth: True values
            n_bins: Number of bins for the diagram
            
        Returns:
            Tuple of (expected_confidence, observed_confidence, bin_counts)
        """
        from scipy import stats
        
        # Compute z-scores (how many std away is GT from mean)
        z_scores = np.abs((ground_truth - predictions_mean) / (predictions_std + 1e-8))
        
        # Convert z-scores to confidence levels
        # P(|Z| < z) = 2 * Phi(z) - 1
        confidence = 2 * stats.norm.cdf(z_scores.flatten()) - 1
        
        # Bin by predicted confidence (based on std)
        # Higher std = lower confidence
        predicted_conf = 1 / (1 + predictions_std.flatten())  # Simple mapping
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        observed_conf = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (predicted_conf >= bin_edges[i]) & (predicted_conf < bin_edges[i + 1])
            if np.sum(mask) > 0:
                observed_conf[i] = np.mean(confidence[mask])
                bin_counts[i] = np.sum(mask)
        
        return bin_centers, observed_conf, bin_counts


class EnsembleUncertainty:
    """
    Uncertainty estimation from model ensembles.
    
    Combines predictions from multiple models (different architectures,
    seeds, or checkpoints) to estimate uncertainty.
    
    Args:
        models: List of models to ensemble
        weights: Optional weights for each model (default: equal)
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
    ):
        self.models = models
        self.n_models = len(models)
        
        if weights is None:
            self.weights = [1.0 / self.n_models] * self.n_models
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.device = next(models[0].parameters()).device
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        conditioning: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        n_samples_per_model: int = 1,
        n_sampling_steps: Optional[int] = None,
        show_progress: bool = True,
    ) -> UncertaintyResult:
        """
        Generate ensemble predictions with uncertainty.
        
        Args:
            conditioning: Input conditioning tensor
            params: Optional parameter conditioning
            n_samples_per_model: Samples per model (for stochastic models)
            n_sampling_steps: Override default sampling steps
            show_progress: Show progress bar
            
        Returns:
            UncertaintyResult with ensemble statistics
        """
        conditioning = conditioning.to(self.device)
        if params is not None:
            params = params.to(self.device)
        
        batch_size = conditioning.shape[0]
        all_samples = []
        
        # Collect samples from each model
        iterator = enumerate(self.models)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Ensemble sampling")
        
        for i, model in iterator:
            model.eval()
            
            for _ in range(n_samples_per_model):
                if hasattr(model, 'draw_samples'):
                    kwargs = {}
                    if n_sampling_steps is not None:
                        kwargs['n_sampling_steps'] = n_sampling_steps
                    if params is not None:
                        kwargs['param_conditioning'] = params
                    
                    sample = model.draw_samples(
                        conditioning,
                        batch_size=batch_size,
                        **kwargs
                    )
                else:
                    raise NotImplementedError(
                        f"Model {type(model).__name__} does not have draw_samples method"
                    )
                
                all_samples.append(sample.cpu().numpy())
        
        # Stack: (N_total, B, C, H, W)
        samples = np.stack(all_samples, axis=0)
        
        # Weighted mean (for weighted ensemble)
        # Expand weights for broadcasting
        if n_samples_per_model == 1:
            weights_expanded = np.array(self.weights)[:, None, None, None, None]
            mean = np.sum(samples * weights_expanded, axis=0)
        else:
            # Equal weight per sample when multiple samples per model
            mean = np.mean(samples, axis=0)
        
        # Standard deviation (ensemble disagreement)
        std = np.std(samples, axis=0)
        variance = np.var(samples, axis=0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = np.where(np.abs(mean) > 1e-8, std / np.abs(mean), 0.0)
        
        return UncertaintyResult(
            mean=mean,
            std=std,
            samples=samples,
            variance=variance,
            coefficient_of_variation=cv,
        )
    
    def compute_ensemble_disagreement(
        self,
        conditioning: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        n_sampling_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute pairwise disagreement between ensemble members.
        
        Returns:
            Disagreement matrix (N_models, N_models) with MSE between pairs
        """
        conditioning = conditioning.to(self.device)
        if params is not None:
            params = params.to(self.device)
        
        batch_size = conditioning.shape[0]
        predictions = []
        
        for model in self.models:
            model.eval()
            if hasattr(model, 'draw_samples'):
                kwargs = {}
                if n_sampling_steps is not None:
                    kwargs['n_sampling_steps'] = n_sampling_steps
                if params is not None:
                    kwargs['param_conditioning'] = params
                
                pred = model.draw_samples(
                    conditioning,
                    batch_size=batch_size,
                    **kwargs
                )
                predictions.append(pred.cpu().numpy())
        
        # Compute pairwise MSE
        n = len(predictions)
        disagreement = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                disagreement[i, j] = np.mean((predictions[i] - predictions[j]) ** 2)
        
        return disagreement


def add_mc_dropout(model: nn.Module, dropout_rate: float = 0.1) -> nn.Module:
    """
    Add dropout layers to a model for MC Dropout inference.
    
    This modifies the model in-place to add dropout after each
    activation function.
    
    Args:
        model: Model to modify
        dropout_rate: Dropout probability
        
    Returns:
        Modified model
    """
    # Find all ReLU/GELU/SiLU activations and add dropout after them
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
            # Get parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # Wrap activation with dropout
            setattr(
                parent,
                parts[-1],
                nn.Sequential(module, nn.Dropout(p=dropout_rate))
            )
    
    return model


def compute_uncertainty_maps(
    samples: np.ndarray,
    method: str = 'std',
) -> np.ndarray:
    """
    Compute per-pixel uncertainty maps from multiple samples.
    
    Args:
        samples: Array of samples (N, C, H, W) or (N, B, C, H, W)
        method: 'std', 'var', 'iqr' (interquartile range), 'entropy'
        
    Returns:
        Uncertainty map with same spatial dimensions
    """
    if method == 'std':
        return np.std(samples, axis=0)
    elif method == 'var':
        return np.var(samples, axis=0)
    elif method == 'iqr':
        q75 = np.percentile(samples, 75, axis=0)
        q25 = np.percentile(samples, 25, axis=0)
        return q75 - q25
    elif method == 'entropy':
        # Discretize and compute entropy
        from scipy import stats
        n_bins = 50
        entropy_map = np.zeros(samples.shape[1:])
        
        for idx in np.ndindex(samples.shape[1:]):
            values = samples[(slice(None),) + idx]
            hist, _ = np.histogram(values, bins=n_bins, density=True)
            hist = hist[hist > 0]  # Remove zeros
            entropy_map[idx] = stats.entropy(hist)
        
        return entropy_map
    else:
        raise ValueError(f"Unknown method: {method}")
