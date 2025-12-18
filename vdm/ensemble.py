"""
Model Ensemble Support for VDM-BIND.

Provides tools for combining predictions from multiple models:
1. Simple averaging (equal weights)
2. Weighted averaging (learned or manual weights)
3. Stacking (meta-learner combines predictions)
4. Selection (choose best model per sample)

Usage:
    from vdm.ensemble import ModelEnsemble, WeightedEnsemble
    
    # Simple averaging
    ensemble = ModelEnsemble([model1, model2, model3])
    prediction = ensemble.predict(conditioning, params)
    
    # Weighted averaging
    ensemble = WeightedEnsemble(
        [model1, model2, model3],
        weights=[0.5, 0.3, 0.2]
    )
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Union, Callable
from dataclasses import dataclass
import warnings


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""
    mean: torch.Tensor          # Ensemble mean prediction
    std: Optional[torch.Tensor] = None  # Ensemble std (disagreement)
    individual: Optional[List[torch.Tensor]] = None  # Individual model predictions
    weights_used: Optional[List[float]] = None


class ModelEnsemble(nn.Module):
    """
    Simple model ensemble with equal weighting.
    
    Combines predictions from multiple models by averaging.
    All models should have the same input/output format.
    
    Args:
        models: List of models to ensemble
        reduction: How to combine predictions ('mean', 'median')
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        reduction: str = 'mean',
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.reduction = reduction
        
    def forward(self, *args, **kwargs):
        """Forward pass through all models and combine."""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(*args, **kwargs)
            predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)
        
        if self.reduction == 'mean':
            return stacked.mean(dim=0)
        elif self.reduction == 'median':
            return stacked.median(dim=0).values
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    @torch.no_grad()
    def draw_samples(
        self,
        conditioning: torch.Tensor,
        batch_size: int = 1,
        n_sampling_steps: Optional[int] = None,
        param_conditioning: Optional[torch.Tensor] = None,
        return_individual: bool = False,
    ) -> Union[torch.Tensor, EnsemblePrediction]:
        """
        Draw samples from ensemble.
        
        Args:
            conditioning: Input conditioning tensor
            batch_size: Batch size
            n_sampling_steps: Override sampling steps
            param_conditioning: Optional parameter conditioning
            return_individual: Return individual model predictions
            
        Returns:
            Ensemble prediction (or EnsemblePrediction if return_individual=True)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            
            kwargs = {}
            if n_sampling_steps is not None:
                kwargs['n_sampling_steps'] = n_sampling_steps
            if param_conditioning is not None:
                kwargs['param_conditioning'] = param_conditioning
            
            pred = model.draw_samples(
                conditioning,
                batch_size=batch_size,
                **kwargs
            )
            predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)  # (N_models, B, C, H, W)
        
        if self.reduction == 'mean':
            mean_pred = stacked.mean(dim=0)
        elif self.reduction == 'median':
            mean_pred = stacked.median(dim=0).values
        else:
            mean_pred = stacked.mean(dim=0)
        
        if return_individual:
            return EnsemblePrediction(
                mean=mean_pred,
                std=stacked.std(dim=0),
                individual=predictions,
                weights_used=[1.0 / self.n_models] * self.n_models,
            )
        
        return mean_pred


class WeightedEnsemble(nn.Module):
    """
    Weighted model ensemble.
    
    Combines predictions using specified weights per model.
    Weights can be fixed or learned.
    
    Args:
        models: List of models to ensemble
        weights: Per-model weights (will be normalized)
        learnable_weights: Whether to make weights learnable parameters
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.learnable_weights = learnable_weights
        
        # Initialize weights
        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models
        
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()  # Normalize
        
        if learnable_weights:
            # Use softmax to ensure valid weights
            self.weight_logits = nn.Parameter(torch.log(weights + 1e-8))
        else:
            self.register_buffer('weights', weights)
    
    @property
    def normalized_weights(self) -> torch.Tensor:
        """Get normalized weights."""
        if self.learnable_weights:
            return torch.softmax(self.weight_logits, dim=0)
        return self.weights
    
    def forward(self, *args, **kwargs):
        """Forward pass through all models and combine."""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(*args, **kwargs)
            predictions.append(pred)
        
        weights = self.normalized_weights
        
        # Weighted sum
        result = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            result = result + weights[i] * pred
        
        return result
    
    @torch.no_grad()
    def draw_samples(
        self,
        conditioning: torch.Tensor,
        batch_size: int = 1,
        n_sampling_steps: Optional[int] = None,
        param_conditioning: Optional[torch.Tensor] = None,
        return_individual: bool = False,
    ) -> Union[torch.Tensor, EnsemblePrediction]:
        """Draw weighted samples from ensemble."""
        predictions = []
        
        for model in self.models:
            model.eval()
            
            kwargs = {}
            if n_sampling_steps is not None:
                kwargs['n_sampling_steps'] = n_sampling_steps
            if param_conditioning is not None:
                kwargs['param_conditioning'] = param_conditioning
            
            pred = model.draw_samples(
                conditioning,
                batch_size=batch_size,
                **kwargs
            )
            predictions.append(pred)
        
        weights = self.normalized_weights
        
        # Weighted sum
        result = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            result = result + weights[i] * pred
        
        if return_individual:
            stacked = torch.stack(predictions, dim=0)
            return EnsemblePrediction(
                mean=result,
                std=stacked.std(dim=0),
                individual=predictions,
                weights_used=weights.tolist(),
            )
        
        return result
    
    def optimize_weights(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        criterion: Callable = nn.MSELoss(),
        n_epochs: int = 10,
        lr: float = 0.1,
    ):
        """
        Optimize ensemble weights on validation data.
        
        Args:
            val_dataloader: Validation dataloader
            criterion: Loss function to minimize
            n_epochs: Number of optimization epochs
            lr: Learning rate for weight optimization
        """
        if not self.learnable_weights:
            warnings.warn("Weights are not learnable. Creating learnable version.")
            self.learnable_weights = True
            self.weight_logits = nn.Parameter(
                torch.log(self.weights + 1e-8)
            )
        
        optimizer = torch.optim.Adam([self.weight_logits], lr=lr)
        
        device = next(self.models[0].parameters()).device
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            
            for batch in val_dataloader:
                # Unpack batch
                if len(batch) == 4:
                    conditions, large_scale, targets, params = batch
                    conditioning = torch.cat([conditions, large_scale], dim=1).to(device)
                    params = params.to(device)
                else:
                    conditions, targets, params = batch[:3]
                    conditioning = conditions.to(device)
                    params = params.to(device) if params is not None else None
                
                targets = targets.to(device)
                
                # Get predictions from all models
                predictions = []
                for model in self.models:
                    model.eval()
                    kwargs = {'param_conditioning': params} if params is not None else {}
                    pred = model.draw_samples(conditioning, batch_size=conditioning.shape[0], **kwargs)
                    predictions.append(pred)
                
                # Weighted combination
                weights = self.normalized_weights
                ensemble_pred = torch.zeros_like(predictions[0])
                for i, pred in enumerate(predictions):
                    ensemble_pred = ensemble_pred + weights[i] * pred
                
                # Compute loss
                loss = criterion(ensemble_pred, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.6f}, Weights = {self.normalized_weights.tolist()}")


class ChannelWiseEnsemble(nn.Module):
    """
    Ensemble with different weights per output channel.
    
    Useful when different models excel at different channels
    (e.g., one model better for DM, another for Stars).
    
    Args:
        models: List of models to ensemble
        channel_weights: Shape (n_models, n_channels) weights
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        channel_weights: Optional[torch.Tensor] = None,
        n_channels: int = 3,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.n_channels = n_channels
        
        if channel_weights is None:
            # Equal weights per channel
            channel_weights = torch.ones(self.n_models, n_channels) / self.n_models
        
        self.register_buffer('channel_weights', channel_weights)
    
    @torch.no_grad()
    def draw_samples(
        self,
        conditioning: torch.Tensor,
        batch_size: int = 1,
        n_sampling_steps: Optional[int] = None,
        param_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Draw samples with per-channel weighting."""
        predictions = []
        
        for model in self.models:
            model.eval()
            
            kwargs = {}
            if n_sampling_steps is not None:
                kwargs['n_sampling_steps'] = n_sampling_steps
            if param_conditioning is not None:
                kwargs['param_conditioning'] = param_conditioning
            
            pred = model.draw_samples(
                conditioning,
                batch_size=batch_size,
                **kwargs
            )
            predictions.append(pred)
        
        # predictions: list of (B, C, H, W)
        # channel_weights: (N_models, C)
        
        result = torch.zeros_like(predictions[0])
        for ch in range(self.n_channels):
            for i, pred in enumerate(predictions):
                result[:, ch] = result[:, ch] + self.channel_weights[i, ch] * pred[:, ch]
        
        return result


class DiversityEnsemble(nn.Module):
    """
    Ensemble that promotes diversity in predictions.
    
    Uses negative correlation learning or similar techniques
    to ensure ensemble members make diverse predictions.
    
    Args:
        models: List of models to ensemble
        diversity_weight: Weight for diversity regularization
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.diversity_weight = diversity_weight
    
    def compute_diversity_loss(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute diversity loss to encourage different predictions.
        
        Uses negative correlation: high loss when predictions are similar.
        """
        if len(predictions) < 2:
            return torch.tensor(0.0)
        
        # Stack predictions
        stacked = torch.stack(predictions, dim=0)  # (N, B, C, H, W)
        
        # Compute pairwise correlations
        flat = stacked.flatten(start_dim=2)  # (N, B, -1)
        
        # Mean across batch
        flat_mean = flat.mean(dim=1)  # (N, -1)
        
        # Correlation matrix
        flat_centered = flat_mean - flat_mean.mean(dim=1, keepdim=True)
        norms = flat_centered.norm(dim=1, keepdim=True)
        normalized = flat_centered / (norms + 1e-8)
        
        corr_matrix = normalized @ normalized.T  # (N, N)
        
        # Diversity loss: penalize high off-diagonal correlations
        mask = 1 - torch.eye(self.n_models, device=corr_matrix.device)
        diversity_loss = (corr_matrix * mask).abs().mean()
        
        return diversity_loss
    
    @torch.no_grad()
    def draw_samples(
        self,
        conditioning: torch.Tensor,
        batch_size: int = 1,
        n_sampling_steps: Optional[int] = None,
        param_conditioning: Optional[torch.Tensor] = None,
        return_diversity: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """Draw samples and optionally return diversity metric."""
        predictions = []
        
        for model in self.models:
            model.eval()
            
            kwargs = {}
            if n_sampling_steps is not None:
                kwargs['n_sampling_steps'] = n_sampling_steps
            if param_conditioning is not None:
                kwargs['param_conditioning'] = param_conditioning
            
            pred = model.draw_samples(
                conditioning,
                batch_size=batch_size,
                **kwargs
            )
            predictions.append(pred)
        
        # Simple mean
        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0)
        
        if return_diversity:
            diversity = self.compute_diversity_loss(predictions)
            return mean_pred, diversity.item()
        
        return mean_pred


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    model_class: type,
    model_kwargs: Dict = None,
    weights: Optional[List[float]] = None,
    device: str = 'cuda',
) -> WeightedEnsemble:
    """
    Create an ensemble from multiple checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        model_class: Model class to instantiate
        model_kwargs: Arguments for model constructor
        weights: Optional weights for each model
        device: Device to load models on
        
    Returns:
        WeightedEnsemble with loaded models
    """
    model_kwargs = model_kwargs or {}
    models = []
    
    for ckpt_path in checkpoint_paths:
        # Load checkpoint
        model = model_class.load_from_checkpoint(ckpt_path, **model_kwargs)
        model = model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded: {ckpt_path}")
    
    return WeightedEnsemble(models, weights=weights)


def create_multi_seed_ensemble(
    config_path: str,
    checkpoint_dir: str,
    n_seeds: int = 5,
    device: str = 'cuda',
) -> ModelEnsemble:
    """
    Create ensemble from models trained with different random seeds.
    
    Assumes checkpoints are named with seed suffix: model_seed0.ckpt, etc.
    
    Args:
        config_path: Path to config file
        checkpoint_dir: Directory containing checkpoints
        n_seeds: Number of seeds to load
        device: Device to load on
        
    Returns:
        ModelEnsemble with models from different seeds
    """
    from pathlib import Path
    from bind.config_loader import ConfigLoader
    from bind.model_manager import ModelManager
    
    checkpoint_dir = Path(checkpoint_dir)
    models = []
    
    # Try different naming conventions
    patterns = [
        "seed{}_best.ckpt",
        "model_seed{}.ckpt", 
        "version_{}/checkpoints/best.ckpt",
    ]
    
    for seed in range(n_seeds):
        loaded = False
        for pattern in patterns:
            ckpt_path = checkpoint_dir / pattern.format(seed)
            if ckpt_path.exists():
                config = ConfigLoader(config_path, verbose=False)
                config.best_ckpt = str(ckpt_path)
                _, model = ModelManager.initialize(config, verbose=False, skip_data_loading=True)
                model = model.to(device)
                model.eval()
                models.append(model)
                print(f"Loaded seed {seed}: {ckpt_path}")
                loaded = True
                break
        
        if not loaded:
            warnings.warn(f"Could not find checkpoint for seed {seed}")
    
    if len(models) == 0:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    return ModelEnsemble(models)
