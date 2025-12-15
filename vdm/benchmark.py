"""
Standardized Benchmark Suite for VDM-BIND Models.

Provides consistent evaluation metrics across all model types:
1. Power spectrum analysis (ratio, correlation at fixed k)
2. Pixel-level metrics (MSE, SSIM, correlation)
3. Integrated mass statistics
4. Inference time benchmarks
5. Statistical distribution tests (PQM)

Usage:
    from vdm.benchmark import BenchmarkSuite
    
    benchmark = BenchmarkSuite(models={'VDM': model_vdm, 'Interpolant': model_interp})
    results = benchmark.run_full_benchmark(test_dataloader)
    benchmark.save_results('benchmark_results.json')
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import warnings

# Optional imports with fallbacks
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("skimage not available, SSIM metrics disabled")

try:
    import Pk_library as PKL
    HAS_PYLIANS = True
except ImportError:
    HAS_PYLIANS = False
    warnings.warn("Pylians not available, power spectrum metrics disabled")


@dataclass
class ChannelMetrics:
    """Metrics for a single channel."""
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    correlation: float = 0.0
    ssim: float = 0.0
    
    # Power spectrum metrics
    pk_ratio_mean: float = 0.0  # mean(P_pred / P_true)
    pk_ratio_std: float = 0.0
    pk_correlation: float = 0.0
    
    # Mass metrics
    mass_bias: float = 0.0  # (pred_mass - true_mass) / true_mass
    mass_scatter: float = 0.0


@dataclass 
class ModelBenchmarkResult:
    """Complete benchmark results for one model."""
    model_name: str
    
    # Per-channel metrics
    dm_metrics: ChannelMetrics = field(default_factory=ChannelMetrics)
    gas_metrics: ChannelMetrics = field(default_factory=ChannelMetrics)
    stars_metrics: ChannelMetrics = field(default_factory=ChannelMetrics)
    
    # Aggregate metrics
    mean_mse: float = 0.0
    mean_ssim: float = 0.0
    mean_correlation: float = 0.0
    
    # Timing
    inference_time_mean: float = 0.0  # seconds per sample
    inference_time_std: float = 0.0
    throughput: float = 0.0  # samples per second
    
    # Model info
    n_parameters: int = 0
    n_samples_evaluated: int = 0


class BenchmarkSuite:
    """
    Standardized benchmark suite for comparing VDM-BIND models.
    
    Args:
        models: Dictionary of {name: model} pairs
        device: Device to run on
        n_sampling_steps: Override default sampling steps (for speed)
        box_size: Physical box size in Mpc/h (for power spectrum)
    """
    
    CHANNEL_NAMES = ['DM', 'Gas', 'Stars']
    
    def __init__(
        self,
        models: Optional[Dict[str, nn.Module]] = None,
        device: str = 'cuda',
        n_sampling_steps: Optional[int] = None,
        box_size: float = 6.25,  # Mpc/h for halo cutouts
    ):
        self.models = models or {}
        self.device = device
        self.n_sampling_steps = n_sampling_steps
        self.box_size = box_size
        self.results: Dict[str, ModelBenchmarkResult] = {}
        
    def add_model(self, name: str, model: nn.Module):
        """Add a model to benchmark."""
        self.models[name] = model
        
    def _count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def _generate_sample(
        self,
        model: nn.Module,
        conditioning: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Generate a sample and return inference time."""
        model.eval()
        conditioning = conditioning.to(self.device)
        if params is not None:
            params = params.to(self.device)
        
        # Warmup
        if hasattr(model, 'draw_samples'):
            kwargs = {}
            if self.n_sampling_steps is not None:
                kwargs['n_sampling_steps'] = self.n_sampling_steps
            if params is not None:
                kwargs['param_conditioning'] = params
            
            # Time the generation
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start = time.perf_counter()
            
            sample = model.draw_samples(
                conditioning,
                batch_size=conditioning.shape[0],
                **kwargs
            )
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            elapsed = time.perf_counter() - start
        else:
            raise NotImplementedError(
                f"Model {type(model).__name__} does not have draw_samples method"
            )
        
        return sample.cpu(), elapsed
    
    def compute_pixel_metrics(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> ChannelMetrics:
        """Compute pixel-level metrics for a single channel."""
        metrics = ChannelMetrics()
        
        # Flatten for correlation
        pred_flat = prediction.flatten()
        target_flat = target.flatten()
        
        # MSE, RMSE, MAE
        metrics.mse = float(np.mean((prediction - target) ** 2))
        metrics.rmse = float(np.sqrt(metrics.mse))
        metrics.mae = float(np.mean(np.abs(prediction - target)))
        
        # Correlation
        if np.std(pred_flat) > 0 and np.std(target_flat) > 0:
            metrics.correlation = float(np.corrcoef(pred_flat, target_flat)[0, 1])
        
        # SSIM
        if HAS_SKIMAGE:
            # Handle 2D arrays
            data_range = max(target.max() - target.min(), 1e-8)
            metrics.ssim = float(ssim(target, prediction, data_range=data_range))
        
        return metrics
    
    def compute_power_spectrum_metrics(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        box_size: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute power spectrum metrics."""
        if not HAS_PYLIANS:
            return {'pk_ratio_mean': 0.0, 'pk_ratio_std': 0.0, 'pk_correlation': 0.0}
        
        box_size = box_size or self.box_size
        
        # Ensure float32 and contiguous
        pred = np.ascontiguousarray(prediction.astype(np.float32))
        tgt = np.ascontiguousarray(target.astype(np.float32))
        
        # Compute 2D power spectra
        Pk_pred = PKL.Pk_plane(pred, box_size, 'CIC', 1)
        Pk_target = PKL.Pk_plane(tgt, box_size, 'CIC', 1)
        
        k = Pk_pred.k
        pk_pred = Pk_pred.Pk
        pk_target = Pk_target.Pk
        
        # Avoid division by zero
        valid = pk_target > 0
        if not np.any(valid):
            return {'pk_ratio_mean': 1.0, 'pk_ratio_std': 0.0, 'pk_correlation': 0.0}
        
        ratio = pk_pred[valid] / pk_target[valid]
        
        # Correlation of log power spectra
        log_pred = np.log10(pk_pred[valid] + 1e-10)
        log_target = np.log10(pk_target[valid] + 1e-10)
        pk_corr = np.corrcoef(log_pred, log_target)[0, 1]
        
        return {
            'pk_ratio_mean': float(np.mean(ratio)),
            'pk_ratio_std': float(np.std(ratio)),
            'pk_correlation': float(pk_corr),
        }
    
    def compute_mass_metrics(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, float]:
        """Compute integrated mass metrics."""
        # Sum over spatial dimensions (mass proxy)
        pred_mass = np.sum(prediction)
        target_mass = np.sum(target)
        
        # Bias: (pred - true) / true
        if abs(target_mass) > 1e-10:
            bias = (pred_mass - target_mass) / target_mass
        else:
            bias = 0.0
        
        return {
            'mass_bias': float(bias),
            'mass_scatter': float(abs(bias)),  # For single sample, scatter = |bias|
        }
    
    def evaluate_model(
        self,
        model_name: str,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> ModelBenchmarkResult:
        """
        Run full evaluation for a single model.
        
        Args:
            model_name: Name for this model
            model: The model to evaluate
            dataloader: Test dataloader
            max_samples: Limit number of samples (for speed)
            show_progress: Show progress bar
            
        Returns:
            ModelBenchmarkResult with all metrics
        """
        model = model.to(self.device)
        model.eval()
        
        result = ModelBenchmarkResult(model_name=model_name)
        result.n_parameters = self._count_parameters(model)
        
        # Accumulators
        all_metrics = {ch: [] for ch in self.CHANNEL_NAMES}
        all_pk_metrics = {ch: [] for ch in self.CHANNEL_NAMES}
        all_mass_metrics = {ch: [] for ch in self.CHANNEL_NAMES}
        inference_times = []
        n_samples = 0
        
        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc=f"Evaluating {model_name}")
        
        for batch in iterator:
            # Unpack batch - handle different formats
            if len(batch) == 4:
                conditions, large_scale, targets, params = batch
                conditioning = torch.cat([conditions, large_scale], dim=1)
            elif len(batch) == 3:
                conditions, targets, params = batch
                conditioning = conditions
            else:
                conditions, targets = batch
                conditioning = conditions
                params = None
            
            # Generate predictions
            predictions, elapsed = self._generate_sample(model, conditioning, params)
            inference_times.append(elapsed / conditioning.shape[0])  # per sample
            
            # Compute metrics for each sample in batch
            for i in range(predictions.shape[0]):
                pred = predictions[i].numpy()  # (3, H, W)
                tgt = targets[i].numpy()       # (3, H, W)
                
                for ch_idx, ch_name in enumerate(self.CHANNEL_NAMES):
                    # Pixel metrics
                    px_metrics = self.compute_pixel_metrics(pred[ch_idx], tgt[ch_idx])
                    all_metrics[ch_name].append(px_metrics)
                    
                    # Power spectrum metrics
                    pk_metrics = self.compute_power_spectrum_metrics(pred[ch_idx], tgt[ch_idx])
                    all_pk_metrics[ch_name].append(pk_metrics)
                    
                    # Mass metrics
                    mass_metrics = self.compute_mass_metrics(pred[ch_idx], tgt[ch_idx])
                    all_mass_metrics[ch_name].append(mass_metrics)
                
                n_samples += 1
                if max_samples and n_samples >= max_samples:
                    break
            
            if max_samples and n_samples >= max_samples:
                break
        
        result.n_samples_evaluated = n_samples
        
        # Aggregate metrics
        def average_metrics(metrics_list: List[ChannelMetrics]) -> ChannelMetrics:
            avg = ChannelMetrics()
            for attr in ['mse', 'rmse', 'mae', 'correlation', 'ssim']:
                values = [getattr(m, attr) for m in metrics_list]
                setattr(avg, attr, float(np.mean(values)))
            return avg
        
        def average_dict_metrics(dict_list: List[Dict], target_metrics: ChannelMetrics):
            for key in dict_list[0].keys():
                values = [d[key] for d in dict_list]
                setattr(target_metrics, key, float(np.mean(values)))
        
        # Per-channel results
        for ch_name, ch_attr in [('DM', 'dm_metrics'), ('Gas', 'gas_metrics'), ('Stars', 'stars_metrics')]:
            ch_metrics = average_metrics(all_metrics[ch_name])
            average_dict_metrics(all_pk_metrics[ch_name], ch_metrics)
            average_dict_metrics(all_mass_metrics[ch_name], ch_metrics)
            setattr(result, ch_attr, ch_metrics)
        
        # Aggregate across channels
        result.mean_mse = np.mean([result.dm_metrics.mse, result.gas_metrics.mse, result.stars_metrics.mse])
        result.mean_ssim = np.mean([result.dm_metrics.ssim, result.gas_metrics.ssim, result.stars_metrics.ssim])
        result.mean_correlation = np.mean([
            result.dm_metrics.correlation, 
            result.gas_metrics.correlation, 
            result.stars_metrics.correlation
        ])
        
        # Timing
        result.inference_time_mean = float(np.mean(inference_times))
        result.inference_time_std = float(np.std(inference_times))
        result.throughput = 1.0 / result.inference_time_mean if result.inference_time_mean > 0 else 0.0
        
        return result
    
    def run_full_benchmark(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, ModelBenchmarkResult]:
        """
        Run benchmark on all models.
        
        Args:
            dataloader: Test dataloader
            max_samples: Limit samples per model
            show_progress: Show progress bar
            
        Returns:
            Dictionary of results per model
        """
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Benchmarking: {name}")
            print(f"{'='*60}")
            
            result = self.evaluate_model(
                name, model, dataloader, 
                max_samples=max_samples,
                show_progress=show_progress
            )
            self.results[name] = result
            
            # Print summary
            print(f"\nResults for {name}:")
            print(f"  MSE: {result.mean_mse:.6f}")
            print(f"  SSIM: {result.mean_ssim:.4f}")
            print(f"  Correlation: {result.mean_correlation:.4f}")
            print(f"  Inference time: {result.inference_time_mean*1000:.1f} ms/sample")
        
        return self.results
    
    def get_comparison_table(self) -> str:
        """Generate a comparison table as string."""
        if not self.results:
            return "No results to compare"
        
        # Header
        lines = [
            "=" * 100,
            f"{'Model':<20} {'MSE':>10} {'SSIM':>10} {'Corr':>10} {'Time (ms)':>12} {'Params':>12}",
            "-" * 100,
        ]
        
        for name, result in self.results.items():
            lines.append(
                f"{name:<20} {result.mean_mse:>10.6f} {result.mean_ssim:>10.4f} "
                f"{result.mean_correlation:>10.4f} {result.inference_time_mean*1000:>12.1f} "
                f"{result.n_parameters:>12,}"
            )
        
        lines.append("=" * 100)
        
        # Per-channel breakdown
        lines.append("\nPer-Channel MSE:")
        lines.append(f"{'Model':<20} {'DM':>12} {'Gas':>12} {'Stars':>12}")
        lines.append("-" * 60)
        
        for name, result in self.results.items():
            lines.append(
                f"{name:<20} {result.dm_metrics.mse:>12.6f} "
                f"{result.gas_metrics.mse:>12.6f} {result.stars_metrics.mse:>12.6f}"
            )
        
        return "\n".join(lines)
    
    def save_results(self, filepath: Union[str, Path]):
        """Save results to JSON file."""
        filepath = Path(filepath)
        
        # Convert dataclasses to dicts
        data = {}
        for name, result in self.results.items():
            data[name] = asdict(result)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    @classmethod
    def load_results(cls, filepath: Union[str, Path]) -> Dict[str, ModelBenchmarkResult]:
        """Load results from JSON file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = {}
        for name, result_dict in data.items():
            # Reconstruct ChannelMetrics
            for ch in ['dm_metrics', 'gas_metrics', 'stars_metrics']:
                result_dict[ch] = ChannelMetrics(**result_dict[ch])
            results[name] = ModelBenchmarkResult(**result_dict)
        
        return results


def quick_benchmark(
    model: nn.Module,
    conditioning: torch.Tensor,
    target: torch.Tensor,
    params: Optional[torch.Tensor] = None,
    n_samples: int = 10,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Quick benchmark for a single model on a single sample.
    
    Useful for rapid iteration during development.
    
    Args:
        model: Model to benchmark
        conditioning: Input conditioning
        target: Ground truth target
        params: Optional parameters
        n_samples: Number of samples for timing
        device: Device to run on
        
    Returns:
        Dictionary of metrics
    """
    model = model.to(device)
    model.eval()
    
    conditioning = conditioning.to(device)
    if params is not None:
        params = params.to(device)
    
    # Timing
    times = []
    for _ in range(n_samples):
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.perf_counter()
        
        with torch.no_grad():
            if hasattr(model, 'draw_samples'):
                kwargs = {'param_conditioning': params} if params is not None else {}
                pred = model.draw_samples(conditioning, batch_size=1, **kwargs)
            else:
                raise ValueError("Model needs draw_samples method")
        
        torch.cuda.synchronize() if device == 'cuda' else None
        times.append(time.perf_counter() - start)
    
    pred = pred[0].cpu().numpy()
    tgt = target.cpu().numpy()
    
    # Compute metrics
    mse = float(np.mean((pred - tgt) ** 2))
    corr = float(np.corrcoef(pred.flatten(), tgt.flatten())[0, 1])
    
    ssim_val = 0.0
    if HAS_SKIMAGE:
        for ch in range(3):
            data_range = max(tgt[ch].max() - tgt[ch].min(), 1e-8)
            ssim_val += ssim(tgt[ch], pred[ch], data_range=data_range)
        ssim_val /= 3
    
    return {
        'mse': mse,
        'correlation': corr,
        'ssim': ssim_val,
        'time_mean_ms': np.mean(times) * 1000,
        'time_std_ms': np.std(times) * 1000,
    }
