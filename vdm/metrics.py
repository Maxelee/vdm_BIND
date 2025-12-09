"""
Metrics for evaluating generative models.
Simple, fast metrics for detecting overfitting and assessing sample quality.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
"""
Metrics for evaluating generative models.
Simple, fast metrics for detecting overfitting and assessing sample quality.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_mse(real_samples: torch.Tensor, generated_samples: torch.Tensor) -> float:
    """
    Compute Mean Squared Error between real and generated samples.
    
    Lower MSE = better match. Simple and fast metric for overfitting detection.
    
    Args:
        real_samples: Real samples, shape (N, C, H, W)
        generated_samples: Generated samples, shape (N, C, H, W)
        
    Returns:
        MSE (scalar)
    """
    mse = torch.mean((real_samples - generated_samples) ** 2).item()
    return float(mse)


def compute_mae(real_samples: torch.Tensor, generated_samples: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error between real and generated samples.
    
    Args:
        real_samples: Real samples, shape (N, C, H, W)
        generated_samples: Generated samples, shape (N, C, H, W)
        
    Returns:
        MAE (scalar)
    """
    mae = torch.mean(torch.abs(real_samples - generated_samples)).item()
    return float(mae)


def compute_ssim_batch(real_samples: torch.Tensor, generated_samples: torch.Tensor, 
                       data_range: Optional[float] = None) -> float:
    """
    Compute Structural Similarity Index (SSIM) between real and generated samples.
    
    SSIM is better for perceptual quality than MSE. Range: [-1, 1], higher is better.
    
    Args:
        real_samples: Real samples, shape (N, C, H, W)
        generated_samples: Generated samples, shape (N, C, H, W)
        data_range: Data range (max - min). If None, computed from data.
        
    Returns:
        Average SSIM across all samples
    """
    N, C, H, W = real_samples.shape
    
    # Convert to numpy
    real_np = real_samples.cpu().numpy()
    gen_np = generated_samples.cpu().numpy()
    
    # Compute data range if not provided
    if data_range is None:
        data_range = max(real_np.max() - real_np.min(), 
                        gen_np.max() - gen_np.min())
    
    ssim_values = []
    for i in range(N):
        for c in range(C):
            s = ssim(real_np[i, c], gen_np[i, c], data_range=data_range)
            ssim_values.append(s)
    
    return float(np.mean(ssim_values))


def compute_psnr_batch(real_samples: torch.Tensor, generated_samples: torch.Tensor,
                       data_range: Optional[float] = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between real and generated samples.
    
    Higher PSNR = better quality. Typical range: 20-50 dB.
    
    Args:
        real_samples: Real samples, shape (N, C, H, W)
        generated_samples: Generated samples, shape (N, C, H, W)
        data_range: Data range (max - min). If None, computed from data.
        
    Returns:
        Average PSNR across all samples
    """
    N, C, H, W = real_samples.shape
    
    # Convert to numpy
    real_np = real_samples.cpu().numpy()
    gen_np = generated_samples.cpu().numpy()
    
    # Compute data range if not provided
    if data_range is None:
        data_range = max(real_np.max() - real_np.min(), 
                        gen_np.max() - gen_np.min())
    
    psnr_values = []
    for i in range(N):
        for c in range(C):
            p = psnr(real_np[i, c], gen_np[i, c], data_range=data_range)
            psnr_values.append(p)
    
    return float(np.mean(psnr_values))


def compute_all_metrics(real_samples: torch.Tensor, generated_samples: torch.Tensor,
                       channel_names: Optional[list] = None) -> Dict[str, float]:
    """
    Compute all metrics: MSE, MAE, SSIM, PSNR (overall and per-channel).
    
    Args:
        real_samples: Real samples, shape (N, C, H, W)
        generated_samples: Generated samples, shape (N, C, H, W)
        channel_names: Names for each channel (e.g., ['dm_hydro', 'gas', 'stars'])
        
    Returns:
        Dictionary with all metrics
    """
    assert real_samples.shape == generated_samples.shape, \
        "Real and generated samples must have the same shape"
    
    N, C, H, W = real_samples.shape
    
    if channel_names is None:
        channel_names = [f'channel_{i}' for i in range(C)]
    
    results = {}
    
    # Overall metrics
    results['mse_overall'] = compute_mse(real_samples, generated_samples)
    results['mae_overall'] = compute_mae(real_samples, generated_samples)
    results['ssim_overall'] = compute_ssim_batch(real_samples, generated_samples)
    results['psnr_overall'] = compute_psnr_batch(real_samples, generated_samples)
    
    # Per-channel metrics
    for i, name in enumerate(channel_names):
        real_channel = real_samples[:, i:i+1, :, :]
        gen_channel = generated_samples[:, i:i+1, :, :]
        
        results[f'mse_{name}'] = compute_mse(real_channel, gen_channel)
        results[f'mae_{name}'] = compute_mae(real_channel, gen_channel)
        results[f'ssim_{name}'] = compute_ssim_batch(real_channel, gen_channel)
        results[f'psnr_{name}'] = compute_psnr_batch(real_channel, gen_channel)
    
    return results


# Keep old FID functions for backwards compatibility but mark as deprecated
def compute_statistics(samples: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance statistics for a set of samples.
    
    Args:
        samples: Tensor of shape (N, C, H, W) or (N, D) where N is number of samples
        
    Returns:
        mu: Mean vector of shape (D,)
        sigma: Covariance matrix of shape (D, D)
    """
    # Flatten spatial dimensions if needed
    if samples.ndim == 4:  # (N, C, H, W)
        N, C, H, W = samples.shape
        samples_flat = samples.reshape(N, -1)  # (N, C*H*W)
    else:
        samples_flat = samples
    
    # Convert to numpy for numerical stability
    if isinstance(samples_flat, torch.Tensor):
        samples_flat = samples_flat.cpu().numpy()
    
    # Compute statistics
    mu = np.mean(samples_flat, axis=0)
    sigma = np.cov(samples_flat, rowvar=False)
    
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Calculate Fréchet distance between two multivariate Gaussians.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        eps: Small value for numerical stability
        
    Returns:
        Fréchet distance (scalar)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be complex with small negative eigenvalues,
    # so we take the absolute value of eigenvalues
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m} too large')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_fid(
    real_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    eps: float = 1e-6
) -> float:
    """
    DEPRECATED: Use compute_all_metrics() instead for faster, simpler overfitting detection.
    
    This function is kept for backwards compatibility only.
    """
    # For backwards compatibility, return MSE scaled to look like FID
    return compute_mse(real_samples, generated_samples) * 1000.0


def compute_channel_wise_fid(
    real_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    channel_names: Optional[list] = None,
    eps: float = 1e-6
) -> dict:
    """
    DEPRECATED: Use compute_all_metrics() instead for faster, simpler overfitting detection.
    
    This function is kept for backwards compatibility only.
    """
    # Use the new metrics function and rename keys for compatibility
    metrics = compute_all_metrics(real_samples, generated_samples, channel_names)
    
    # Map new metric names to old FID-style names
    result = {}
    for key, value in metrics.items():
        if key.startswith('mse_'):
            # Scale MSE to look like FID values
            new_key = key.replace('mse_', 'fid_')
            result[new_key] = value * 1000.0
    
    return result


def compute_pixel_statistics(samples: torch.Tensor) -> dict:
    """
    Compute basic pixel-level statistics for samples.
    
    Args:
        samples: Tensor of shape (N, C, H, W)
        
    Returns:
        Dictionary with mean, std, min, max per channel
    """
    stats = {}
    num_channels = samples.shape[1]
    
    for i in range(num_channels):
        channel_data = samples[:, i, :, :]
        stats[f'channel_{i}_mean'] = float(channel_data.mean())
        stats[f'channel_{i}_std'] = float(channel_data.std())
        stats[f'channel_{i}_min'] = float(channel_data.min())
        stats[f'channel_{i}_max'] = float(channel_data.max())
    
    return stats
