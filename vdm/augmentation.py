import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import Resize as VisionResize
import joblib
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import Resize as VisionResize
import joblib
import numpy as np


import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import Resize as VisionResize
import joblib
import numpy as np


import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import Resize as VisionResize
import joblib
import numpy as np


class Normalize(torch.nn.Module):
    def __init__(
        self,
        mean_input,
        std_input,
        mean_target_dmh,
        std_target_dmh,
        mean_target_gas,
        std_target_gas,
        # Stellar normalization stats (REQUIRED for 3-channel mode unless using quantile)
        mean_target_stars=None,
        std_target_stars=None,
        # Quantile transformation (optional, alternative to Z-score for stellar)
        quantile_transformer=None,
    ):
        """
        Normalization transform for astrophysical data (3-channel mode).
        
        3-channel mode (CleanVDM): [DM, Gas, Stars]
           - All channels log-transformed: log10(x + 1)
           - DM and Gas: Z-score normalized
           - Stars: Z-score normalized OR quantile transformed (if quantile_transformer provided)
                    When using quantile: small noise (1e-4) added to ALL pixels to match transformer training
        
        Args:
            mean_input: Mean for DM input normalization
            std_input: Std for DM input normalization
            mean_target_dmh: Mean for DM target normalization
            std_target_dmh: Std for DM target normalization
            mean_target_gas: Mean for Gas normalization
            std_target_gas: Std for Gas normalization
            mean_target_stars: Mean for stellar normalization (log space) - Required if not using quantile
            std_target_stars: Std for stellar normalization (log space) - Required if not using quantile
            quantile_transformer: sklearn QuantileTransformer for stellar channel (optional)
        """
        super().__init__()
        self.mean_input = mean_input
        self.std_input = std_input
        self.mean_target_dmh = mean_target_dmh
        self.std_target_dmh = std_target_dmh
        self.mean_target_gas = mean_target_gas
        self.std_target_gas = std_target_gas
        self.mean_target_stars = mean_target_stars
        self.std_target_stars = std_target_stars
        self.quantile_transformer = quantile_transformer
        
        # Validate stellar normalization configuration
        if quantile_transformer is None:
            # Using Z-score: require mean and std
            if mean_target_stars is None or std_target_stars is None:
                raise ValueError(
                    "⚠️  Stellar normalization stats are REQUIRED for 3-channel mode!\n"
                    "   Please provide mean_target_stars and std_target_stars,\n"
                    "   OR provide quantile_transformer for quantile normalization"
                )
        
        print(f"\n📊 NORMALIZATION CONFIG (3-channel mode):")
        print(f"  DM input: mean={mean_input:.4f}, std={std_input:.4f}")
        print(f"  DM target: mean={mean_target_dmh:.4f}, std={std_target_dmh:.4f}")
        print(f"  Gas target: mean={mean_target_gas:.4f}, std={std_target_gas:.4f}")
        if quantile_transformer is not None:
            print(f"  Stars target: QUANTILE TRANSFORMATION (n_quantiles={len(quantile_transformer.quantiles_)})")
        else:
            print(f"  Stars target: mean={mean_target_stars:.4f}, std={std_target_stars:.4f}")

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """
        Apply normalization to data (3-channel mode).
        
        Input format (after log transform):
            conditioning: (1, H, W) - DM input (already log10)
            large_scale: (N, H, W) - Large-scale fields (already log10)
            target: (3, H, W) - [DM_log, Gas_log, Stars_log] (ALL log-transformed)
        
        Output format (3-channel mode):
            conditioning: (1, H, W) - Normalized DM input
            large_scale: (N, H, W) - Normalized large-scale fields
            target: (3, H, W) - [DM_norm, Gas_norm, Stars_norm] (ALL Z-score normalized)
        """
        conditioning, large_scale, target = inputs
        
        # Normalize conditioning (DM data)
        transformed_conditioning = F.normalize(
            conditioning,
            self.mean_input,
            self.std_input,
        )
        
        # Normalize large_scale (DM data)
        # Handle both (1, H, W) and (N, H, W) shapes
        if large_scale.shape[0] == 1:
            # Single channel: use scalar mean/std
            transformed_large_scale = F.normalize(
                large_scale,
                self.mean_input,
                self.std_input,
            )
        else:
            # Multiple channels: replicate mean/std for each channel
            mean_large_scale = [self.mean_input] * large_scale.shape[0]
            std_large_scale = [self.std_input] * large_scale.shape[0]
            transformed_large_scale = F.normalize(
                large_scale,
                mean_large_scale,
                std_large_scale,
            )
        
        # Normalize target - 3-channel mode
        # Input target is (3, H, W) -> [DM_log, Gas_log, Stars_log] (ALL already log-transformed!)
        # Output target is (3, H, W) -> [DM_norm, Gas_norm, Stars_norm]
        
        dmh_channel = target[0:1, :, :]
        gas_channel = target[1:2, :, :]
        stellar_channel = target[2:3, :, :]  # Already in log space
        
        # Normalize DM and Gas with Z-score
        transformed_dmh = F.normalize(
            dmh_channel,
            [self.mean_target_dmh],
            [self.std_target_dmh]
        )
        transformed_gas = F.normalize(
            gas_channel,
            [self.mean_target_gas],
            [self.std_target_gas]
        )
        
        # Normalize stellar channel: quantile OR Z-score
        if self.quantile_transformer is not None:
            # Quantile transformation
            # Add small noise to ALL pixels (matches transformer training in notebook)
            stellar_with_noise = stellar_channel.clone()
            noise = torch.randn_like(stellar_channel) * 1e-4
            stellar_with_noise = stellar_channel + noise
            
            # Apply quantile transformation
            original_shape = stellar_with_noise.shape
            stellar_flat = stellar_with_noise.flatten().cpu().numpy().reshape(-1, 1)
            transformed_stellar_np = self.quantile_transformer.transform(stellar_flat)
            transformed_stars = torch.from_numpy(transformed_stellar_np).reshape(original_shape).to(stellar_channel.device)
        else:
            # Z-score normalization (original behavior)
            transformed_stars = F.normalize(
                stellar_channel,
                [self.mean_target_stars],
                [self.std_target_stars]
            )
        
        # Output 3 channels: [DM, Gas, Stars] - ALL normalized consistently
        transformed_target = torch.cat([
            transformed_dmh,
            transformed_gas,
            transformed_stars
        ], dim=0)
        
        return transformed_conditioning, transformed_large_scale, transformed_target




    
class Resize(torch.nn.Module):
    def __init__(
        self,
        size=(32, 32),
    ):
        super().__init__()
        self.size = size
        self.resize = VisionResize(
            size,
            antialias=True,
        )

    def forward(self, sample: Tensor) -> Tensor:
        conditioning, large_scale, target = sample
        return self.resize(conditioning), self.resize(large_scale), self.resize(target)

class Translate(object):

    def __call__(self, sample):
        in_img, large_scale, tgt_img = sample # (C, H, W)

        x_shift = torch.randint(in_img.shape[-2], (1,)).item()
        y_shift = torch.randint(in_img.shape[-1], (1,)).item()
        
        in_img = torch.roll(in_img, (x_shift, y_shift), dims=(-2, -1))
        large_scale = torch.roll(large_scale, (x_shift, y_shift), dims=(-2, -1))
        tgt_img = torch.roll(tgt_img, (x_shift, y_shift), dims=(-2, -1))

        return in_img, large_scale, tgt_img

class Flip(object):

    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, 'flipping is ambiguous for 1D scalars/vectors'

        self.axes = torch.randint(2, (self.ndim,), dtype=torch.bool)
        self.axes = torch.arange(self.ndim)[self.axes]

        in_img, large_scale, tgt_img = sample

#        if in_img.shape[0] == self.ndim:  # flip vector components
#            in_img[self.axes] = - in_img[self.axes]

        shifted_axes = (1 + self.axes).tolist()
        in_img = torch.flip(in_img, shifted_axes)
        large_scale = torch.flip(large_scale, shifted_axes)

        if tgt_img.shape[0] == self.ndim:  # flip vector components
            tgt_img[self.axes] = - tgt_img[self.axes]

        shifted_axes = (1 + self.axes).tolist()
        tgt_img = torch.flip(tgt_img, shifted_axes)

        return in_img, large_scale, tgt_img

class RandomRotate(object):
    def __init__(self, degrees=180):
        self.degrees = degrees

    def __call__(self, sample):
        in_img, large_scale, tgt_img = sample  # Each is (C, H, W)
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
        # Apply the same rotation to both input and target
        in_img = F.rotate(in_img, angle, interpolation=F.InterpolationMode.BILINEAR)
        large_scale = F.rotate(large_scale, angle, interpolation=F.InterpolationMode.BILINEAR)
        tgt_img = F.rotate(tgt_img, angle, interpolation=F.InterpolationMode.BILINEAR)
        return in_img, large_scale, tgt_img

class Permutate(object):

    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, 'permutation is not necessary for 1D fields'

        self.axes = torch.randperm(self.ndim)

        in_img, large_scale, tgt_img = sample

        if in_img.shape[0] == self.ndim:  # permutate vector components
            in_img = in_img[self.axes]

        shifted_axes = [0] + (1 + self.axes).tolist()
        in_img = in_img.permute(shifted_axes)
        large_scale = large_scale.permute(shifted_axes)

        if tgt_img.shape[0] == self.ndim:  # permutate vector components
            tgt_img = tgt_img[self.axes]

        shifted_axes = [0] + (1 + self.axes).tolist()
        tgt_img = tgt_img.permute(shifted_axes)

        return in_img, large_scale, tgt_img
