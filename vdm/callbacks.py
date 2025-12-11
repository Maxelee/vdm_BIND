"""
PyTorch Lightning callbacks for validation plotting, FID tracking, gradient monitoring, and EMA.
"""

import os
import copy
import torch
import numpy as np
from typing import Optional, List, Dict, Any
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from vdm.validation_plots import ValidationPlotter
from vdm.metrics import compute_fid, compute_channel_wise_fid


# ============================================================================
# EMA (Exponential Moving Average) Callback
# ============================================================================

class EMACallback(Callback):
    """
    Exponential Moving Average (EMA) callback for PyTorch Lightning.
    
    Maintains a shadow copy of model weights that is updated with an exponential
    moving average of the training weights. EMA weights typically produce better
    samples in diffusion models.
    
    The EMA update rule is:
        ema_weight = decay * ema_weight + (1 - decay) * model_weight
    
    During validation/inference, the EMA weights are swapped in temporarily.
    The original weights are restored after validation.
    
    Args:
        decay: EMA decay factor (default: 0.9999). Higher = slower updates.
               Common values: 0.999, 0.9999, 0.99999
        update_after_step: Number of steps before starting EMA updates (warmup).
                          Allows model to train normally at start.
        update_every: Update EMA weights every N steps (default: 1).
        use_ema_for_validation: If True, use EMA weights for validation (default: True).
        save_ema_weights: If True, save EMA weights in checkpoints (default: True).
    
    Example:
        ```python
        ema_callback = EMACallback(decay=0.9999, update_after_step=1000)
        trainer = Trainer(callbacks=[ema_callback])
        ```
    
    To access EMA model for inference:
        ```python
        # During training (callback has reference)
        ema_model = ema_callback.ema_model
        
        # Or swap weights manually
        ema_callback.swap_weights(pl_module)  # Now pl_module has EMA weights
        # ... do inference ...
        ema_callback.swap_weights(pl_module)  # Restore original weights
        ```
    """
    
    def __init__(
        self,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        use_ema_for_validation: bool = True,
        save_ema_weights: bool = True,
    ):
        super().__init__()
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.use_ema_for_validation = use_ema_for_validation
        self.save_ema_weights = save_ema_weights
        
        # Shadow weights (will be initialized on first training step)
        self.ema_weights: Optional[Dict[str, torch.Tensor]] = None
        self.original_weights: Optional[Dict[str, torch.Tensor]] = None
        self.num_updates: int = 0
        self._weights_swapped: bool = False
        
        print(f"\n{'='*60}")
        print(f"EMA Callback Initialized")
        print(f"{'='*60}")
        print(f"  Decay: {decay}")
        print(f"  Update after step: {update_after_step}")
        print(f"  Update every: {update_every} steps")
        print(f"  Use EMA for validation: {use_ema_for_validation}")
        print(f"  Save EMA weights: {save_ema_weights}")
        print(f"{'='*60}\n")
    
    def _get_decay(self, step: int) -> float:
        """
        Optionally implement decay warmup. Currently returns fixed decay.
        Could be extended to use: decay * (1 - exp(-step / decay_warmup_steps))
        """
        return self.decay
    
    def _init_ema_weights(self, pl_module) -> None:
        """Initialize EMA weights as a copy of model weights."""
        self.ema_weights = {}
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.ema_weights[name] = param.data.clone().detach()
        print(f"✓ EMA: Initialized {len(self.ema_weights)} parameter tensors")
    
    @torch.no_grad()
    def _update_ema_weights(self, pl_module, step: int) -> None:
        """Update EMA weights with current model weights."""
        if self.ema_weights is None:
            return
        
        decay = self._get_decay(step)
        
        for name, param in pl_module.named_parameters():
            if name in self.ema_weights and param.requires_grad:
                # EMA update: ema = decay * ema + (1 - decay) * current
                self.ema_weights[name].mul_(decay).add_(param.data, alpha=1 - decay)
        
        self.num_updates += 1
    
    @torch.no_grad()
    def swap_weights(self, pl_module) -> None:
        """Swap model weights with EMA weights."""
        if self.ema_weights is None:
            return
        
        for name, param in pl_module.named_parameters():
            if name in self.ema_weights and param.requires_grad:
                # Swap: temp = model, model = ema, ema = temp
                temp = param.data.clone()
                param.data.copy_(self.ema_weights[name])
                self.ema_weights[name].copy_(temp)
        
        self._weights_swapped = not self._weights_swapped
    
    def on_train_start(self, trainer, pl_module) -> None:
        """Initialize EMA weights at the start of training."""
        if self.ema_weights is None:
            self._init_ema_weights(pl_module)
    
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        """Update EMA weights after each training batch."""
        step = trainer.global_step
        
        # Skip warmup period
        if step < self.update_after_step:
            return
        
        # Update every N steps
        if step % self.update_every != 0:
            return
        
        # Initialize if needed (e.g., when resuming)
        if self.ema_weights is None:
            self._init_ema_weights(pl_module)
        
        self._update_ema_weights(pl_module, step)
        
        # Log EMA stats occasionally
        if step % 1000 == 0 and step > 0:
            trainer.logger.log_metrics({
                'ema/num_updates': self.num_updates,
                'ema/decay': self._get_decay(step),
            }, step=step)
    
    def on_validation_start(self, trainer, pl_module) -> None:
        """Swap to EMA weights before validation."""
        if self.use_ema_for_validation and self.ema_weights is not None:
            self.swap_weights(pl_module)
    
    def on_validation_end(self, trainer, pl_module) -> None:
        """Restore original weights after validation."""
        if self._weights_swapped:
            self.swap_weights(pl_module)
    
    def on_test_start(self, trainer, pl_module) -> None:
        """Swap to EMA weights before testing."""
        if self.ema_weights is not None:
            self.swap_weights(pl_module)
    
    def on_test_end(self, trainer, pl_module) -> None:
        """Restore original weights after testing."""
        if self._weights_swapped:
            self.swap_weights(pl_module)
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            'ema_weights': self.ema_weights,
            'num_updates': self.num_updates,
            'decay': self.decay,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.ema_weights = state_dict.get('ema_weights')
        self.num_updates = state_dict.get('num_updates', 0)
        # Note: decay is set at init, but could be updated here if needed
        if self.ema_weights is not None:
            print(f"✓ EMA: Loaded {len(self.ema_weights)} parameter tensors "
                  f"({self.num_updates} updates)")
    
    def on_save_checkpoint(
        self, trainer, pl_module, checkpoint: Dict[str, Any]
    ) -> None:
        """Save EMA state in checkpoint."""
        if self.save_ema_weights and self.ema_weights is not None:
            checkpoint['ema_callback_state'] = self.state_dict()
    
    def on_load_checkpoint(
        self, trainer, pl_module, checkpoint: Dict[str, Any]
    ) -> None:
        """Load EMA state from checkpoint."""
        if 'ema_callback_state' in checkpoint:
            self.load_state_dict(checkpoint['ema_callback_state'])
    
    def get_ema_model_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get EMA weights as a state dict (for saving/loading separately).
        Returns None if EMA hasn't been initialized.
        """
        return self.ema_weights


class ValidationPlotCallback(Callback):
    """
    Lightning callback to generate validation plots during training.
    """
    
    def __init__(
        self,
        val_dataloader,
        dataset: str = 'IllustrisTNG',
        boxsize: float = 6.25,
        log_every_n_steps: int = 500,
        save_dir: str = 'validation_plots',
        n_image_samples: int = 4,
        n_power_samples: int = 16,
        stellar_stats_path: Optional[str] = None
    ):
        """
        Initialize validation plot callback.
        
        Parameters:
        -----------
        val_dataloader : DataLoader
            Validation data loader
        dataset : str
            Dataset name for normalization
        boxsize : float
            Physical box size in Mpc/h
        log_every_n_steps : int
            Generate plots every N training steps
        save_dir : str
            Directory to save plots
        n_image_samples : int
            Number of image samples to display
        n_power_samples : int
            Number of samples for statistics (power spectra, profiles)
        stellar_stats_path : str, optional
            Path to stellar normalization stats file (.npz)
            If provided, will use 4-channel mode with mask + magnitude
        """
        super().__init__()
        self.plotter = ValidationPlotter(
            dataset=dataset, 
            boxsize=boxsize,
            stellar_stats_path=stellar_stats_path
        )
        self.log_every_n_steps = log_every_n_steps
        self.save_dir = save_dir
        self.n_image_samples = n_image_samples
        self.n_power_samples = n_power_samples
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Store dataloader instead of batch to avoid memory leak
        print("Storing validation dataloader for callback...")
        self.val_dataloader = val_dataloader
        print(f"Validation dataloader stored")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called after every training batch."""
        global_step = trainer.global_step
        
        # Check if we should generate a plot
        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            self._generate_plot(trainer, pl_module, global_step)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Also generate plot at the end of validation epochs."""
        global_step = trainer.global_step
        self._generate_plot(trainer, pl_module, global_step)
    
    def _generate_plot(self, trainer, pl_module, global_step):
        """Generate and save validation plot."""
        print(f"\n{'='*60}")
        print(f"Generating validation plot at step {global_step}...")
        print(f"{'='*60}")
        
        # Load a fresh batch each time to avoid memory leak
        val_batch = next(iter(self.val_dataloader))
        
        # Move batch to correct device
        device = next(pl_module.parameters()).device
        val_batch_device = tuple(
            x.to(device) if x is not None and isinstance(x, torch.Tensor) else x 
            for x in val_batch
        )
        
        save_path = os.path.join(self.save_dir, f'validation_step_{global_step:06d}.png')
        
        try:
            self.plotter.generate_validation_plot(
                model=pl_module,
                val_batch=val_batch_device,
                global_step=global_step,
                n_samples=self.n_image_samples,
                n_power_samples=self.n_power_samples,
                save_path=save_path
            )
            print(f"✓ Validation plot saved to: {save_path}")
            
            # Log to tensorboard if available
            if trainer.logger is not None:
                try:
                    import matplotlib.pyplot as plt
                    from PIL import Image
                    import numpy as np
                    
                    # Read the saved image and log to tensorboard
                    img = Image.open(save_path)
                    img_array = np.array(img)
                    
                    # Log as image (convert to CHW format)
                    if len(img_array.shape) == 3:
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    else:
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    
                    trainer.logger.experiment.add_image(
                        'validation/comparison_plot',
                        img_tensor,
                        global_step=global_step,
                        dataformats='CHW'
                    )
                    print("✓ Validation plot logged to TensorBoard")
                except Exception as e:
                    print(f"Could not log to TensorBoard: {e}")
        
        except Exception as e:
            print(f"✗ Error generating validation plot: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*60}\n")


class FIDMonitorCallback(Callback):
    """
    Callback to compute and track FID (Fréchet Inception Distance) for both
    training and validation sets to monitor overfitting.
    
    This version is robust against OOM errors with:
    - Aggressive memory management
    - Mini-batch processing with immediate cleanup
    - Exception handling to prevent crashes
    - Reduced sampling steps
    - CPU-based statistics computation
    
    Example interpretation:
    - Train FID = 5.0, Val FID = 15.0 → Clear overfitting
    - Train FID = 8.0, Val FID = 9.5 → Good generalization
    """
    
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        compute_every_n_epochs: int = 5,  # Less frequent by default
        n_samples: int = 50,  # Reduced default
        channel_names: Optional[List[str]] = None,
        verbose: bool = True,
        max_history_size: int = 50,  # Reduced history
        n_sampling_steps: int = 25,  # Fewer diffusion steps
        batch_size: int = 4,  # Process in small batches
        enable_train_fid: bool = False,  # Skip training FID by default
    ):
        """
        Initialize FID monitoring callback.
        
        Parameters:
        -----------
        train_dataloader : DataLoader
            Training data loader
        val_dataloader : DataLoader
            Validation data loader
        compute_every_n_epochs : int
            Compute FID every N epochs (default: 5)
        n_samples : int
            Number of samples to use for FID computation (default: 50)
        channel_names : Optional[List[str]]
            Names for each channel (e.g., ['DMO', 'Gas', 'Stars'])
        verbose : bool
            Print detailed FID information
        max_history_size : int
            Maximum number of FID history entries to keep
        n_sampling_steps : int
            Number of diffusion sampling steps (fewer = faster, less memory)
        batch_size : int
            Process samples in batches of this size to reduce memory
        enable_train_fid : bool
            Whether to compute training FID (disabled by default to save memory)
        """
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.compute_every_n_epochs = compute_every_n_epochs
        self.n_samples = n_samples
        self.channel_names = channel_names or ['DMO', 'Gas', 'Stars']
        self.verbose = verbose
        self.max_history_size = max_history_size
        self.n_sampling_steps = n_sampling_steps
        self.batch_size = batch_size
        self.enable_train_fid = enable_train_fid
        
        # Track FID history (limited size to prevent OOM)
        self.train_fid_history = []
        self.val_fid_history = []
        
        print(f"FID Callback initialized: n_samples={n_samples}, "
              f"sampling_steps={n_sampling_steps}, batch_size={batch_size}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute FID at the end of validation epochs."""
        current_epoch = trainer.current_epoch
        
        # Only compute at specified intervals
        if current_epoch % self.compute_every_n_epochs != 0:
            return
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Computing FID at epoch {current_epoch}...")
            print(f"{'='*60}")
        
        try:
            device = next(pl_module.parameters()).device
            
            # Compute training FID only if enabled
            if self.enable_train_fid:
                train_fid = self._compute_fid_for_split(
                    pl_module, self.train_dataloader, device, split_name='train'
                )
            else:
                train_fid = None
            
            # Always compute validation FID
            val_fid = self._compute_fid_for_split(
                pl_module, self.val_dataloader, device, split_name='val'
            )
            
            # Store history
            if train_fid is not None:
                self.train_fid_history.append(train_fid)
                if len(self.train_fid_history) > self.max_history_size:
                    self.train_fid_history = self.train_fid_history[-self.max_history_size:]
            
            self.val_fid_history.append(val_fid)
            if len(self.val_fid_history) > self.max_history_size:
                self.val_fid_history = self.val_fid_history[-self.max_history_size:]
            
            # Log metrics
            if trainer.logger is not None and val_fid is not None:
                metrics = {
                    'fid/val_overall': val_fid['fid_overall'],
                }
                
                if train_fid is not None:
                    metrics['fid/train_overall'] = train_fid['fid_overall']
                    metrics['fid/overfitting_gap'] = val_fid['fid_overall'] - train_fid['fid_overall']
                
                # Add per-channel FIDs
                for name in self.channel_names:
                    if f'fid_{name}' in val_fid:
                        metrics[f'fid/val_{name}'] = val_fid[f'fid_{name}']
                    if train_fid is not None and f'fid_{name}' in train_fid:
                        metrics[f'fid/train_{name}'] = train_fid[f'fid_{name}']
                
                trainer.logger.log_metrics(metrics, step=trainer.global_step)
            
            # Print summary
            if self.verbose and val_fid is not None:
                print(f"\nFID Summary (Epoch {current_epoch}):")
                if train_fid is not None:
                    print(f"  Overall - Train: {train_fid['fid_overall']:.2f}, Val: {val_fid['fid_overall']:.2f}")
                    print(f"  Overfitting Gap: {val_fid['fid_overall'] - train_fid['fid_overall']:.2f}")
                else:
                    print(f"  Overall - Val: {val_fid['fid_overall']:.2f}")
                
                for name in self.channel_names:
                    if f'fid_{name}' in val_fid:
                        if train_fid is not None and f'fid_{name}' in train_fid:
                            print(f"  {name} - Train: {train_fid[f'fid_{name}']:.2f}, "
                                  f"Val: {val_fid[f'fid_{name}']:.2f}")
                        else:
                            print(f"  {name} - Val: {val_fid[f'fid_{name}']:.2f}")
                
                # Interpretation
                if train_fid is not None:
                    gap = val_fid['fid_overall'] - train_fid['fid_overall']
                    if gap > 5.0:
                        print(f"  ⚠️  High overfitting gap ({gap:.2f}) detected!")
                    elif gap < 2.0:
                        print(f"  ✓ Good generalization (gap: {gap:.2f})")
                
                print(f"{'='*60}\n")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️  FID computation failed due to OOM at epoch {current_epoch}")
                print(f"  Consider reducing: n_samples ({self.n_samples}), "
                      f"batch_size ({self.batch_size}), or n_sampling_steps ({self.n_sampling_steps})")
                print(f"  Training will continue normally.\n")
                # Clear cache and continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"\n⚠️  FID computation failed at epoch {current_epoch}: {e}")
                print(f"  Training will continue normally.\n")
        
        except Exception as e:
            print(f"\n⚠️  Unexpected error in FID computation at epoch {current_epoch}: {e}")
            print(f"  Training will continue normally.\n")
            import traceback
            traceback.print_exc()
    
    def _compute_fid_for_split(
        self,
        pl_module,
        dataloader,
        device,
        split_name: str
    ) -> Optional[dict]:
        """
        Compute FID for a data split with aggressive memory management.
        
        Returns None if computation fails.
        """
        pl_module.eval()
        
        # Collect statistics incrementally to avoid storing all samples
        real_stats_list = []
        gen_stats_list = []
        
        samples_collected = 0
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if samples_collected >= self.n_samples:
                        break
                    
                    # Determine how many samples to take from this batch
                    remaining = self.n_samples - samples_collected
                    
                    # Move batch to device
                    m_dm, large_scale, x, param_conditioning = batch
                    
                    # Only process what we need
                    batch_size = min(x.shape[0], remaining, self.batch_size)
                    
                    m_dm = m_dm[:batch_size].to(device)
                    large_scale = large_scale[:batch_size].to(device)
                    x = x[:batch_size]
                    if param_conditioning is not None:
                        param_conditioning = param_conditioning[:batch_size].to(device)
                    
                    # Concatenate conditioning
                    conditioning = torch.cat([m_dm, large_scale], dim=1)
                    
                    # Generate samples with reduced steps
                    generated = pl_module.draw_samples(
                        conditioning=conditioning,
                        batch_size=batch_size,
                        n_sampling_steps=self.n_sampling_steps,
                        param_conditioning=param_conditioning,
                        verbose=False
                    )
                    
                    # Move to CPU immediately and store
                    real_cpu = x.cpu()
                    gen_cpu = generated.cpu()
                    
                    real_stats_list.append(real_cpu)
                    gen_stats_list.append(gen_cpu)
                    
                    samples_collected += batch_size
                    
                    # Aggressive cleanup
                    del m_dm, large_scale, x, param_conditioning, conditioning, generated
                    del real_cpu, gen_cpu
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if self.verbose and batch_idx % 5 == 0:
                        print(f"  {split_name}: Collected {samples_collected}/{self.n_samples} samples")
            
            # Concatenate on CPU
            if len(real_stats_list) == 0:
                print(f"  ⚠️  No samples collected for {split_name} FID")
                return None
            
            real_samples = torch.cat(real_stats_list, dim=0)[:self.n_samples]
            generated_samples = torch.cat(gen_stats_list, dim=0)[:self.n_samples]
            
            # Clear intermediate lists
            del real_stats_list, gen_stats_list
            
            if self.verbose:
                print(f"  {split_name}: Computing FID with {real_samples.shape[0]} samples...")
            
            # Compute FID on CPU
            fid_dict = compute_channel_wise_fid(
                real_samples,
                generated_samples,
                channel_names=self.channel_names
            )
            
            # Final cleanup
            del real_samples, generated_samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            pl_module.train()
            return fid_dict
        
        except RuntimeError as e:
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pl_module.train()
            raise e


class GradientMonitorCallback(Callback):
    """
    Callback to monitor gradient norms and detect training instabilities.
    
    Tracks:
    - Global gradient norm
    - Per-layer gradient norms
    - Gradient norm statistics (mean, std, max)
    """
    
    def __init__(
        self,
        log_every_n_steps: int = 100,
        verbose: bool = False,
        max_history_size: int = 1000  # Limit history to prevent OOM
    ):
        """
        Initialize gradient monitoring callback.
        
        Parameters:
        -----------
        log_every_n_steps : int
            Log gradient statistics every N training steps
        verbose : bool
            Print gradient warnings
        max_history_size : int
            Maximum number of gradient norms to store in history
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose
        self.max_history_size = max_history_size
        
        # Track gradient history (limited size to prevent OOM)
        self.grad_norm_history = []
    
    def on_after_backward(self, trainer, pl_module):
        """Called after backward pass."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        # Compute gradient norms
        total_norm = 0.0
        grad_norms = {}
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_norms[name] = param_norm
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        self.grad_norm_history.append(total_norm)
        
        # Trim history to prevent memory issues (keep only recent entries)
        if len(self.grad_norm_history) > self.max_history_size:
            self.grad_norm_history = self.grad_norm_history[-self.max_history_size:]
        
        # Compute statistics
        if len(self.grad_norm_history) > 0:
            recent_norms = self.grad_norm_history[-100:]  # Last 100 steps
            mean_norm = np.mean(recent_norms)
            std_norm = np.std(recent_norms)
            max_norm = np.max(recent_norms)
            
            # Log to tensorboard
            if trainer.logger is not None:
                metrics = {
                    'gradients/total_norm': total_norm,
                    'gradients/mean_norm': mean_norm,
                    'gradients/std_norm': std_norm,
                    'gradients/max_norm': max_norm,
                }
                trainer.logger.log_metrics(metrics, step=trainer.global_step)
            
            # Check for instabilities
            if self.verbose:
                if total_norm > 10.0:
                    print(f"⚠️  High gradient norm detected: {total_norm:.2f}")
                elif total_norm < 1e-5:
                    print(f"⚠️  Very small gradient norm: {total_norm:.2e}")
                
                # Check for NaN or Inf
                if not np.isfinite(total_norm):
                    print(f"❌ Non-finite gradient norm detected: {total_norm}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print summary at end of epoch."""
        if self.verbose and len(self.grad_norm_history) > 0:
            recent_norms = self.grad_norm_history[-trainer.num_training_batches:]
            print(f"\nEpoch {trainer.current_epoch} Gradient Summary:")
            print(f"  Mean: {np.mean(recent_norms):.4f}")
            print(f"  Std: {np.std(recent_norms):.4f}")
            print(f"  Max: {np.max(recent_norms):.4f}")


class CustomEarlyStopping(EarlyStopping):
    """
    Enhanced early stopping with better logging and backwards compatibility.
    
    Stops training when validation loss doesn't improve for N epochs.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 30,
        verbose: bool = True,
        mode: str = 'min',
        min_delta: float = 0.0,
        **kwargs
    ):
        """
        Initialize early stopping callback.
        
        Parameters:
        -----------
        monitor : str
            Metric to monitor (default: 'val_loss')
        patience : int
            Number of epochs with no improvement before stopping (default: 30)
        verbose : bool
            Print early stopping messages
        mode : str
            'min' for metrics that should decrease, 'max' for metrics that should increase
        min_delta : float
            Minimum change to qualify as an improvement
        """
        super().__init__(
            monitor=monitor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            **kwargs
        )
        
        print(f"Early Stopping initialized: monitoring '{monitor}' with patience={patience}")
    
    def on_validation_end(self, trainer, pl_module):
        """Override to add custom logging."""
        super().on_validation_end(trainer, pl_module)
        
        # Log early stopping state
        if self.verbose and trainer.current_epoch % 5 == 0:
            metric_value = trainer.callback_metrics.get(self.monitor)
            if metric_value is not None:
                print(f"Early Stopping: {self.monitor}={metric_value:.4f}, "
                      f"Best={self.best_score:.4f}, "
                      f"Patience={self.wait_count}/{self.patience}")


class ComprehensiveMetricsCallback(Callback):
    """
    Callback to compute comprehensive metrics every N epochs during training.
    
    This runs the full evaluation pipeline (matching model_evaluation_comprehensive.ipynb)
    to track quality, mass conservation, radial profiles, and parameter sensitivity.
    """
    
    def __init__(
        self,
        config_path: str,
        output_dir: str,
        eval_every_n_epochs: int = 5,
        n_test_samples: int = 500,
        n_sampling_steps: int = 50,
        num_radial_bins: int = 50,
    ):
        """
        Args:
            config_path: Path to model config file
            output_dir: Base output directory  
            eval_every_n_epochs: Compute metrics every N epochs
            n_test_samples: Number of test samples to evaluate
            n_sampling_steps: Number of sampling steps for generation
            num_radial_bins: Number of radial bins for density profiles
        """
        super().__init__()
        self.config_path = config_path
        self.output_dir = output_dir
        self.eval_every_n_epochs = eval_every_n_epochs
        self.n_test_samples = n_test_samples
        self.n_sampling_steps = n_sampling_steps
        self.num_radial_bins = num_radial_bins
        
        # Create output directory for metrics
        self.metrics_dir = os.path.join(output_dir, 'comprehensive_metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Run comprehensive evaluation every N epochs."""
        current_epoch = trainer.current_epoch
        
        # Check if we should evaluate this epoch
        if current_epoch == 0 or (current_epoch + 1) % self.eval_every_n_epochs != 0:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 COMPUTING COMPREHENSIVE METRICS - Epoch {current_epoch + 1}")
        print(f"{'='*80}\n")
        
        try:
            import subprocess
            import sys
            from pathlib import Path
            
            # Find the most recent checkpoint
            checkpoint_callback = None
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_callback = callback
                    break
            
            if checkpoint_callback is None or checkpoint_callback.best_model_path == "":
                print("⚠️  No checkpoint available yet, skipping metrics")
                return
            
            checkpoint_path = checkpoint_callback.best_model_path
            print(f"Using checkpoint: {checkpoint_path}")
            
            # Output file for this epoch's metrics
            model_name = Path(self.config_path).stem
            metrics_file = os.path.join(
                self.metrics_dir,
                f'{model_name}_epoch_{current_epoch + 1:03d}.json'
            )
            
            # Run evaluate_comprehensive_metrics.py
            cmd = [
                sys.executable,
                'evaluate_comprehensive_metrics.py',
                '--config', self.config_path,
                '--checkpoint', checkpoint_path,
                '--n-test', str(self.n_test_samples),
                '--n-sampling-steps', str(self.n_sampling_steps),
                '--num-radial-bins', str(self.num_radial_bins),
                '--output', metrics_file,
            ]
            
            # Run in subprocess (non-blocking for training)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Metrics saved to: {metrics_file}")
                
                # Load and display key metrics
                import json
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                print(f"\n📊 Key Metrics (Epoch {current_epoch + 1}):")
                print(f"  Overall MSE: {metrics.get('mse_overall', 'N/A'):.4f}")
                print(f"  Stars Mass Error: {metrics.get('mass_mae_rel_error_stars', 'N/A'):.4f}")
                print(f"  Gas Mass Error: {metrics.get('mass_mae_rel_error_gas', 'N/A'):.4f}")
                print(f"  DM Mass Error: {metrics.get('mass_mae_rel_error_dm_hydro', 'N/A'):.4f}")
                
                # Log to TensorBoard
                if hasattr(pl_module, 'log'):
                    pl_module.log('metrics/mse_overall', metrics.get('mse_overall', 0))
                    pl_module.log('metrics/stars_mass_error', metrics.get('mass_mae_rel_error_stars', 0))
                    pl_module.log('metrics/gas_mass_error', metrics.get('mass_mae_rel_error_gas', 0))
                    pl_module.log('metrics/dm_mass_error', metrics.get('mass_mae_rel_error_dm_hydro', 0))
                
            else:
                print(f"❌ Metrics computation failed:")
                print(f"   {result.stderr[-500:]}" if result.stderr else "")
                
        except Exception as e:
            print(f"❌ Error computing comprehensive metrics: {e}")
        
        print(f"{'='*80}\n")

