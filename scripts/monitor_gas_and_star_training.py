#!/usr/bin/env python
"""
Monitor training progress for clean_vdm_gas_and_star model.

This script tracks:
- ELBO and diffusion loss
- Parameter prediction loss (10x stronger than baseline)
- Baryon fraction loss and correlation (new auxiliary loss)
- Per-channel losses

Usage:
    python scripts/monitor_gas_and_star_training.py [--plot] [--watch N]
    
Options:
    --plot      Generate and save training curve plots
    --watch N   Continuously monitor every N seconds (default: off)
"""

import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Configuration
LOG_DIR = '/mnt/home/mlee1/ceph/tb_logs3/clean_vdm_gas_and_star'
BASELINE_LOG_DIR = '/mnt/home/mlee1/ceph/tb_logs3/clean_vdm_regularized'
OUTPUT_DIR = LOG_DIR


def find_latest_version(log_dir):
    """Find the latest version directory."""
    if not os.path.exists(log_dir):
        return None
    versions = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    if not versions:
        return None
    return os.path.join(log_dir, sorted(versions)[-1])


def load_metrics(log_dir):
    """Load all metrics from TensorBoard logs."""
    if not HAS_TENSORBOARD:
        return None
        
    if not os.path.exists(log_dir):
        return None
    
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    metrics = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        if len(events) > 0:
            metrics[tag] = {
                'steps': np.array([e.step for e in events]),
                'values': np.array([e.value for e in events]),
                'last_value': events[-1].value,
                'last_step': events[-1].step
            }
    return metrics


def print_progress(metrics, show_all=False):
    """Print current training progress."""
    if not metrics:
        print("No metrics available yet.")
        return
    
    print("\n" + "=" * 70)
    print(f"CLEAN_VDM_GAS_AND_STAR TRAINING PROGRESS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Get current step
    if 'train/elbo_step' in metrics:
        current_step = metrics['train/elbo_step']['last_step']
        total_steps_per_epoch = 639  # Based on dataset size
        current_epoch = current_step / total_steps_per_epoch
        print(f"\nProgress: Step {current_step} (~Epoch {current_epoch:.2f})")
    
    # Key metrics
    key_metrics = [
        ('train/elbo_step', 'ELBO', 'lower is better'),
        ('train/diffusion_loss_step', 'Diffusion Loss', ''),
        ('train/param_loss_step', 'Param Loss', '10x weight'),
        ('train/baryon_loss_step', 'Baryon Loss', 'new'),
        ('train/baryon_correlation_step', 'Baryon Corr', 'target: 0.97'),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Metric':25s} {'Current':>12s} {'Start':>12s} {'Change':>12s}")
    print("-" * 70)
    
    for metric_key, metric_name, note in key_metrics:
        if metric_key in metrics:
            m = metrics[metric_key]
            current = m['last_value']
            start = m['values'][0] if len(m['values']) > 0 else current
            change = current - start
            note_str = f"  ({note})" if note else ""
            print(f"{metric_name:25s} {current:12.4f} {start:12.4f} {change:+12.4f}{note_str}")
    
    # Per-channel losses
    if show_all:
        print("\n" + "-" * 70)
        print("Per-Channel Diffusion Losses:")
        for ch in ['dm', 'gas', 'stars']:
            key = f'train/diffusion_loss_{ch}_step'
            if key in metrics:
                m = metrics[key]
                print(f"  {ch.upper():10s}: {m['last_value']:.4f}")
    
    # Baryon loss diagnostics
    print("\n" + "-" * 70)
    print("Baryon Fraction Diagnostics:")
    if 'train/baryon_fb_mean_step' in metrics:
        fb = metrics['train/baryon_fb_mean_step']['last_value']
        print(f"  Mean cosmic f_b (Ω_b/Ω_m): {fb:.4f}")
    if 'train/baryon_ratio_true_mean_step' in metrics:
        ratio = metrics['train/baryon_ratio_true_mean_step']['last_value']
        print(f"  Mean true Gas/DM ratio:    {ratio:.4f}")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("Interpretation:")
    if 'train/baryon_correlation_step' in metrics:
        corr = metrics['train/baryon_correlation_step']['last_value']
        if corr < 0.1:
            print(f"  ⚠️  Baryon correlation = {corr:.3f} (still learning)")
        elif corr < 0.5:
            print(f"  📈 Baryon correlation = {corr:.3f} (improving)")
        elif corr < 0.8:
            print(f"  ✓  Baryon correlation = {corr:.3f} (good progress)")
        else:
            print(f"  ✅ Baryon correlation = {corr:.3f} (excellent!)")
        print(f"      Target: ~0.97 (correlation in training data)")
    
    if 'train/param_loss_step' in metrics:
        param_loss = metrics['train/param_loss_step']['last_value']
        if param_loss > 0.01:
            print(f"  ⚠️  Param loss = {param_loss:.4f} (still high)")
        else:
            print(f"  ✓  Param loss = {param_loss:.4f} (decreasing)")


def plot_training_curves(metrics, output_path=None):
    """Generate training curve plots."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for plotting")
        return
    
    if not metrics or 'train/elbo_step' not in metrics:
        print("Not enough data to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('clean_vdm_gas_and_star Training Progress', fontsize=14)
    
    # Plot 1: ELBO
    ax = axes[0, 0]
    if 'train/elbo_step' in metrics:
        m = metrics['train/elbo_step']
        ax.plot(m['steps'], m['values'], 'b-', alpha=0.7, linewidth=0.5)
        # Moving average
        if len(m['values']) > 20:
            window = min(50, len(m['values']) // 5)
            ma = np.convolve(m['values'], np.ones(window)/window, mode='valid')
            ax.plot(m['steps'][window-1:], ma, 'b-', linewidth=2, label='Moving avg')
        ax.set_ylabel('ELBO')
        ax.set_xlabel('Step')
        ax.set_title('Training ELBO')
        ax.set_yscale('log')
    
    # Plot 2: Diffusion Loss
    ax = axes[0, 1]
    if 'train/diffusion_loss_step' in metrics:
        m = metrics['train/diffusion_loss_step']
        ax.plot(m['steps'], m['values'], 'g-', alpha=0.5, linewidth=0.5)
        if len(m['values']) > 20:
            window = min(50, len(m['values']) // 5)
            ma = np.convolve(m['values'], np.ones(window)/window, mode='valid')
            ax.plot(m['steps'][window-1:], ma, 'g-', linewidth=2)
        ax.set_ylabel('Diffusion Loss')
        ax.set_xlabel('Step')
        ax.set_title('Diffusion Loss')
    
    # Plot 3: Per-channel diffusion loss
    ax = axes[0, 2]
    colors = {'dm': 'blue', 'gas': 'green', 'stars': 'orange'}
    for ch, color in colors.items():
        key = f'train/diffusion_loss_{ch}_step'
        if key in metrics:
            m = metrics[key]
            ax.plot(m['steps'], m['values'], color=color, alpha=0.3, linewidth=0.5)
            if len(m['values']) > 20:
                window = min(50, len(m['values']) // 5)
                ma = np.convolve(m['values'], np.ones(window)/window, mode='valid')
                ax.plot(m['steps'][window-1:], ma, color=color, linewidth=2, label=ch.upper())
    ax.set_ylabel('Channel Loss')
    ax.set_xlabel('Step')
    ax.set_title('Per-Channel Diffusion Loss')
    ax.legend()
    
    # Plot 4: Param prediction loss
    ax = axes[1, 0]
    if 'train/param_loss_step' in metrics:
        m = metrics['train/param_loss_step']
        ax.plot(m['steps'], m['values'], 'r-', alpha=0.5, linewidth=0.5)
        if len(m['values']) > 20:
            window = min(50, len(m['values']) // 5)
            ma = np.convolve(m['values'], np.ones(window)/window, mode='valid')
            ax.plot(m['steps'][window-1:], ma, 'r-', linewidth=2)
        ax.set_ylabel('Param Loss')
        ax.set_xlabel('Step')
        ax.set_title('Parameter Prediction Loss (weight=0.1)')
        ax.set_yscale('log')
    
    # Plot 5: Baryon fraction loss
    ax = axes[1, 1]
    if 'train/baryon_loss_step' in metrics:
        m = metrics['train/baryon_loss_step']
        ax.plot(m['steps'], m['values'], 'm-', alpha=0.5, linewidth=0.5)
        if len(m['values']) > 20:
            window = min(50, len(m['values']) // 5)
            ma = np.convolve(m['values'], np.ones(window)/window, mode='valid')
            ax.plot(m['steps'][window-1:], ma, 'm-', linewidth=2)
        ax.set_ylabel('Baryon Loss')
        ax.set_xlabel('Step')
        ax.set_title('Baryon Fraction Loss (weight=0.1)')
    
    # Plot 6: Baryon correlation (KEY METRIC)
    ax = axes[1, 2]
    if 'train/baryon_correlation_step' in metrics:
        m = metrics['train/baryon_correlation_step']
        ax.plot(m['steps'], m['values'], 'c-', alpha=0.5, linewidth=0.5)
        if len(m['values']) > 20:
            window = min(50, len(m['values']) // 5)
            ma = np.convolve(m['values'], np.ones(window)/window, mode='valid')
            ax.plot(m['steps'][window-1:], ma, 'c-', linewidth=2, label='Moving avg')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.97, color='red', linestyle='--', alpha=0.7, label='Target (0.97)')
        ax.fill_between([m['steps'].min(), m['steps'].max()], 0.8, 1.0, 
                       alpha=0.1, color='green', label='Good range')
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Step')
        ax.set_title('Baryon Correlation (f_b vs Gas/DM) ⭐')
        ax.set_ylim(-0.2, 1.1)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def check_job_status():
    """Check if the SLURM job is still running."""
    import subprocess
    try:
        # Search for jobs containing 'gas' in the name
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', 'mlee1')],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split('\n')
        for line in lines[1:]:  # Skip header
            if 'gas' in line.lower() or 'vdm_gas' in line.lower():
                job_info = line.split()
                if len(job_info) >= 6:
                    job_id = job_info[0]
                    status = job_info[4]
                    runtime = job_info[5]
                    node = job_info[7] if len(job_info) > 7 else 'N/A'
                    return {'running': True, 'job_id': job_id, 'status': status, 
                            'runtime': runtime, 'node': node}
        return {'running': False}
    except Exception as e:
        return {'running': None, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Monitor clean_vdm_gas_and_star training')
    parser.add_argument('--plot', action='store_true', help='Generate training plots')
    parser.add_argument('--watch', type=int, default=0, 
                       help='Continuously monitor every N seconds')
    parser.add_argument('--all', action='store_true', help='Show all metrics')
    args = parser.parse_args()
    
    # Check job status
    job_status = check_job_status()
    if job_status.get('running'):
        print(f"🟢 Job {job_status['job_id']} RUNNING on {job_status['node']} "
              f"(runtime: {job_status['runtime']})")
    elif job_status.get('running') is False:
        print("🔴 Job not found in queue (may have completed or failed)")
    else:
        print(f"⚠️  Could not check job status: {job_status.get('error', 'unknown')}")
    
    # Find log directory
    log_dir = find_latest_version(LOG_DIR)
    if not log_dir:
        print(f"\nNo logs found in {LOG_DIR}")
        print("Training may not have started yet.")
        return
    
    print(f"📁 Log directory: {log_dir}")
    
    def update():
        metrics = load_metrics(log_dir)
        print_progress(metrics, show_all=args.all)
        
        if args.plot and metrics:
            output_path = os.path.join(OUTPUT_DIR, 'training_progress.png')
            plot_training_curves(metrics, output_path)
    
    if args.watch > 0:
        print(f"\n👁️  Watching every {args.watch} seconds (Ctrl+C to stop)")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                update()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        update()


if __name__ == '__main__':
    main()
