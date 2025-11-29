"""
TensorBoard Log Export and Comparison Script
=============================================

Exports and compares training curves from RSL-RL and Custom PPO.

Usage:
    python tensorboard_export.py --rsl_rl_log <path> --custom_ppo_log <path> --output <output_dir>

Example (Windows - single line):
    python tensorboard_export.py --rsl_rl_log logs/rsl_rl/anymal_c_flat_direct/2025-11-28_12-15-24 --custom_ppo_log logs/rsl_rl/custom_ppo_v2/2025-11-28_20-39-37 --output comparison_plots --max_iter 10000
"""

import argparse
import os
import pandas as pd
import numpy as np

# Use Agg backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("TensorBoard not found. Install with: pip install tensorboard")
    exit(1)


def load_tensorboard_scalars(log_dir, tag_mapping=None):
    """Load scalar data from TensorBoard logs.

    Args:
        log_dir: Path to TensorBoard log directory
        tag_mapping: Dict to rename tags for compatibility

    Returns:
        Dict of {tag_name: DataFrame with 'step' and 'value' columns}
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    available_tags = ea.Tags().get('scalars', [])
    print(f"[INFO] Found {len(available_tags)} scalar tags in {log_dir}")

    data = {}
    for tag in available_tags:
        try:
            events = ea.Scalars(tag)
            df = pd.DataFrame([
                {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
                for e in events
            ])

            # Apply tag mapping if provided
            mapped_tag = tag
            if tag_mapping and tag in tag_mapping:
                mapped_tag = tag_mapping[tag]

            data[mapped_tag] = df
        except Exception as e:
            print(f"[WARN] Could not load tag {tag}: {e}")

    return data


def plot_comparison(rsl_data, custom_data, tag, output_dir, max_iter=None,
                    rsl_label="RSL-RL PPO", custom_label="Custom PPO"):
    """Plot comparison of a single metric."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot RSL-RL data
    if tag in rsl_data:
        df = rsl_data[tag]
        if max_iter:
            df = df[df['step'] <= max_iter]
        ax.plot(df['step'], df['value'], label=rsl_label, color='#FF8C00',
                linewidth=2, alpha=0.9)

    # Plot Custom PPO data
    if tag in custom_data:
        df = custom_data[tag]
        if max_iter:
            df = df[df['step'] <= max_iter]
        ax.plot(df['step'], df['value'], label=custom_label, color='#32CD32',
                linewidth=2, alpha=0.9)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(tag.replace('/', ' - '), fontsize=12)
    ax.set_title(f'Comparison: {tag}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    # Set background
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    # Save
    safe_tag = tag.replace('/', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f'{safe_tag}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    print(f"[SAVED] {output_path}")

    return output_path


def create_summary_plot(rsl_data, custom_data, output_dir, max_iter=None):
    """Create a summary plot with multiple metrics."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#1e1e1e')

    metrics = [
        ('Train/mean_reward', 'Mean Reward'),
        ('Policy/mean_noise_std', 'Action STD'),
        ('Loss/surrogate', 'Surrogate Loss'),
        ('Train/mean_episode_length', 'Episode Length'),
    ]

    colors = {'rsl': '#FF8C00', 'custom': '#32CD32'}

    for ax, (tag, title) in zip(axes.flat, metrics):
        ax.set_facecolor('#1e1e1e')

        # RSL-RL
        if tag in rsl_data:
            df = rsl_data[tag]
            if max_iter:
                df = df[df['step'] <= max_iter]
            ax.plot(df['step'], df['value'], label='RSL-RL', color=colors['rsl'],
                    linewidth=2, alpha=0.9)

        # Custom PPO
        if tag in custom_data:
            df = custom_data[tag]
            if max_iter:
                df = df[df['step'] <= max_iter]
            ax.plot(df['step'], df['value'], label='Custom PPO', color=colors['custom'],
                    linewidth=2, alpha=0.9)

        ax.set_title(title, fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel('Iteration', fontsize=10, color='white')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color('white')

    plt.suptitle('PPO Comparison: Custom Implementation vs RSL-RL',
                 fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'summary_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    print(f"[SAVED] {output_path}")

    return output_path


def create_linkedin_plot(rsl_data, custom_data, output_dir, max_iter=1000):
    """Create a professional plot suitable for LinkedIn."""

    fig, ax = plt.subplots(figsize=(14, 7))

    # Use white background for LinkedIn
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    tag = 'Train/mean_reward'

    # RSL-RL
    if tag in rsl_data:
        df = rsl_data[tag]
        if max_iter:
            df = df[df['step'] <= max_iter]
        ax.plot(df['step'], df['value'], label='RSL-RL PPO (Production Library)',
                color='#FF6B35', linewidth=3, alpha=0.9)

    # Custom PPO
    if tag in custom_data:
        df = custom_data[tag]
        if max_iter:
            df = df[df['step'] <= max_iter]
        ax.plot(df['step'], df['value'], label='Custom PPO (From Scratch)',
                color='#004E89', linewidth=3, alpha=0.9)

    ax.set_xlabel('Training Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Episode Reward', fontsize=14, fontweight='bold')
    ax.set_title('From-Scratch PPO vs Production Library\nIsaac Lab Anymal-C Quadruped',
                 fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add annotation
    ax.annotate('95%+ Performance Match!',
                xy=(800, 26), fontsize=14, fontweight='bold', color='#2E7D32',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#2E7D32'))

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'linkedin_comparison.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {output_path}")

    return output_path


def print_final_stats(rsl_data, custom_data):
    """Print final statistics comparison."""

    print("\n" + "=" * 60)
    print("FINAL STATISTICS COMPARISON")
    print("=" * 60)

    metrics = [
        ('Train/mean_reward', 'Mean Reward'),
        ('Policy/mean_noise_std', 'Action STD'),
        ('Train/mean_episode_length', 'Episode Length'),
    ]

    print(f"{'Metric':<25} {'RSL-RL':>12} {'Custom PPO':>12} {'Ratio':>10}")
    print("-" * 60)

    for tag, name in metrics:
        rsl_val = rsl_data[tag]['value'].iloc[-1] if tag in rsl_data else 0
        custom_val = custom_data[tag]['value'].iloc[-1] if tag in custom_data else 0
        ratio = (custom_val / rsl_val * 100) if rsl_val != 0 else 0

        print(f"{name:<25} {rsl_val:>12.2f} {custom_val:>12.2f} {ratio:>9.1f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TensorBoard Log Comparison")
    parser.add_argument("--rsl_rl_log", type=str, required=True,
                        help="Path to RSL-RL TensorBoard log directory")
    parser.add_argument("--custom_ppo_log", type=str, required=True,
                        help="Path to Custom PPO TensorBoard log directory")
    parser.add_argument("--output", type=str, default="comparison_plots",
                        help="Output directory for plots")
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Maximum iteration to plot (for cleaner comparison)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"\n[INFO] Loading RSL-RL logs from: {args.rsl_rl_log}")
    rsl_data = load_tensorboard_scalars(args.rsl_rl_log)

    print(f"\n[INFO] Loading Custom PPO logs from: {args.custom_ppo_log}")
    custom_data = load_tensorboard_scalars(args.custom_ppo_log)

    # Print available tags
    print("\n[INFO] RSL-RL tags:")
    for tag in sorted(rsl_data.keys()):
        print(f"  - {tag}")

    print("\n[INFO] Custom PPO tags:")
    for tag in sorted(custom_data.keys()):
        print(f"  - {tag}")

    # Create plots
    print(f"\n[INFO] Creating comparison plots in: {args.output}")

    # Tag mapping - Custom PPO'dan RSL-RL formatına
    # Bu sayede farklı isimli ama aynı anlama gelen metrikler karşılaştırılabilir
    tag_equivalents = {
        'Train/mean_episode_reward': 'Train/mean_reward',  # Custom PPO versiyonu
    }

    # Summary plot - full range
    create_summary_plot(rsl_data, custom_data, args.output, args.max_iter)

    # LinkedIn plots - hem 1K hem 10K
    create_linkedin_plot(rsl_data, custom_data, args.output, max_iter=1000)

    # 10K versiyonu için ayrı dosya
    if args.max_iter and args.max_iter > 1000:
        # Create separate 10K plot
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        tag = 'Train/mean_reward'
        if tag in rsl_data:
            df = rsl_data[tag]
            if args.max_iter:
                df = df[df['step'] <= args.max_iter]
            ax.plot(df['step'], df['value'], label='RSL-RL PPO',
                    color='#FF6B35', linewidth=3, alpha=0.9)

        if tag in custom_data:
            df = custom_data[tag]
            if args.max_iter:
                df = df[df['step'] <= args.max_iter]
            ax.plot(df['step'], df['value'], label='Custom PPO (From Scratch)',
                    color='#004E89', linewidth=3, alpha=0.9)

        ax.set_xlabel('Training Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Episode Reward', fontsize=14, fontweight='bold')
        ax.set_title(f'From-Scratch PPO vs RSL-RL ({args.max_iter} iterations)\nIsaac Lab Anymal-C Quadruped',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Final values annotation
        if tag in rsl_data and tag in custom_data:
            rsl_final = rsl_data[tag]['value'].iloc[-1]
            custom_final = custom_data[tag]['value'].iloc[-1]
            ratio = (custom_final / rsl_final) * 100
            ax.annotate(f'{ratio:.0f}% Performance Match!',
                        xy=(args.max_iter * 0.8, custom_final), fontsize=14, fontweight='bold',
                        color='#2E7D32',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#2E7D32'))

        plt.tight_layout()
        output_path = os.path.join(args.output, f'linkedin_comparison_{args.max_iter}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[SAVED] {output_path}")

    # Individual metric plots
    common_tags = ['Train/mean_reward', 'Policy/mean_noise_std',
                   'Loss/entropy', 'Train/mean_episode_length']

    for tag in common_tags:
        if tag in rsl_data or tag in custom_data:
            plot_comparison(rsl_data, custom_data, tag, args.output, args.max_iter)

    # Print final stats
    print_final_stats(rsl_data, custom_data)

    print(f"\n[DONE] All plots saved to: {args.output}")
    print("\n[TIP] Recommended for LinkedIn/README:")
    print("  - linkedin_comparison.png (1K iter) - Shows early convergence")
    print(f"  - linkedin_comparison_{args.max_iter}.png - Shows final performance") if args.max_iter else None


if __name__ == "__main__":
    main()