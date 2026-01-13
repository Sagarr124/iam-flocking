"""
Parameter Sweep with Correlation Analysis

This module extends the basic sweep to include velocity correlation analysis,
which measures information transfer through the flock (as per Ballerini et al.).

The correlation length ξ represents how far information (velocity fluctuations)
propagates through the group - a key indicator of collective behavior.
"""

import csv
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.correlations import (
    calculate_correlation_length,
    calculate_velocity_correlation,
)
from src.analysis.metrics import calculate_fragmentation, calculate_order_parameter
from src.config import Config
from src.core.flock import Flock


def run_sweep(n_ranges=25, max_range=60, n_trials=5, warmup_steps=400):
    """
    Run parameter sweep with correlation length analysis.

    The correlation length ξ is particularly important because:
    - High ξ indicates strong collective behavior (information travels far)
    - Low ξ indicates poor coordination (local interactions only)
    - The relationship between ξ and visibility reveals how metric limits
      affect the topological rule's effectiveness
    """
    print("=" * 60)
    print("CORRELATION LENGTH ANALYSIS - PARAMETER SWEEP")
    print("=" * 60)
    print(f"Configuration: N={Config.N_AGENTS}, Nc={Config.NC}")
    print(f"Sweep: {n_ranges} visibility values from 0 to {max_range}")
    print("=" * 60)

    sensing_ranges = np.linspace(0, max_range, n_ranges)

    all_cluster_sizes = []
    all_order_params = []
    all_corr_lengths = []

    for r in tqdm(sensing_ranges, desc="Sweeping Sensing Range"):
        trial_clusters = []
        trial_orders = []
        trial_corrs = []

        for trial in range(n_trials):
            flock = Flock()

            # Warmup phase
            for _ in range(warmup_steps):
                flock.update(metric_override=r)

            # Measure
            n_comp, largest = calculate_fragmentation(
                flock.pos, connection_radius=Config.SEPARATION_RADIUS * 3.0
            )
            order = calculate_order_parameter(flock.vel)

            # Correlation analysis
            rs, Cr = calculate_velocity_correlation(
                flock.pos, flock.vel, dr=1.0, max_r=Config.WIDTH / 2
            )
            xi = calculate_correlation_length(rs, Cr)

            trial_clusters.append(largest / Config.N_AGENTS)
            trial_orders.append(order)
            trial_corrs.append(xi)

        all_cluster_sizes.append(trial_clusters)
        all_order_params.append(trial_orders)
        all_corr_lengths.append(trial_corrs)

    # Convert to arrays and compute statistics
    all_cluster_sizes = np.array(all_cluster_sizes)
    all_order_params = np.array(all_order_params)
    all_corr_lengths = np.array(all_corr_lengths)

    results = {
        "sensing_ranges": sensing_ranges,
        "cluster_mean": np.mean(all_cluster_sizes, axis=1),
        "cluster_std": np.std(all_cluster_sizes, axis=1),
        "order_mean": np.mean(all_order_params, axis=1),
        "order_std": np.std(all_order_params, axis=1),
        "corr_mean": np.mean(all_corr_lengths, axis=1),
        "corr_std": np.std(all_corr_lengths, axis=1),
        "config": {
            "N_AGENTS": Config.N_AGENTS,
            "NC": Config.NC,
            "BLIND_SPOT_ANGLE": Config.BLIND_SPOT_ANGLE_DEG,
            "n_trials": n_trials,
        },
    }

    return results


def export_results_csv(results, filename="results/sweep_corr_data.csv"):
    """Export correlation sweep results to CSV."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# Correlation Length Sweep Results"])
        writer.writerow([f"# Generated: {datetime.now().isoformat()}"])
        writer.writerow([f"# N_AGENTS: {results['config']['N_AGENTS']}"])
        writer.writerow([f"# NC: {results['config']['NC']}"])
        writer.writerow([])

        writer.writerow(
            [
                "sensing_range",
                "cluster_mean",
                "cluster_std",
                "order_mean",
                "order_std",
                "correlation_length_mean",
                "correlation_length_std",
            ]
        )

        for i, r in enumerate(results["sensing_ranges"]):
            writer.writerow(
                [
                    f"{r:.2f}",
                    f"{results['cluster_mean'][i]:.4f}",
                    f"{results['cluster_std'][i]:.4f}",
                    f"{results['order_mean'][i]:.4f}",
                    f"{results['order_std'][i]:.4f}",
                    f"{results['corr_mean'][i]:.2f}",
                    f"{results['corr_std'][i]:.2f}",
                ]
            )

    print(f"Data exported to {filename}")


def plot_results(results, save_path="results/sweep_results_corr.png"):
    """Generate visualization with correlation length analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sensing_ranges = results["sensing_ranges"]

    # Plot 1: Cohesion & Alignment
    ax1 = axes[0]
    color1 = "tab:red"
    ax1.errorbar(
        sensing_ranges,
        results["cluster_mean"],
        yerr=results["cluster_std"],
        marker="o",
        color=color1,
        capsize=3,
        label="Cluster Size",
    )
    ax1.set_xlabel("Metric Sensing Range (R_vis)")
    ax1.set_ylabel("Giant Component Size", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax1b = ax1.twinx()
    color2 = "tab:blue"
    ax1b.errorbar(
        sensing_ranges,
        results["order_mean"],
        yerr=results["order_std"],
        marker="s",
        linestyle="--",
        color=color2,
        capsize=3,
        label="Order",
    )
    ax1b.set_ylabel("Order Parameter", color=color2)
    ax1b.tick_params(axis="y", labelcolor=color2)
    ax1b.set_ylim(0, 1.05)
    ax1.set_title("Cohesion & Alignment")

    # Plot 2: Correlation Length
    ax2 = axes[1]
    ax2.errorbar(
        sensing_ranges,
        results["corr_mean"],
        yerr=results["corr_std"],
        marker="^",
        color="purple",
        capsize=3,
    )
    ax2.set_xlabel("Metric Sensing Range (R_vis)")
    ax2.set_ylabel("Correlation Length (ξ)")
    ax2.set_title("Information Transfer Range")
    ax2.grid(True, alpha=0.3)

    # Add reference line for system size
    ax2.axhline(
        y=Config.WIDTH / 2,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"System size / 2 = {Config.WIDTH / 2}",
    )
    ax2.legend()

    # Plot 3: Normalized Correlation (ξ / L)
    ax3 = axes[2]
    normalized_corr = results["corr_mean"] / Config.WIDTH
    normalized_std = results["corr_std"] / Config.WIDTH

    ax3.errorbar(
        sensing_ranges,
        normalized_corr,
        yerr=normalized_std,
        marker="d",
        color="darkgreen",
        capsize=3,
    )
    ax3.set_xlabel("Metric Sensing Range (R_vis)")
    ax3.set_ylabel("ξ / L (Normalized Correlation Length)")
    ax3.set_title("Scale-Free Correlation Measure")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Velocity Correlation Analysis: Metric Sensing vs Information Transfer\n"
        f"(N={Config.N_AGENTS}, Nc={Config.NC}, Blind Spot={Config.BLIND_SPOT_ANGLE_DEG}°)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

    return fig


def print_summary(results):
    """Print correlation analysis summary."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("=" * 60)

    # Find peak correlation length
    max_corr_idx = np.argmax(results["corr_mean"])
    print(f"\nPeak Correlation Length:")
    print(
        f"  ξ_max = {results['corr_mean'][max_corr_idx]:.2f} "
        f"at R_vis = {results['sensing_ranges'][max_corr_idx]:.1f}"
    )
    print(
        f"  (Normalized: ξ/L = {results['corr_mean'][max_corr_idx] / Config.WIDTH:.3f})"
    )

    # Find visibility where correlation starts dropping
    high_corr = results["corr_mean"] > 0.8 * results["corr_mean"][max_corr_idx]
    if np.any(high_corr):
        first_high = np.argmax(high_corr)
        print(f"\nMinimum visibility for strong correlation (>80% of max):")
        print(f"  R_vis ≥ {results['sensing_ranges'][first_high]:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    results = run_sweep(n_ranges=25, max_range=60, n_trials=5, warmup_steps=400)

    plot_results(results)
    export_results_csv(results)
    print_summary(results)
