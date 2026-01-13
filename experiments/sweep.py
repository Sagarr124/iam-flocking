"""
Parameter Sweep: Visibility vs Fragmentation Analysis

This module performs a systematic sweep of metric sensing range values to find
the critical threshold where topological flocking rules begin to fail.

Based on Ballerini et al. topological interaction rules with metric sensing limits.
"""

import csv
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.metrics import calculate_fragmentation, calculate_order_parameter
from src.config import Config
from src.core.flock import Flock


def find_critical_threshold(sensing_ranges, cluster_sizes, threshold=0.5):
    """
    Find the critical visibility threshold where cohesion drops below a threshold.
    Uses linear interpolation between data points.

    Args:
        sensing_ranges: Array of tested visibility values
        cluster_sizes: Corresponding normalized cluster sizes
        threshold: Cohesion threshold (default 0.5 = 50% of agents in largest cluster)

    Returns:
        Critical visibility value, or None if not found
    """
    cluster_sizes = np.array(cluster_sizes)

    # Find where cluster size crosses the threshold
    for i in range(len(cluster_sizes) - 1):
        if cluster_sizes[i] < threshold <= cluster_sizes[i + 1]:
            # Linear interpolation
            r1, r2 = sensing_ranges[i], sensing_ranges[i + 1]
            c1, c2 = cluster_sizes[i], cluster_sizes[i + 1]
            critical = r1 + (threshold - c1) * (r2 - r1) / (c2 - c1)
            return critical

    # Check if always above or always below threshold
    if np.all(cluster_sizes >= threshold):
        return sensing_ranges[0]  # Always cohesive
    if np.all(cluster_sizes < threshold):
        return None  # Never cohesive

    return None


def run_sweep(
    n_ranges=25, max_range=60, n_trials=5, warmup_steps=500, measure_steps=100
):
    """
    Run comprehensive parameter sweep with statistical analysis.

    Args:
        n_ranges: Number of sensing range values to test
        max_range: Maximum sensing range to test
        n_trials: Number of trials per sensing range (for statistical robustness)
        warmup_steps: Simulation steps before measuring
        measure_steps: Number of measurement samples (averaged for stability)

    Returns:
        Dictionary containing all results and statistics
    """
    print("=" * 60)
    print("STARLING FLOCKING SIMULATION - PARAMETER SWEEP")
    print("=" * 60)
    print(f"Configuration: N={Config.N_AGENTS}, Nc={Config.NC}")
    print(f"Sweep: {n_ranges} visibility values from 0 to {max_range}")
    print(f"Trials per value: {n_trials}, Warmup: {warmup_steps} steps")
    print("=" * 60)

    sensing_ranges = np.linspace(0, max_range, n_ranges)

    # Store all trial data for statistics
    all_cluster_sizes = []
    all_order_params = []
    all_n_fragments = []

    for r in tqdm(sensing_ranges, desc="Sweeping Sensing Range"):
        trial_clusters = []
        trial_orders = []
        trial_fragments = []

        for trial in range(n_trials):
            flock = Flock()

            # Warmup phase
            for _ in range(warmup_steps):
                flock.update(metric_override=r)

            # Measurement phase - average over multiple samples for stability
            sample_clusters = []
            sample_orders = []
            sample_fragments = []

            for _ in range(measure_steps):
                flock.update(metric_override=r)

                n_comp, largest = calculate_fragmentation(
                    flock.pos, connection_radius=Config.SEPARATION_RADIUS * 3.0
                )
                order = calculate_order_parameter(flock.vel)

                sample_clusters.append(largest / Config.N_AGENTS)
                sample_orders.append(order)
                sample_fragments.append(n_comp)

            # Average over measurement samples
            trial_clusters.append(np.mean(sample_clusters))
            trial_orders.append(np.mean(sample_orders))
            trial_fragments.append(np.mean(sample_fragments))

        all_cluster_sizes.append(trial_clusters)
        all_order_params.append(trial_orders)
        all_n_fragments.append(trial_fragments)

    # Convert to numpy arrays
    all_cluster_sizes = np.array(all_cluster_sizes)
    all_order_params = np.array(all_order_params)
    all_n_fragments = np.array(all_n_fragments)

    # Calculate statistics
    results = {
        "sensing_ranges": sensing_ranges,
        "cluster_mean": np.mean(all_cluster_sizes, axis=1),
        "cluster_std": np.std(all_cluster_sizes, axis=1),
        "order_mean": np.mean(all_order_params, axis=1),
        "order_std": np.std(all_order_params, axis=1),
        "fragments_mean": np.mean(all_n_fragments, axis=1),
        "fragments_std": np.std(all_n_fragments, axis=1),
        "raw_clusters": all_cluster_sizes,
        "raw_orders": all_order_params,
        "raw_fragments": all_n_fragments,
        "config": {
            "N_AGENTS": Config.N_AGENTS,
            "NC": Config.NC,
            "SEPARATION_RADIUS": Config.SEPARATION_RADIUS,
            "BLIND_SPOT_ANGLE": Config.BLIND_SPOT_ANGLE_DEG,
            "n_trials": n_trials,
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
        },
    }

    # Find critical threshold
    critical_50 = find_critical_threshold(sensing_ranges, results["cluster_mean"], 0.5)
    critical_80 = find_critical_threshold(sensing_ranges, results["cluster_mean"], 0.8)
    critical_90 = find_critical_threshold(sensing_ranges, results["cluster_mean"], 0.9)

    results["critical_threshold_50"] = critical_50
    results["critical_threshold_80"] = critical_80
    results["critical_threshold_90"] = critical_90

    return results


def export_results_csv(results, filename="results/sweep_data.csv"):
    """Export results to CSV for external analysis."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Header with metadata
        writer.writerow(["# Starling Flocking Simulation - Parameter Sweep Results"])
        writer.writerow([f"# Generated: {datetime.now().isoformat()}"])
        writer.writerow([f"# N_AGENTS: {results['config']['N_AGENTS']}"])
        writer.writerow([f"# NC (Topological Neighbors): {results['config']['NC']}"])
        writer.writerow(
            [f"# BLIND_SPOT_ANGLE: {results['config']['BLIND_SPOT_ANGLE']}"]
        )
        writer.writerow([f"# Trials per range: {results['config']['n_trials']}"])
        writer.writerow([])

        # Column headers
        writer.writerow(
            [
                "sensing_range",
                "cluster_size_mean",
                "cluster_size_std",
                "order_param_mean",
                "order_param_std",
                "n_fragments_mean",
                "n_fragments_std",
            ]
        )

        # Data rows
        for i, r in enumerate(results["sensing_ranges"]):
            writer.writerow(
                [
                    f"{r:.2f}",
                    f"{results['cluster_mean'][i]:.4f}",
                    f"{results['cluster_std'][i]:.4f}",
                    f"{results['order_mean'][i]:.4f}",
                    f"{results['order_std'][i]:.4f}",
                    f"{results['fragments_mean'][i]:.2f}",
                    f"{results['fragments_std'][i]:.2f}",
                ]
            )

    print(f"Data exported to {filename}")


def plot_results(results, save_path="results/sweep_results.png"):
    """Generate comprehensive visualization with error bars."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sensing_ranges = results["sensing_ranges"]

    # Plot 1: Giant Component Size with error bars
    ax1 = axes[0, 0]
    ax1.errorbar(
        sensing_ranges,
        results["cluster_mean"],
        yerr=results["cluster_std"],
        marker="o",
        color="tab:red",
        capsize=3,
        label="Cluster Size",
    )
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax1.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label="90% threshold")

    # Mark critical thresholds
    if results["critical_threshold_50"] is not None:
        ax1.axvline(
            x=results["critical_threshold_50"],
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"R_crit(50%)={results['critical_threshold_50']:.1f}",
        )
    if results["critical_threshold_90"] is not None:
        ax1.axvline(
            x=results["critical_threshold_90"],
            color="green",
            linestyle=":",
            alpha=0.7,
            label=f"R_crit(90%)={results['critical_threshold_90']:.1f}",
        )

    ax1.set_xlabel("Metric Sensing Range (R_vis)")
    ax1.set_ylabel("Giant Component Size (Normalized)")
    ax1.set_title("Flock Cohesion vs Visibility")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Plot 2: Order Parameter with error bars
    ax2 = axes[0, 1]
    ax2.errorbar(
        sensing_ranges,
        results["order_mean"],
        yerr=results["order_std"],
        marker="s",
        color="tab:blue",
        capsize=3,
        label="Polarization",
    )
    ax2.set_xlabel("Metric Sensing Range (R_vis)")
    ax2.set_ylabel("Order Parameter (φ)")
    ax2.set_title("Flock Alignment vs Visibility")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Plot 3: Number of Fragments
    ax3 = axes[1, 0]
    ax3.errorbar(
        sensing_ranges,
        results["fragments_mean"],
        yerr=results["fragments_std"],
        marker="^",
        color="tab:purple",
        capsize=3,
        label="# Fragments",
    )
    ax3.set_xlabel("Metric Sensing Range (R_vis)")
    ax3.set_ylabel("Number of Separate Groups")
    ax3.set_title("Fragmentation vs Visibility")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Combined view (Phase Diagram style)
    ax4 = axes[1, 1]
    ax4.fill_between(
        sensing_ranges,
        0,
        results["cluster_mean"],
        alpha=0.3,
        color="red",
        label="Cohesion",
    )
    ax4.fill_between(
        sensing_ranges,
        0,
        results["order_mean"],
        alpha=0.3,
        color="blue",
        label="Alignment",
    )
    ax4.plot(sensing_ranges, results["cluster_mean"], "r-", linewidth=2)
    ax4.plot(sensing_ranges, results["order_mean"], "b-", linewidth=2)

    # Mark transition zone
    if results["critical_threshold_50"] and results["critical_threshold_90"]:
        ax4.axvspan(
            results["critical_threshold_50"],
            results["critical_threshold_90"],
            alpha=0.2,
            color="yellow",
            label="Transition Zone",
        )

    ax4.set_xlabel("Metric Sensing Range (R_vis)")
    ax4.set_ylabel("Normalized Value")
    ax4.set_title("Phase Transition Overview")
    ax4.legend(loc="lower right")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)

    # Overall title
    fig.suptitle(
        f"Impact of Metric Sensing Limit on Topological Flocking\n"
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
    """Print summary statistics and findings."""
    print("\n" + "=" * 60)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  - Number of agents: {results['config']['N_AGENTS']}")
    print(f"  - Topological neighbors (Nc): {results['config']['NC']}")
    print(f"  - Blind spot angle: {results['config']['BLIND_SPOT_ANGLE']}°")

    print(f"\nCritical Thresholds (Visibility R_vis):")
    if results["critical_threshold_90"]:
        print(
            f"  - 90% cohesion maintained above: R_vis = {results['critical_threshold_90']:.2f}"
        )
    if results["critical_threshold_80"]:
        print(
            f"  - 80% cohesion maintained above: R_vis = {results['critical_threshold_80']:.2f}"
        )
    if results["critical_threshold_50"]:
        print(
            f"  - 50% cohesion maintained above: R_vis = {results['critical_threshold_50']:.2f}"
        )
    else:
        print("  - WARNING: Cohesion never drops below 50%")

    print(f"\nPeak Values:")
    max_cluster_idx = np.argmax(results["cluster_mean"])
    max_order_idx = np.argmax(results["order_mean"])
    print(
        f"  - Max cluster size: {results['cluster_mean'][max_cluster_idx]:.3f} "
        f"at R_vis = {results['sensing_ranges'][max_cluster_idx]:.1f}"
    )
    print(
        f"  - Max order param: {results['order_mean'][max_order_idx]:.3f} "
        f"at R_vis = {results['sensing_ranges'][max_order_idx]:.1f}"
    )

    # Minimum visibility for stable flock
    stable_mask = results["cluster_mean"] > 0.9
    if np.any(stable_mask):
        min_stable_idx = np.argmax(stable_mask)
        print(f"\nMinimum visibility for stable flock (>90% cohesion):")
        print(f"  R_vis ≥ {results['sensing_ranges'][min_stable_idx]:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    # Run comprehensive sweep
    results = run_sweep(
        n_ranges=25,  # 25 visibility values
        max_range=60,  # Test up to visibility = 60
        n_trials=5,  # 5 trials per value for statistics
        warmup_steps=500,  # Let flock stabilize
        measure_steps=50,  # Average over 50 measurements
    )

    # Generate outputs
    plot_results(results)
    export_results_csv(results)
    print_summary(results)
