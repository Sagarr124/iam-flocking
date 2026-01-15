"""
Density Change Experiment: Testing Topological Rule Robustness

This experiment tests how topological flocking rules perform during
density changes (expansion/contraction events). This is key to understanding
the biological fitness of topological rules - can the flock maintain
cohesion when density fluctuates?

Based on Ballerini et al.: topological rules should be robust to density
changes, unlike metric rules which fail when the flock expands.
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


def run_density_experiment(
    visibility_values=[10, 20, 30, 50],
    expansion_factor=2.0,
    n_trials=5,
    warmup_steps=300,
    expansion_steps=200,
    measure_interval=10,
):
    """
    Test flock cohesion during a density change event.

    The experiment:
    1. Start with a cohesive flock in normal arena
    2. Suddenly expand the arena (simulating flock expansion/spreading)
    3. Monitor how quickly/if the flock fragments

    This tests the key prediction of Ballerini et al.: topological rules
    should handle density changes better than pure metric rules.
    """
    print("=" * 60)
    print("DENSITY CHANGE EXPERIMENT")
    print("=" * 60)
    print(f"Testing visibility values: {visibility_values}")
    print(f"Expansion factor: {expansion_factor}x")
    print("=" * 60)

    results = {
        "visibility_values": visibility_values,
        "expansion_factor": expansion_factor,
        "time_series": {},
    }

    for vis in visibility_values:
        print(f"\nTesting visibility R_vis = {vis}")

        all_cohesion_before = []
        all_cohesion_after = []
        all_time_series = []

        for trial in range(n_trials):
            flock = Flock()

            # Phase 1: Warmup - establish cohesive flock
            for _ in range(warmup_steps):
                flock.update(metric_override=vis)

            # Measure pre-expansion
            _, largest = calculate_fragmentation(
                flock.pos, connection_radius=Config.SEPARATION_RADIUS * 3.0
            )
            cohesion_before = largest / Config.N_AGENTS
            all_cohesion_before.append(cohesion_before)

            # Phase 2: Expansion event - scale positions outward
            # This simulates the flock "spreading out" by expanding the arena
            # and scaling agent positions relative to center of mass
            original_width = flock.width
            original_height = flock.height

            # Expand arena first
            flock.width *= expansion_factor
            flock.height *= expansion_factor

            # Scale positions relative to center of mass, then shift to new arena center
            com = np.mean(flock.pos, axis=0)
            new_center = np.array([flock.width / 2, flock.height / 2])
            flock.pos = new_center + (flock.pos - com) * expansion_factor

            # Ensure all positions are within bounds (wrap if needed)
            flock.pos[:, 0] = np.mod(flock.pos[:, 0], flock.width)
            flock.pos[:, 1] = np.mod(flock.pos[:, 1], flock.height)

            # Track recovery over time
            time_series = []

            for step in range(expansion_steps):
                flock.update(metric_override=vis)

                if step % measure_interval == 0:
                    _, largest = calculate_fragmentation(
                        flock.pos,
                        connection_radius=Config.SEPARATION_RADIUS
                        * 3.0
                        * expansion_factor,
                    )
                    cohesion = largest / Config.N_AGENTS
                    order = calculate_order_parameter(flock.vel)
                    time_series.append(
                        {"step": step, "cohesion": cohesion, "order": order}
                    )

            # Final measurement
            _, largest = calculate_fragmentation(
                flock.pos,
                connection_radius=Config.SEPARATION_RADIUS * 3.0 * expansion_factor,
            )
            cohesion_after = largest / Config.N_AGENTS
            all_cohesion_after.append(cohesion_after)
            all_time_series.append(time_series)

            # Reset arena for next trial
            flock.width = original_width
            flock.height = original_height

        results["time_series"][vis] = {
            "cohesion_before": np.mean(all_cohesion_before),
            "cohesion_after_mean": np.mean(all_cohesion_after),
            "cohesion_after_std": np.std(all_cohesion_after),
            "recovery_ratio": np.mean(all_cohesion_after)
            / np.mean(all_cohesion_before),
            "series": all_time_series,
        }

        print(
            f"  Before: {np.mean(all_cohesion_before):.3f}, "
            f"After: {np.mean(all_cohesion_after):.3f} ± {np.std(all_cohesion_after):.3f}"
        )

    return results


def plot_density_results(results, save_path="results/density_experiment.svg"):
    """Plot density change experiment results."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Libertinus Serif']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    visibility_values = results["visibility_values"]

    # Plot 1: Before/After comparison
    ax1 = axes[0]
    before_vals = [
        results["time_series"][v]["cohesion_before"] for v in visibility_values
    ]
    after_vals = [
        results["time_series"][v]["cohesion_after_mean"] for v in visibility_values
    ]
    after_stds = [
        results["time_series"][v]["cohesion_after_std"] for v in visibility_values
    ]

    x = np.arange(len(visibility_values))
    width = 0.35

    ax1.bar(
        x - width / 2, before_vals, width, label="Before Expansion", color="steelblue"
    )
    ax1.bar(
        x + width / 2,
        after_vals,
        width,
        yerr=after_stds,
        label="After Expansion",
        color="coral",
        capsize=5,
    )

    ax1.set_xlabel("Visibility (R_vis)")
    ax1.set_ylabel("Cohesion (Largest Cluster / N)")
    ax1.set_title(
        f"Flock Cohesion: Before vs After {results['expansion_factor']}x Expansion"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(visibility_values)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Recovery time series (average across trials)
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(visibility_values)))

    for i, vis in enumerate(visibility_values):
        series_list = results["time_series"][vis]["series"]
        # Average across trials
        steps = [s["step"] for s in series_list[0]]
        avg_cohesion = []
        for step_idx in range(len(steps)):
            vals = [
                series_list[t][step_idx]["cohesion"] for t in range(len(series_list))
            ]
            avg_cohesion.append(np.mean(vals))

        ax2.plot(
            steps,
            avg_cohesion,
            marker="o",
            color=colors[i],
            label=f"R_vis = {vis}",
            markersize=4,
        )

    ax2.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax2.set_xlabel("Steps After Expansion")
    ax2.set_ylabel("Cohesion")
    ax2.set_title("Cohesion Recovery After Density Change")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(
        f"Density Change Experiment: Topological Rule Robustness\n"
        f"(N={Config.N_AGENTS}, Nc={Config.NC}, Expansion={results['expansion_factor']}x)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"Plot saved to {save_path}")

    return fig


def export_density_csv(results, filename="results/density_data.csv"):
    """Export density experiment results to CSV."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# Density Change Experiment Results"])
        writer.writerow([f"# Generated: {datetime.now().isoformat()}"])
        writer.writerow([f"# Expansion Factor: {results['expansion_factor']}"])
        writer.writerow([])

        writer.writerow(
            [
                "visibility",
                "cohesion_before",
                "cohesion_after_mean",
                "cohesion_after_std",
                "recovery_ratio",
            ]
        )

        for vis in results["visibility_values"]:
            data = results["time_series"][vis]
            writer.writerow(
                [
                    vis,
                    f"{data['cohesion_before']:.4f}",
                    f"{data['cohesion_after_mean']:.4f}",
                    f"{data['cohesion_after_std']:.4f}",
                    f"{data['recovery_ratio']:.4f}",
                ]
            )

    print(f"Data exported to {filename}")


def print_density_summary(results):
    """Print summary of density experiment."""
    print("\n" + "=" * 60)
    print("DENSITY EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\nExpansion Factor: {results['expansion_factor']}x")
    print(f"\nRecovery Performance by Visibility:")

    for vis in results["visibility_values"]:
        data = results["time_series"][vis]
        print(f"\n  R_vis = {vis}:")
        print(f"    Pre-expansion cohesion: {data['cohesion_before']:.3f}")
        print(
            f"    Post-expansion cohesion: {data['cohesion_after_mean']:.3f} ± {data['cohesion_after_std']:.3f}"
        )
        print(f"    Recovery ratio: {data['recovery_ratio']:.3f}")

        if data["recovery_ratio"] > 0.9:
            print(f"    Status: ROBUST [OK]")
        elif data["recovery_ratio"] > 0.5:
            print(f"    Status: PARTIAL RECOVERY")
        else:
            print(f"    Status: FRAGMENTED [FAIL]")

    print("=" * 60)


if __name__ == "__main__":
    results = run_density_experiment(
        visibility_values=[5, 10, 15, 20, 30, 50],
        expansion_factor=2.0,
        n_trials=5,
        warmup_steps=300,
        expansion_steps=200,
        measure_interval=10,
    )

    plot_density_results(results)
    export_density_csv(results)
    print_density_summary(results)
