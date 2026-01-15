"""Direct Baseline Comparison: Topological vs Metric Flocking

This experiment compares the existing topological neighbor rule against a
metric-only baseline (radius-based neighbors) under the same conditions.

It reports cohesion before and after a density change (arena expansion) for
both modes.
"""

import csv
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.metrics import calculate_fragmentation
from src.config import Config
from src.core.flock import Flock


def run_comparison(
    visibility_values,
    expansion_factor=2.0,
    n_trials=5,
    warmup_steps=300,
    expansion_steps=200,
    measure_interval=10,
    seed=0,
):
    results = {
        "visibility_values": list(visibility_values),
        "expansion_factor": expansion_factor,
        "n_trials": n_trials,
        "warmup_steps": warmup_steps,
        "expansion_steps": expansion_steps,
        "measure_interval": measure_interval,
        "seed": seed,
        "modes": ["topological", "metric"],
        "data": {},
    }

    for vis_idx, vis in enumerate(tqdm(visibility_values, desc="Baseline comparison (R_vis)")):
        results["data"][vis] = {}
        for mode in results["modes"]:
            cohesion_before = []
            cohesion_after = []
            recovery_ratios = []

            for trial in range(n_trials):
                # Make the two modes comparable by reusing the same RNG seed
                # per trial (same initialization + same noise sequence).
                np.random.seed(seed + trial + 10000 * vis_idx)
                flock = Flock()

                for _ in range(warmup_steps):
                    flock.update(metric_override=vis, mode=mode)

                _, largest = calculate_fragmentation(
                    flock.pos, connection_radius=Config.SEPARATION_RADIUS * 3.0
                )
                cohesion_before.append(largest / Config.N_AGENTS)

                # Expansion event
                original_width = flock.width
                original_height = flock.height

                flock.width *= expansion_factor
                flock.height *= expansion_factor

                com = np.mean(flock.pos, axis=0)
                new_center = np.array([flock.width / 2, flock.height / 2])
                flock.pos = new_center + (flock.pos - com) * expansion_factor

                flock.pos[:, 0] = np.mod(flock.pos[:, 0], flock.width)
                flock.pos[:, 1] = np.mod(flock.pos[:, 1], flock.height)

                for step in range(expansion_steps):
                    flock.update(metric_override=vis, mode=mode)

                    # keep the same measurement schedule as sweep_density.py
                    if step % measure_interval == 0:
                        pass

                _, largest = calculate_fragmentation(
                    flock.pos,
                    connection_radius=Config.SEPARATION_RADIUS
                    * 3.0
                    * expansion_factor,
                )
                after = largest / Config.N_AGENTS
                cohesion_after.append(after)

                before = cohesion_before[-1]
                if before > 0:
                    recovery_ratios.append(after / before)

                flock.width = original_width
                flock.height = original_height

            before_mean = float(np.mean(cohesion_before))
            before_std = float(np.std(cohesion_before))
            after_mean = float(np.mean(cohesion_after))
            after_std = float(np.std(cohesion_after))
            recovery_ratio_mean = float(np.mean(recovery_ratios)) if recovery_ratios else float("nan")
            recovery_ratio_std = float(np.std(recovery_ratios)) if recovery_ratios else float("nan")

            results["data"][vis][mode] = {
                "cohesion_before_mean": before_mean,
                "cohesion_before_std": before_std,
                "cohesion_after_mean": after_mean,
                "cohesion_after_std": after_std,
                "recovery_ratio_mean": recovery_ratio_mean,
                "recovery_ratio_std": recovery_ratio_std,
            }

    return results


def export_csv(results, filename="data/baseline_topo_vs_metric.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# Topological vs Metric Baseline Comparison"])
        writer.writerow([f"# Generated: {datetime.now().isoformat()}"])
        writer.writerow([f"# N_AGENTS: {Config.N_AGENTS}"])
        writer.writerow([f"# NC: {Config.NC}"])
        writer.writerow([f"# BLIND_SPOT_ANGLE_DEG: {Config.BLIND_SPOT_ANGLE_DEG}"])
        writer.writerow([f"# SEED: {results['seed']}"])
        writer.writerow([f"# TRIALS: {results['n_trials']}"])
        writer.writerow([f"# EXPANSION_FACTOR: {results['expansion_factor']}"])
        writer.writerow([])

        writer.writerow(
            [
                "R_vis",
                "mode",
                "cohesion_before_mean",
                "cohesion_before_std",
                "cohesion_after_mean",
                "cohesion_after_std",
                "recovery_ratio_mean",
                "recovery_ratio_std",
            ]
        )

        for vis in results["visibility_values"]:
            for mode in results["modes"]:
                d = results["data"][vis][mode]
                writer.writerow(
                    [
                        vis,
                        mode,
                        f"{d['cohesion_before_mean']:.6f}",
                        f"{d['cohesion_before_std']:.6f}",
                        f"{d['cohesion_after_mean']:.6f}",
                        f"{d['cohesion_after_std']:.6f}",
                        f"{d['recovery_ratio_mean']:.6f}",
                        f"{d['recovery_ratio_std']:.6f}",
                    ]
                )


def plot_results(results, save_path="figures/baseline_topo_vs_metric.svg"):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Libertinus Serif"]

    vis_values = np.array(results["visibility_values"], dtype=float)

    topo_rr = np.array(
        [results["data"][v]["topological"]["recovery_ratio_mean"] for v in vis_values]
    )
    topo_rr_std = np.array(
        [results["data"][v]["topological"]["recovery_ratio_std"] for v in vis_values]
    )
    metric_rr = np.array(
        [results["data"][v]["metric"]["recovery_ratio_mean"] for v in vis_values]
    )
    metric_rr_std = np.array(
        [results["data"][v]["metric"]["recovery_ratio_std"] for v in vis_values]
    )

    topo_after = np.array(
        [results["data"][v]["topological"]["cohesion_after_mean"] for v in vis_values]
    )
    topo_after_std = np.array(
        [results["data"][v]["topological"]["cohesion_after_std"] for v in vis_values]
    )
    metric_after = np.array(
        [results["data"][v]["metric"]["cohesion_after_mean"] for v in vis_values]
    )
    metric_after_std = np.array(
        [results["data"][v]["metric"]["cohesion_after_std"] for v in vis_values]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.errorbar(
        vis_values,
        topo_rr,
        yerr=topo_rr_std,
        marker="o",
        capsize=3,
        label="Topological (Nc fixed)",
    )
    ax1.errorbar(
        vis_values,
        metric_rr,
        yerr=metric_rr_std,
        marker="s",
        capsize=3,
        label="Metric-only (radius)",
    )
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Visibility / Interaction Radius (R_vis)")
    ax1.set_ylabel("Recovery Ratio (post / pre cohesion)")
    ax1.set_title("Density Change Robustness: Recovery Ratio")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    ax2.errorbar(
        vis_values,
        topo_after,
        yerr=topo_after_std,
        marker="o",
        capsize=3,
        label="Topological (Nc fixed)",
    )
    ax2.errorbar(
        vis_values,
        metric_after,
        yerr=metric_after_std,
        marker="s",
        capsize=3,
        label="Metric-only (radius)",
    )
    ax2.set_xlabel("Visibility / Interaction Radius (R_vis)")
    ax2.set_ylabel("Post-expansion cohesion")
    ax2.set_title("Post-expansion Cohesion")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(
        f"Topological vs Metric Baseline (N={Config.N_AGENTS}, Expansion={results['expansion_factor']}x)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    visibility_values = [5, 10, 15, 20, 30, 50]

    results = run_comparison(
        visibility_values=visibility_values,
        expansion_factor=2.0,
        n_trials=5,
        warmup_steps=300,
        expansion_steps=200,
        measure_interval=10,
        seed=0,
    )

    export_csv(results)
    plot_results(results)
