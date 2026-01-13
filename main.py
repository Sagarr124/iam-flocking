#!/usr/bin/env python3
"""
Starling Flocking Simulation

A simulation of collective animal behavior using topological interaction rules,
based on Ballerini et al. (2008) - "Interaction ruling animal collective behavior
depends on topological rather than metric distance."

Usage:
    python main.py                      # Run 2D visualization
    python main.py --interactive        # Run 2D interactive mode with sliders
    python main.py --3d                 # Run 3D visualization
    python main.py --3d --interactive   # Run 3D interactive mode with sliders

For parameter sweeps and experiments:
    python experiments/sweep.py         # Visibility threshold sweep
    python experiments/sweep_density.py # Density change experiment
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Starling Flocking Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode with parameter sliders"
    )
    parser.add_argument(
        "--3d",
        dest="three_d",
        action="store_true",
        help="Run in 3D mode"
    )
    
    args = parser.parse_args()
    
    if args.three_d:
        if args.interactive:
            from experiments.run_interactive_3d import main as run
        else:
            from experiments.run_visual_3d import main as run
    else:
        if args.interactive:
            from experiments.run_interactive import main as run
        else:
            from experiments.run_visual import main as run
    
    run()


if __name__ == "__main__":
    main()

