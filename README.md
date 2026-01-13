# Starling Flocking Simulation

A simulation of collective animal behavior using **topological interaction rules**, based on:

> Ballerini et al. (2008) - *"Interaction ruling animal collective behavior depends on topological rather than metric distance"*

## Key Concept

Unlike traditional flocking models that use a fixed metric distance (Reynolds, 1987), this simulation implements **topological rules** where each agent interacts with a fixed number of nearest neighbors (Nc) regardless of their distance. This approach better matches empirical observations of starling murmurations and provides robustness to density changes.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, NumPy, SciPy, Matplotlib, tqdm

## Usage

### Visualizations

```bash
# 2D visualization
python main.py

# 2D interactive mode (with parameter sliders)
python main.py --interactive

# 3D visualization
python main.py --3d

# 3D interactive mode
python main.py --3d --interactive
```

### Experiments

```bash
# Parameter sweep: visibility threshold vs cohesion/order
python experiments/sweep.py

# Density change experiment: test robustness during flock expansion
python experiments/sweep_density.py
```

## Project Structure

```
├── main.py                 # Entry point
├── src/
│   ├── config.py           # Simulation parameters
│   ├── core/
│   │   ├── flock.py        # 2D flocking model
│   │   └── flock3d.py      # 3D flocking model
│   ├── analysis/
│   │   ├── metrics.py      # Order parameter, fragmentation metrics
│   │   └── correlations.py # Spatial correlation analysis
│   └── vis/
│       ├── visualizer.py       # 2D animation
│       ├── visualizer3d.py     # 3D animation
│       ├── interactive.py      # 2D with sliders
│       └── interactive3d.py    # 3D with sliders
├── experiments/
│   ├── sweep.py            # Visibility parameter sweep
│   └── sweep_density.py    # Density change robustness test
└── results/                # Generated plots and data (gitignored)
```

## Parameters

Key parameters in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_AGENTS` | 400 | Number of agents |
| `NC` | 7 | Topological neighbors (key parameter from Ballerini et al.) |
| `MAX_SENSING_RANGE` | 50 | Maximum visibility distance |
| `BLIND_SPOT_ANGLE_DEG` | 60 | Blind spot behind each agent (degrees) |
| `ALIGNMENT_WEIGHT` | 1.0 | Weight for velocity alignment |
| `COHESION_WEIGHT` | 1.0 | Weight for moving toward neighbors |
| `SEPARATION_WEIGHT` | 1.5 | Weight for avoiding collisions |

## Metrics

The simulation tracks:

- **Order Parameter (φ)**: Measures velocity alignment (0 = disordered, 1 = perfectly aligned)
- **Cohesion**: Fraction of agents in the largest connected cluster
- **Fragmentation**: Number of disconnected clusters

## References

- Ballerini, M., et al. (2008). Interaction ruling animal collective behavior depends on topological rather than metric distance. *PNAS*, 105(4), 1232-1237.
- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. *SIGGRAPH*.

