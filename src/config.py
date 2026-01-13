import numpy as np

class Config:
    # Simulation Parameters
    N_AGENTS = 400            # Number of birds
    DT = 0.1                  # Time step
    STEPS = 1000              # Total simulation steps
    WIDTH = 100.0             # Arena width (2D)
    HEIGHT = 100.0            # Arena height (2D)
    DEPTH = 100.0             # Arena depth (3D)
    DIMENSIONS = 2            # 2 or 3
    
    # Topological Rule
    NC = 7                    # Number of topological neighbors (Ballerini et al.)
    
    # Metric Sensing Limit (The "Borrowed" Extension)
    # Start with a large efficient standard, then sweep this value.
    MAX_SENSING_RANGE = 50.0  # Max distance to even consider a neighbor
    BLIND_SPOT_ANGLE_DEG = 60 # Degrees behind the bird that are blind
    
    # Physics / Flocking (Steering Behaviors)
    MAX_SPEED = 2.0
    MIN_SPEED = 0.5
    MAX_FORCE = 0.05
    
    # Force Weights
    SEPARATION_WEIGHT = 1.5
    ALIGNMENT_WEIGHT = 1.0
    COHESION_WEIGHT = 1.0
    AVOID_WALL_WEIGHT = 2.0
    NOISE_STRENGTH = 0.05
    
    # Separation needs a metric 'personal space' even in topological models
    SEPARATION_RADIUS = 1.5 
    
    @staticmethod
    def get_blind_cosine():
        return np.cos(np.deg2rad(180 - Config.BLIND_SPOT_ANGLE_DEG/2))

    @staticmethod
    def info():
        return f"Flocking Config: N={Config.N_AGENTS}, Nc={Config.NC}, MetricLimit={Config.MAX_SENSING_RANGE}"
