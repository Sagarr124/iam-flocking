import numpy as np
from scipy.spatial import KDTree

from src.config import Config


class Flock:
    def __init__(self):
        self.N = Config.N_AGENTS
        self.width = Config.WIDTH
        self.height = Config.HEIGHT

        # State: Position, Velocity, Acceleration
        # Random initial positions
        self.pos = np.random.rand(self.N, 2) * np.array([self.width, self.height])

        # Random initial velocities with speed clamping
        angles = np.random.rand(self.N) * 2 * np.pi
        speeds = (
            np.random.rand(self.N) * (Config.MAX_SPEED - Config.MIN_SPEED)
        ) + Config.MIN_SPEED
        self.vel = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, None]
        self.acc = np.zeros((self.N, 2))

    def update(self, metric_override=None):
        """
        One simulation step object.
        Optimized vectorized implementation.
        """
        self.acc = np.zeros((self.N, 2))

        sensing_radius = (
            metric_override if metric_override is not None else Config.MAX_SENSING_RANGE
        )
        nc = Config.NC

        # 1. Neighbor Search
        # tree.query returns (N, k). self.pos is (N, 2)
        tree = KDTree(self.pos, boxsize=[self.width, self.height])

        # We query slightly more than Nc to account for metric cutoff
        # Let's grab Nc + 10 candidates
        k_query = nc + 10
        dists, idxs = tree.query(self.pos, k=k_query)

        # dists: (N, k), idxs: (N, k)

        # 2. Relative Vectors (Vectorized)
        # We need dx for all pairs in idxs.
        # idxs contains indices of neighbors.
        # Advanced indexing: self.pos[idxs] gives shape (N, k, 2)
        neighbor_pos = self.pos[idxs]  # (N, k, 2)

        # Self pos needs to be broadcast: (N, 1, 2)
        my_pos = self.pos[:, np.newaxis, :]

        dx = neighbor_pos - my_pos  # (N, k, 2)

        # Periodic Wrap (Vectorized)
        box = np.array([self.width, self.height])
        dx -= np.round(dx / box) * box

        # 3. Masks
        # Exclude self (dist ~ 0). Since k includes self usually at index 0?
        # KDTree list usually puts self at index 0 if present.
        # But separate masks are safer.
        valid_mask = dists > 1e-6  # (N, k)

        # Metric Visibility Mask
        vis_mask = (dists < sensing_radius) & valid_mask  # (N, k)

        # Blind Spot: Agents cannot see directly behind them
        # Check if neighbors are in the blind cone behind the agent
        if Config.BLIND_SPOT_ANGLE_DEG > 0:
            # Normalize velocity to get heading direction
            vel_norm = np.linalg.norm(self.vel, axis=1, keepdims=True)
            vel_norm[vel_norm < 1e-6] = 1.0  # Avoid div by zero
            heading = self.vel / vel_norm  # (N, 2)

            # Direction to each neighbor (normalized dx)
            dx_norm = np.linalg.norm(dx, axis=2, keepdims=True)
            dx_norm[dx_norm < 1e-6] = 1.0
            neighbor_dir = dx / dx_norm  # (N, k, 2)

            # Dot product: heading · neighbor_direction
            # heading: (N, 2) -> (N, 1, 2) for broadcasting
            heading_expanded = heading[:, np.newaxis, :]
            cos_angle = np.sum(heading_expanded * neighbor_dir, axis=2)  # (N, k)

            # Blind spot: neighbors behind (cos < blind_cosine threshold)
            # blind_cosine = cos(180 - blind_angle/2) which is negative for rear cone
            blind_cosine = Config.get_blind_cosine()
            in_view = cos_angle > blind_cosine  # True if NOT in blind spot

            vis_mask = vis_mask & in_view

        # Topological Mask
        # We want the first Nc True values in vis_mask for each row.
        # This is tricky in pure numpy.
        # Approximation: Since KDTree sorts by distance, the "First Nc Visible" are just
        # the first Nc elements of the row that satisfy vis_mask.
        # We can simply use the vis_mask counts.

        # Cumulative sum along the row to find the rank of visibility?
        # cumsum(vis_mask) -> 1, 2, 3... at the True positions.
        # We keep entries where cumsum <= Nc
        topo_ranks = np.cumsum(vis_mask, axis=1)
        topo_mask = (topo_ranks <= nc) & vis_mask  # (N, k)

        # --- Forces ---

        # Alignment: Align with topological neighbors
        # neighbor_vel: (N, k, 2)
        neighbor_vel = self.vel[idxs]

        # Mean velocity of topological neighbors
        # We sum neighbor_vel where topo_mask is True
        # topo_mask expanded: (N, k, 1)
        mask_expanded = topo_mask[:, :, np.newaxis]

        # Count of neighbors per agent
        n_counts = np.sum(topo_mask, axis=1)[:, np.newaxis]  # (N, 1)
        # Avoid div zero
        n_counts[n_counts == 0] = 1.0

        avg_vel = np.sum(neighbor_vel * mask_expanded, axis=1) / n_counts
        steering_ali = avg_vel - self.vel  # (N, 2)
        # Zero out if no neighbors
        has_neighbors = np.any(topo_mask, axis=1)  # (N,)
        steering_ali[~has_neighbors] = 0

        # Cohesion: Seek center of topological neighbors
        # Mean relative position (dx)
        avg_dx = np.sum(dx * mask_expanded, axis=1) / n_counts
        steering_coh = avg_dx  # (N, 2)
        steering_coh[~has_neighbors] = 0

        # Separation: Metric-based (all visible close neighbors)
        # separation acts on anyone within SEPARATION_RADIUS, regardless of topology rank?
        # Usually yes.
        sep_mask = (dists < Config.SEPARATION_RADIUS) & valid_mask  # (N, k)
        sep_mask_exp = sep_mask[:, :, np.newaxis]

        # Repulsion vectors: -dx / dist
        # dists (N, k). Avoid 0
        safe_dists = dists.copy()
        safe_dists[safe_dists < 1e-6] = 0.1

        repel_vecs = -dx / safe_dists[:, :, np.newaxis]  # (N, k, 2)

        # Sum repulsion
        forces_sep = np.sum(repel_vecs * sep_mask_exp, axis=1)  # (N, 2)

        # Normalization helper
        def normalize_limit(vec, max_val):
            m = np.linalg.norm(vec, axis=1, keepdims=True)
            mask = (m > max_val).flatten()
            if np.any(mask):
                vec[mask] = (vec[mask] / m[mask]) * max_val
            return vec

        forces_sep = (
            normalize_limit(forces_sep, Config.MAX_FORCE) * Config.SEPARATION_WEIGHT
        )
        forces_ali = (
            normalize_limit(steering_ali, Config.MAX_FORCE) * Config.ALIGNMENT_WEIGHT
        )
        forces_coh = (
            normalize_limit(steering_coh, Config.MAX_FORCE) * Config.COHESION_WEIGHT
        )

        # Noise
        noise = (np.random.rand(self.N, 2) - 0.5) * Config.NOISE_STRENGTH

        self.acc = forces_sep + forces_ali + forces_coh + noise

        # Integration
        self.vel += self.acc

        # Speed Limit
        speeds = np.linalg.norm(self.vel, axis=1)
        speeds[speeds == 0] = 0.1
        factor = np.clip(speeds, Config.MIN_SPEED, Config.MAX_SPEED) / speeds
        self.vel *= factor[:, None]

        # Update Position
        self.pos += self.vel * Config.DT

        # Boundary Wrap
        self.pos[:, 0] = self.pos[:, 0] % self.width
        self.pos[:, 1] = self.pos[:, 1] % self.height
