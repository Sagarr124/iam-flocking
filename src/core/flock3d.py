import numpy as np
from scipy.spatial import KDTree

from src.config import Config


class Flock3D:
    def __init__(self):
        self.N = Config.N_AGENTS
        self.width = Config.WIDTH
        self.height = Config.HEIGHT
        self.depth = Config.DEPTH
        self.dims = 3

        # State: Position, Velocity, Acceleration (3D)
        # Random initial positions
        self.pos = np.random.rand(self.N, 3) * np.array(
            [self.width, self.height, self.depth]
        )

        # Random initial velocities on a sphere
        # Standard Normal -> Normalize
        v = np.random.randn(self.N, 3)
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        speeds = (
            np.random.rand(self.N) * (Config.MAX_SPEED - Config.MIN_SPEED)
        ) + Config.MIN_SPEED
        self.vel = v * speeds[:, None]
        self.acc = np.zeros((self.N, 3))

    def update(self, metric_override=None):
        """
        One simulation step (Vectorized 3D).
        """
        self.acc = np.zeros((self.N, 3))

        sensing_radius = (
            metric_override if metric_override is not None else Config.MAX_SENSING_RANGE
        )
        nc = Config.NC

        # KDTree supports 3D nature automatically
        tree = KDTree(self.pos, boxsize=[self.width, self.height, self.depth])

        k_query = nc + 10
        dists, idxs = tree.query(self.pos, k=k_query)

        # Relative vectors (N, k, 3)
        neighbor_pos = self.pos[idxs]
        my_pos = self.pos[:, np.newaxis, :]
        dx = neighbor_pos - my_pos

        # Periodic Wrap 3D
        box = np.array([self.width, self.height, self.depth])
        dx -= np.round(dx / box) * box

        # Masks
        valid_mask = dists > 1e-6
        vis_mask = (dists < sensing_radius) & valid_mask

        # Blind Spot: Agents cannot see directly behind them (3D)
        if Config.BLIND_SPOT_ANGLE_DEG > 0:
            # Normalize velocity to get heading direction
            vel_norm = np.linalg.norm(self.vel, axis=1, keepdims=True)
            vel_norm[vel_norm < 1e-6] = 1.0
            heading = self.vel / vel_norm  # (N, 3)

            # Direction to each neighbor (normalized dx)
            dx_norm = np.linalg.norm(dx, axis=2, keepdims=True)
            dx_norm[dx_norm < 1e-6] = 1.0
            neighbor_dir = dx / dx_norm  # (N, k, 3)

            # Dot product for 3D
            heading_expanded = heading[:, np.newaxis, :]
            cos_angle = np.sum(heading_expanded * neighbor_dir, axis=2)  # (N, k)

            blind_cosine = Config.get_blind_cosine()
            in_view = cos_angle > blind_cosine

            vis_mask = vis_mask & in_view

        topo_ranks = np.cumsum(vis_mask, axis=1)
        topo_mask = (topo_ranks <= nc) & vis_mask

        # --- Forces (Vectorized) ---
        neighbor_vel = self.vel[idxs]
        mask_expanded = topo_mask[:, :, np.newaxis]
        n_counts = np.sum(topo_mask, axis=1)[:, np.newaxis]
        n_counts[n_counts == 0] = 1.0

        # Alignment
        avg_vel = np.sum(neighbor_vel * mask_expanded, axis=1) / n_counts
        steering_ali = avg_vel - self.vel
        has_neighbors = np.any(topo_mask, axis=1)
        steering_ali[~has_neighbors] = 0

        # Cohesion
        avg_dx = np.sum(dx * mask_expanded, axis=1) / n_counts
        steering_coh = avg_dx
        steering_coh[~has_neighbors] = 0

        # Separation
        sep_mask = (dists < Config.SEPARATION_RADIUS) & valid_mask
        sep_mask_exp = sep_mask[:, :, np.newaxis]

        safe_dists = dists.copy()
        safe_dists[safe_dists < 1e-6] = 0.1
        repel_vecs = -dx / safe_dists[:, :, np.newaxis]
        forces_sep = np.sum(repel_vecs * sep_mask_exp, axis=1)

        # Normalize
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

        noise = (np.random.rand(self.N, 3) - 0.5) * Config.NOISE_STRENGTH

        self.acc = forces_sep + forces_ali + forces_coh + noise
        self.vel += self.acc

        speeds = np.linalg.norm(self.vel, axis=1)
        speeds[speeds == 0] = 0.1
        factor = np.clip(speeds, Config.MIN_SPEED, Config.MAX_SPEED) / speeds
        self.vel *= factor[:, None]

        self.pos += self.vel * Config.DT

        # Boundary Wrap
        self.pos[:, 0] = self.pos[:, 0] % self.width
        self.pos[:, 1] = self.pos[:, 1] % self.height
        self.pos[:, 2] = self.pos[:, 2] % self.depth
