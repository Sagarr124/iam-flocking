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

    def update(self, metric_override=None, mode="topological"):
        """
        One simulation step object.
        Optimized vectorized implementation.
        """
        self.acc = np.zeros((self.N, 2))

        if mode not in {"topological", "metric"}:
            raise ValueError(f"Unknown mode: {mode}")

        sensing_radius = (
            metric_override if metric_override is not None else Config.MAX_SENSING_RANGE
        )
        nc = Config.NC

        # Normalization helper
        def normalize_limit(vec, max_val):
            m = np.linalg.norm(vec, axis=1, keepdims=True)
            mask = (m > max_val).flatten()
            if np.any(mask):
                vec[mask] = (vec[mask] / m[mask]) * max_val
            return vec

        # 1. Neighbor Search
        # In topological mode we only need a small set of candidate neighbors; in metric
        # mode we require the full (variable-sized) neighborhood within sensing_radius.
        tree = KDTree(self.pos, boxsize=[self.width, self.height])
        box = np.array([self.width, self.height])

        if mode == "topological":
            # tree.query returns (N, k). self.pos is (N, 2)
            k_query = nc + 10
            dists, idxs = tree.query(self.pos, k=k_query)

            neighbor_pos = self.pos[idxs]  # (N, k, 2)
            my_pos = self.pos[:, np.newaxis, :]
            dx = neighbor_pos - my_pos  # (N, k, 2)
            dx -= np.round(dx / box) * box

            valid_mask = dists > 1e-6  # (N, k)
            vis_mask = (dists < sensing_radius) & valid_mask  # (N, k)

            if Config.BLIND_SPOT_ANGLE_DEG > 0:
                vel_norm = np.linalg.norm(self.vel, axis=1, keepdims=True)
                vel_norm[vel_norm < 1e-6] = 1.0
                heading = self.vel / vel_norm  # (N, 2)

                dx_norm = np.linalg.norm(dx, axis=2, keepdims=True)
                dx_norm[dx_norm < 1e-6] = 1.0
                neighbor_dir = dx / dx_norm  # (N, k, 2)

                heading_expanded = heading[:, np.newaxis, :]
                cos_angle = np.sum(heading_expanded * neighbor_dir, axis=2)  # (N, k)

                blind_cosine = Config.get_blind_cosine()
                in_view = cos_angle > blind_cosine
                vis_mask = vis_mask & in_view

            topo_ranks = np.cumsum(vis_mask, axis=1)
            topo_mask = (topo_ranks <= nc) & vis_mask  # (N, k)

            neighbor_vel = self.vel[idxs]
            mask_expanded = topo_mask[:, :, np.newaxis]

            n_counts = np.sum(topo_mask, axis=1)[:, np.newaxis]
            n_counts[n_counts == 0] = 1.0

            avg_vel = np.sum(neighbor_vel * mask_expanded, axis=1) / n_counts
            steering_ali = avg_vel - self.vel
            has_neighbors = np.any(topo_mask, axis=1)
            steering_ali[~has_neighbors] = 0

            avg_dx = np.sum(dx * mask_expanded, axis=1) / n_counts
            steering_coh = avg_dx
            steering_coh[~has_neighbors] = 0

            sep_mask = (dists < Config.SEPARATION_RADIUS) & valid_mask
            sep_mask_exp = sep_mask[:, :, np.newaxis]

            safe_dists = dists.copy()
            safe_dists[safe_dists < 1e-6] = 0.1
            repel_vecs = -dx / safe_dists[:, :, np.newaxis]
            forces_sep = np.sum(repel_vecs * sep_mask_exp, axis=1)

            forces_sep = (
                normalize_limit(forces_sep, Config.MAX_FORCE) * Config.SEPARATION_WEIGHT
            )
            forces_ali = (
                normalize_limit(steering_ali, Config.MAX_FORCE)
                * Config.ALIGNMENT_WEIGHT
            )
            forces_coh = (
                normalize_limit(steering_coh, Config.MAX_FORCE) * Config.COHESION_WEIGHT
            )
        else:
            # Metric-only: use all visible neighbors within sensing_radius
            forces_sep = np.zeros((self.N, 2))
            steering_ali = np.zeros((self.N, 2))
            steering_coh = np.zeros((self.N, 2))

            blind_cosine = Config.get_blind_cosine()
            vel_norm = np.linalg.norm(self.vel, axis=1, keepdims=True)
            vel_norm[vel_norm < 1e-6] = 1.0
            heading = self.vel / vel_norm

            for i in range(self.N):
                neigh = tree.query_ball_point(self.pos[i], r=sensing_radius)
                if i in neigh:
                    neigh.remove(i)

                if neigh:
                    neigh_idx = np.array(neigh, dtype=int)
                    dx_i = self.pos[neigh_idx] - self.pos[i]
                    dx_i -= np.round(dx_i / box) * box

                    if Config.BLIND_SPOT_ANGLE_DEG > 0:
                        dx_norm = np.linalg.norm(dx_i, axis=1)
                        valid = dx_norm > 1e-6
                        dx_i = dx_i[valid]
                        neigh_idx = neigh_idx[valid]

                        if dx_i.shape[0] > 0:
                            neighbor_dir = dx_i / dx_norm[valid][:, None]
                            cos_angle = np.sum(neighbor_dir * heading[i], axis=1)
                            in_view = cos_angle > blind_cosine
                            dx_i = dx_i[in_view]
                            neigh_idx = neigh_idx[in_view]

                    if dx_i.shape[0] > 0:
                        avg_vel = np.mean(self.vel[neigh_idx], axis=0)
                        steering_ali[i] = avg_vel - self.vel[i]
                        steering_coh[i] = np.mean(dx_i, axis=0)

                close = tree.query_ball_point(self.pos[i], r=Config.SEPARATION_RADIUS)
                if i in close:
                    close.remove(i)

                if close:
                    close_idx = np.array(close, dtype=int)
                    dx_c = self.pos[close_idx] - self.pos[i]
                    dx_c -= np.round(dx_c / box) * box
                    d = np.linalg.norm(dx_c, axis=1)
                    valid = d > 1e-6
                    dx_c = dx_c[valid]
                    d = d[valid]
                    if dx_c.shape[0] > 0:
                        repel_vecs = -dx_c / d[:, None]
                        forces_sep[i] = np.sum(repel_vecs, axis=0)

            forces_sep = (
                normalize_limit(forces_sep, Config.MAX_FORCE) * Config.SEPARATION_WEIGHT
            )
            forces_ali = (
                normalize_limit(steering_ali, Config.MAX_FORCE)
                * Config.ALIGNMENT_WEIGHT
            )
            forces_coh = (
                normalize_limit(steering_coh, Config.MAX_FORCE) * Config.COHESION_WEIGHT
            )

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
