import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from src.analysis.metrics import calculate_fragmentation, calculate_order_parameter
from src.config import Config


class InteractiveVisualizer:
    def __init__(self, flock):
        self.flock = flock
        # Compact layout: Sliders at bottom, but organized in 2 columns
        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        self.fig.subplots_adjust(left=0.05, top=0.90, bottom=0.25)

        self.ax.set_xlim(0, Config.WIDTH)
        self.ax.set_ylim(0, Config.HEIGHT)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.scat = self.ax.scatter(
            self.flock.pos[:, 0], self.flock.pos[:, 1], s=10, c="black", marker=">"
        )

        # Metrics display panel
        self.time_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # --- Sliders ---
        axcolor = "lightgoldenrodyellow"

        # Grid parameters
        # Cols: x=0.15 and x=0.60. Width=0.30
        # Rows: bottom=0.02, 0.06, 0.10, 0.14

        def make_slider(col, row, label, min_val, max_val, init_val, valfmt=None):
            x = 0.15 if col == 0 else 0.60
            y = 0.02 + (row * 0.04)
            ax = plt.axes([x, y, 0.30, 0.03], facecolor=axcolor)
            return Slider(ax, label, min_val, max_val, valinit=init_val, valfmt=valfmt)

        # Column 0
        self.s_vis = make_slider(
            0, 3, "Visibility", 0.1, 100.0, Config.MAX_SENSING_RANGE
        )
        self.s_sep_r = make_slider(
            0, 2, "Sep. Radius", 0.1, 5.0, Config.SEPARATION_RADIUS
        )
        self.s_nc = make_slider(0, 1, "Neighbors", 1, 20, Config.NC, valfmt="%0.0f")
        self.s_speed = make_slider(0, 0, "Max Speed", 0.1, 5.0, Config.MAX_SPEED)

        # Column 1
        self.s_align = make_slider(1, 3, "Align Wgt", 0.0, 5.0, Config.ALIGNMENT_WEIGHT)
        self.s_coh = make_slider(1, 2, "Cohesion Wgt", 0.0, 5.0, Config.COHESION_WEIGHT)
        self.s_sep_w = make_slider(1, 1, "Sep. Wgt", 0.0, 5.0, Config.SEPARATION_WEIGHT)
        self.s_force = make_slider(1, 0, "Max Force", 0.01, 0.2, Config.MAX_FORCE)

        # Callbacks
        self.s_vis.on_changed(self.update_vis)
        self.s_sep_r.on_changed(lambda v: setattr(Config, "SEPARATION_RADIUS", v))
        self.s_nc.on_changed(self.update_nc)
        self.s_speed.on_changed(lambda v: setattr(Config, "MAX_SPEED", v))

        self.s_align.on_changed(lambda v: setattr(Config, "ALIGNMENT_WEIGHT", v))
        self.s_coh.on_changed(lambda v: setattr(Config, "COHESION_WEIGHT", v))
        self.s_sep_w.on_changed(lambda v: setattr(Config, "SEPARATION_WEIGHT", v))
        self.s_force.on_changed(lambda v: setattr(Config, "MAX_FORCE", v))

        self.current_vis = Config.MAX_SENSING_RANGE

    def update_vis(self, val):
        self.current_vis = val

    def update_nc(self, val):
        Config.NC = int(val)

    def update(self, frame):
        self.flock.update(metric_override=self.current_vis)
        self.scat.set_offsets(self.flock.pos)

        # Calculate real-time metrics
        order = calculate_order_parameter(self.flock.vel)
        n_fragments, largest = calculate_fragmentation(
            self.flock.pos, connection_radius=Config.SEPARATION_RADIUS * 3.0
        )
        cohesion = largest / Config.N_AGENTS

        # Display metrics
        metrics_text = (
            f"Step: {frame:5d}  |  Visibility: {self.current_vis:.1f}\n"
            f"Order (φ): {order:.3f}  |  Cohesion: {cohesion:.3f}\n"
            f"Fragments: {n_fragments:3d}  |  Nc: {Config.NC}"
        )
        self.time_text.set_text(metrics_text)

        return self.scat, self.time_text

    def run(self):
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=Config.STEPS, interval=20, blit=False
        )
        plt.show()
