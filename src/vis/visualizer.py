import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.config import Config

class Visualizer:
    def __init__(self, flock):
        self.flock = flock
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, Config.WIDTH)
        self.ax.set_ylim(0, Config.HEIGHT)
        self.ax.set_aspect('equal')
        
        # Quiver plot for birds
        self.quiver = self.ax.quiver(
            self.flock.pos[:, 0], self.flock.pos[:, 1],
            self.flock.vel[:, 0], self.flock.vel[:, 1],
            scale=50, width=0.003, headwidth=4, color='black'
        )
        
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)

    def update(self, frame):
        # Update simulation
        self.flock.update()
        
        # Update drawing
        self.quiver.set_offsets(self.flock.pos)
        self.quiver.set_UVC(self.flock.vel[:, 0], self.flock.vel[:, 1])
        self.time_text.set_text(f"Step: {frame}")
        return self.quiver, self.time_text

    def run(self):
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=Config.STEPS,
            interval=30, blit=True
        )
        plt.show()
