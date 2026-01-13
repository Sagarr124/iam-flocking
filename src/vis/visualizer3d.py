import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.config import Config

class Visualizer3D:
    def __init__(self, flock):
        self.flock = flock
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlim(0, Config.WIDTH)
        self.ax.set_ylim(0, Config.HEIGHT)
        self.ax.set_zlim(0, Config.DEPTH)
        
        # Quiver 3D
        # For performance, maybe just scatter points? Quiver 3D is slow in MPL.
        # Let's try quiver first, if slow, switch to scatter (+ small velocity line?)
        # 3D Quiver: ax.quiver(x, y, z, u, v, w)
        self.quiver = self.ax.quiver(
            self.flock.pos[:, 0], self.flock.pos[:, 1], self.flock.pos[:, 2],
            self.flock.vel[:, 0], self.flock.vel[:, 1], self.flock.vel[:, 2],
            length=2.0, normalize=True, color='black', arrow_length_ratio=0.3
        )
        
        self.time_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes)

    def update(self, frame):
        self.flock.update()
        
        self.quiver.remove() # 3D Quiver doesn't support set_UVC efficiently? 
        # Actually in Matplotlib 3D, set_segments is hard.
        # Clearing and redrawing is common but slow.
        # Efficient hack: modify segments. 
        # But simpler: use scatter for position and a separate line for velocity?
        # Let's try redraw for N=400.
        
        self.quiver = self.ax.quiver(
            self.flock.pos[:, 0], self.flock.pos[:, 1], self.flock.pos[:, 2],
            self.flock.vel[:, 0], self.flock.vel[:, 1], self.flock.vel[:, 2],
            length=2.0, normalize=True, color='black'
        )
        
        self.time_text.set_text(f"Step: {frame}")
        return self.quiver, self.time_text

    def run(self):
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=Config.STEPS,
            interval=50, blit=False 
        )
        plt.show()
