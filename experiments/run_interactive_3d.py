import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.core.flock3d import Flock3D
from src.vis.interactive3d import InteractiveVisualizer3D


def main():
    print("Interactive 3D Mode: Use sliders to control visibility and separation.")
    flock = Flock3D()
    vis = InteractiveVisualizer3D(flock)
    vis.run()


if __name__ == "__main__":
    main()
