import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.core.flock3d import Flock3D
from src.vis.visualizer3d import Visualizer3D


def main():
    print(Config.info() + " (3D Mode)")
    flock = Flock3D()
    vis = Visualizer3D(flock)
    vis.run()


if __name__ == "__main__":
    main()
