import os
import sys

# Put project root in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.core.flock import Flock
from src.vis.visualizer import Visualizer


def main():
    print(Config.info())
    flock = Flock()
    vis = Visualizer(flock)
    vis.run()


if __name__ == "__main__":
    main()
