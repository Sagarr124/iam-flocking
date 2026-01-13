import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.core.flock import Flock
from src.vis.interactive import InteractiveVisualizer


def main():
    print("Interactive Mode: Use sliders to control visibility and separation.")
    flock = Flock()
    vis = InteractiveVisualizer(flock)
    vis.run()


if __name__ == "__main__":
    main()
