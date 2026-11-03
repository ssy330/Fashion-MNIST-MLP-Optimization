from pathlib import Path

from architecture_experiment_base import run_experiment
from architecture_visualization import plot_width_depth


WIDTHS = [16, 32, 64, 128, 256, 512]
DEPTHS = [1, 2, 3, 4, 5]

CONFIGURATIONS = [
    (f"w{width}_d{depth}", [width] * depth)
    for width in WIDTHS
    for depth in DEPTHS
]


if __name__ == "__main__":
    summary_path = run_experiment(
        experiment_name=Path(__file__).stem,
        configurations=CONFIGURATIONS,
    )
    plot_width_depth(summary_path)
