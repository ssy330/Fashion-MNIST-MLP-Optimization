from pathlib import Path

from architecture_experiment_base import run_experiment
from architecture_visualization import plot_budget_depth


BUDGET_WIDTHS = {
    50_000: [63, 59, 55, 52, 50],
    100_000: [126, 110, 100, 93, 87],
    200_000: [252, 201, 175, 157, 145],
    400_000: [503, 349, 290, 256, 232],
}

CONFIGURATIONS = [
    (f"b{budget // 1000}k_d{depth}", [width] * depth)
    for budget, widths in BUDGET_WIDTHS.items()
    for depth, width in enumerate(widths, start=1)
]


if __name__ == "__main__":
    summary_path = run_experiment(
        experiment_name=Path(__file__).stem,
        configurations=CONFIGURATIONS,
    )
    plot_budget_depth(summary_path)
