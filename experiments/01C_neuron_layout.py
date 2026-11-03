from pathlib import Path

from architecture_experiment_base import run_experiment
from architecture_visualization import plot_neuron_layout


CONFIGURATIONS = [
    ("b50k_uniform", [55, 55, 55]),
    ("b50k_increasing", [48, 77, 97]),
    ("b50k_decreasing", [58, 47, 29]),
    ("b50k_expand_contract", [50, 102, 49]),
    ("b50k_contract_expand", [57, 35, 70]),
    ("b100k_uniform", [100, 100, 100]),
    ("b100k_increasing", [80, 147, 160]),
    ("b100k_decreasing", [110, 78, 56]),
    ("b100k_expand_contract", [85, 170, 103]),
    ("b100k_contract_expand", [110, 56, 111]),
    ("b200k_uniform", [175, 175, 175]),
    ("b200k_increasing", [149, 178, 298]),
    ("b200k_decreasing", [197, 149, 99]),
    ("b200k_expand_contract", [146, 289, 143]),
    ("b200k_contract_expand", [200, 102, 199]),
    ("b400k_uniform", [290, 290, 290]),
    ("b400k_increasing", [226, 320, 453]),
    ("b400k_decreasing", [365, 203, 183]),
    ("b400k_expand_contract", [232, 460, 235]),
    ("b400k_contract_expand", [341, 182, 363]),
]


if __name__ == "__main__":
    summary_path = run_experiment(
        experiment_name=Path(__file__).stem,
        configurations=CONFIGURATIONS,
    )
    plot_neuron_layout(summary_path)
