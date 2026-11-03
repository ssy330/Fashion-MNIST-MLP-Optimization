from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ACCURACY_FIELD = "best_model_val_acc"


def _load_results(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _finish_plot(output_path: Path) -> None:
    plt.ylabel("Validation Accuracy")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_width_depth(summary_path: Path) -> list[Path]:
    """Create the two visualizations for experiment 01A."""
    pattern = re.compile(r"w(?P<width>\d+)_d(?P<depth>\d+)")
    results: dict[tuple[int, int], float] = {}

    for row in _load_results(summary_path):
        match = pattern.fullmatch(row["run_id"])
        if match:
            key = (int(match["width"]), int(match["depth"]))
            results[key] = float(row[ACCURACY_FIELD])

    widths = sorted({width for width, _ in results})
    depths = sorted({depth for _, depth in results})
    output_dir = summary_path.parent

    plt.figure(figsize=(8, 5))
    for depth in depths:
        x = [width for width in widths if (width, depth) in results]
        y = [results[(width, depth)] for width in x]
        plt.plot(x, y, marker="o", label=f"Depth {depth}")
    plt.xscale("log", base=2)
    plt.xticks(widths, [str(width) for width in widths])
    plt.xlabel("Hidden Layer Width")
    plt.title("Validation Accuracy by Width")
    plt.legend(title="Hidden Layers")
    width_path = output_dir / "width_vs_accuracy.png"
    _finish_plot(width_path)

    plt.figure(figsize=(8, 5))
    for width in widths:
        x = [depth for depth in depths if (width, depth) in results]
        y = [results[(width, depth)] for depth in x]
        plt.plot(x, y, marker="o", label=f"Width {width}")
    plt.xticks(depths)
    plt.xlabel("Hidden Layer Depth")
    plt.title("Validation Accuracy by Depth")
    plt.legend(title="Width", ncol=2)
    depth_path = output_dir / "depth_vs_accuracy.png"
    _finish_plot(depth_path)

    return [width_path, depth_path]


def plot_budget_depth(summary_path: Path) -> Path:
    """Create the parameter-budget versus depth visualization for 01B."""
    pattern = re.compile(r"b(?P<budget>\d+)k_d(?P<depth>\d+)")
    results: dict[tuple[int, int], float] = {}

    for row in _load_results(summary_path):
        match = pattern.fullmatch(row["run_id"])
        if match:
            key = (int(match["budget"]), int(match["depth"]))
            results[key] = float(row[ACCURACY_FIELD])

    budgets = sorted({budget for budget, _ in results})
    depths = sorted({depth for _, depth in results})

    plt.figure(figsize=(8, 5))
    for budget in budgets:
        x = [depth for depth in depths if (budget, depth) in results]
        y = [results[(budget, depth)] for depth in x]
        plt.plot(x, y, marker="o", label=f"{budget}k parameters")
    plt.xticks(depths)
    plt.xlabel("Hidden Layer Depth")
    plt.title("Validation Accuracy by Depth and Parameter Budget")
    plt.legend(title="Parameter Budget")
    output_path = summary_path.parent / "budget_depth_accuracy.png"
    _finish_plot(output_path)

    return output_path


def plot_neuron_layout(summary_path: Path) -> Path:
    """Create the mean bars and per-budget points for experiment 01C."""
    pattern = re.compile(r"b(?P<budget>\d+)k_(?P<layout>[a-z_]+)")
    results: dict[str, dict[int, float]] = defaultdict(dict)

    for row in _load_results(summary_path):
        match = pattern.fullmatch(row["run_id"])
        if match:
            results[match["layout"]][int(match["budget"])] = float(
                row[ACCURACY_FIELD]
            )

    layouts = [
        "uniform",
        "increasing",
        "decreasing",
        "expand_contract",
        "contract_expand",
    ]
    labels = [
        "Uniform",
        "Increasing",
        "Decreasing",
        "Expand-\nContract",
        "Contract-\nExpand",
    ]
    budgets = sorted({budget for values in results.values() for budget in values})
    positions = list(range(len(layouts)))
    means = [
        sum(results[layout].values()) / len(results[layout])
        if results[layout] else 0.0
        for layout in layouts
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(positions, means, color="#9ecae1", label="Mean accuracy")

    offsets = {
        budget: (index - (len(budgets) - 1) / 2) * 0.13
        for index, budget in enumerate(budgets)
    }
    for budget in budgets:
        x = [position + offsets[budget] for position in positions]
        y = [results[layout].get(budget, float("nan")) for layout in layouts]
        plt.scatter(x, y, s=42, label=f"{budget}k parameters", zorder=3)

    plt.xticks(positions, labels)
    plt.xlabel("Neuron Layout")
    plt.title("Validation Accuracy by Neuron Layout")
    plt.legend(ncol=2)
    output_path = summary_path.parent / "neuron_layout_accuracy.png"
    _finish_plot(output_path)

    return output_path
