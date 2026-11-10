"""Train one random-search-style model from terminal-supplied parameters.

Example (rs_39 with only its hidden-layer structure changed):
    python experiments/run_single_model.py --run-id rs_39_256-256-256 \
        --hidden-sizes 256 256 256 --activation relu --weight-init he \
        --weight-decay 0.001 --use-dropout --dropout-ratio 0.3 \
        --batch-size 64 --optimizer momentum --lr 0.001
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import architecture_experiment_base as base


RESULT_DIR = base.RESULTS_DIR / "single_model"
COMBINED_SUMMARY_PATH = RESULT_DIR / "single_model_summary.csv"


def load_random_search_module():
    """Load the training function without renaming 00_random_search.py."""
    module_path = Path(__file__).with_name("00_random_search.py")
    spec = importlib.util.spec_from_file_location(
        "random_search_experiment", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Fashion-MNIST MLP with supplied parameters."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--hidden-sizes",
        required=True,
        type=int,
        nargs="+",
        metavar="WIDTH",
        help="One or more hidden-layer widths, separated by spaces.",
    )
    parser.add_argument(
        "--activation", choices=("relu", "sigmoid"), required=True
    )
    parser.add_argument(
        "--weight-init", choices=("he", "xavier"), required=True
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-dropout", action="store_true")
    parser.add_argument("--dropout-ratio", type=float)
    parser.add_argument("--use-batchnorm", action="store_true")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "momentum", "adagrad", "adam"),
        required=True,
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument(
        "--max-training-time",
        type=float,
        metavar="SECONDS",
        help="Optional time limit per run; omit to run until completion.",
    )
    arguments = parser.parse_args()

    if any(width <= 0 for width in arguments.hidden_sizes):
        parser.error("--hidden-sizes values must be positive")
    if arguments.weight_decay < 0:
        parser.error("--weight-decay must not be negative")
    if arguments.batch_size <= 0 or arguments.lr <= 0:
        parser.error("--batch-size and --lr must be positive")
    if (
        arguments.max_training_time is not None
        and arguments.max_training_time <= 0
    ):
        parser.error("--max-training-time must be positive")
    if arguments.use_dropout:
        if arguments.dropout_ratio is None:
            parser.error("--use-dropout requires --dropout-ratio")
        if not 0 < arguments.dropout_ratio < 1:
            parser.error("--dropout-ratio must be between 0 and 1")
    elif arguments.dropout_ratio is not None:
        parser.error("--dropout-ratio requires --use-dropout")

    return arguments


def update_combined_summary(new_summary: dict | None = None) -> int:
    """Merge legacy per-model summaries and update the shared summary file."""
    summaries: dict[str, dict] = {}
    fieldnames: list[str] | None = None
    summary_paths = [
        COMBINED_SUMMARY_PATH,
        *sorted(RESULT_DIR.glob("*_summary.csv")),
    ]

    for path in summary_paths:
        if path == COMBINED_SUMMARY_PATH or not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                continue
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Summary format mismatch: {path}")
            for row in reader:
                summaries[row["run_id"]] = row

    if COMBINED_SUMMARY_PATH.is_file():
        with COMBINED_SUMMARY_PATH.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is not None:
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                elif reader.fieldnames != fieldnames:
                    raise ValueError(
                        f"Summary format mismatch: {COMBINED_SUMMARY_PATH}"
                    )
                for row in reader:
                    summaries[row["run_id"]] = row

    if new_summary is not None:
        if fieldnames is None:
            fieldnames = list(new_summary)
        elif list(new_summary) != fieldnames:
            raise ValueError("New summary format does not match existing data")
        summaries[new_summary["run_id"]] = new_summary

    if fieldnames is None:
        return 0

    with COMBINED_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run_id in sorted(summaries):
            writer.writerow(summaries[run_id])

    return len(summaries)


def main() -> None:
    arguments = parse_arguments()
    random_search = load_random_search_module()
    configuration = {
        "run_id": arguments.run_id,
        "hidden_sizes": arguments.hidden_sizes,
        "activation": arguments.activation,
        "weight_init": arguments.weight_init,
        "weight_decay": arguments.weight_decay,
        "use_dropout": arguments.use_dropout,
        "dropout_ratio": arguments.dropout_ratio,
        "use_batchnorm": arguments.use_batchnorm,
        "batch_size": arguments.batch_size,
        "optimizer": arguments.optimizer,
        "lr": arguments.lr,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RESULT_DIR / f"{arguments.run_id}_checkpoint.pkl"
    history_path = RESULT_DIR / f"{arguments.run_id}_history.csv"
    model_path = RESULT_DIR / f"{arguments.run_id}_best.pkl"

    dataset = base.load_training_and_validation()
    summary, history, best_network_bytes = random_search.train_model(
        configuration=configuration,
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        max_training_time_sec=arguments.max_training_time,
    )

    if summary is None:
        print(f"Paused; checkpoint saved to: {checkpoint_path}")
        return

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    with model_path.open("wb") as file:
        file.write(best_network_bytes)

    summary_count = update_combined_summary(summary)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("\n========== SINGLE MODEL COMPLETED ==========")
    print(f"Best Val Loss: {summary['best_val_loss']:.6f}")
    print(f"Saved Model Val Acc: {summary['best_model_val_acc']:.4%}")
    print(f"Best Val Acc: {summary['best_val_acc']:.4%}")
    print(f"Summary ({summary_count} models): {COMBINED_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
