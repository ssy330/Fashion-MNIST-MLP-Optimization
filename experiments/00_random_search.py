from __future__ import annotations

import csv
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

import architecture_experiment_base as base


# ============================================================
# Random Search 설정
# ============================================================

EXPERIMENT_NAME = Path(__file__).stem
N_TRIALS = 100
SEARCH_SEED = 42
# Set to None to run each trial until early stopping or EPOCHS.
# A limit can also be supplied with --max-training-time.
MAX_TRAINING_TIME_SEC_PER_RUN: float | None = None

HIDDEN_WIDTHS = [16, 32, 64, 128, 256, 512]
HIDDEN_DEPTHS = [1, 2, 3, 4, 5]
ACTIVATIONS = ["relu", "sigmoid"]
WEIGHT_DECAYS = [0, 1e-5, 1e-4, 1e-3, 1e-2]
DROPOUT_RATIOS = [0.1, 0.2, 0.3, 0.5]
BATCH_SIZES = [64, 128, 256]
OPTIMIZERS = ["sgd", "momentum", "adagrad", "adam"]
LEARNING_RATES_BY_OPTIMIZER = {
    "sgd": [0.001, 0.01, 0.1],
    "momentum": [0.001, 0.01, 0.1],
    "adagrad": [0.001, 0.01],
    "adam": [0.0001, 0.0003, 0.001, 0.003],
}


def is_valid_configuration(configuration: dict) -> bool:
    """탐색 예산을 낭비할 가능성이 큰 조합을 제외한다."""
    dropout_ratio = configuration["dropout_ratio"]
    min_width = min(configuration["hidden_sizes"])

    if (
        configuration["use_dropout"]
        and dropout_ratio == 0.5
        and min_width <= 32
    ):
        return False

    if (
        configuration["weight_decay"] == 1e-2
        and configuration["use_dropout"]
        and dropout_ratio >= 0.3
    ):
        return False

    if (
        configuration["activation"] == "sigmoid"
        and len(configuration["hidden_sizes"]) >= 4
        and not configuration["use_batchnorm"]
    ):
        return False

    return True


def sample_configurations() -> list[dict]:
    """문서에 정의된 후보군에서 중복 없이 N_TRIALS개를 추출한다."""
    rng = np.random.default_rng(SEARCH_SEED)
    configurations = []
    signatures = set()

    while len(configurations) < N_TRIALS:
        depth = int(rng.choice(HIDDEN_DEPTHS))
        hidden_sizes = [
            int(rng.choice(HIDDEN_WIDTHS))
            for _ in range(depth)
        ]
        activation = str(rng.choice(ACTIVATIONS))
        use_dropout = bool(rng.integers(0, 2))
        optimizer = str(rng.choice(OPTIMIZERS))

        configuration = {
            "hidden_sizes": hidden_sizes,
            "activation": activation,
            "weight_init": "he" if activation == "relu" else "xavier",
            "weight_decay": float(rng.choice(WEIGHT_DECAYS)),
            "use_dropout": use_dropout,
            "dropout_ratio": (
                float(rng.choice(DROPOUT_RATIOS))
                if use_dropout else None
            ),
            "use_batchnorm": bool(rng.integers(0, 2)),
            "batch_size": int(rng.choice(BATCH_SIZES)),
            "optimizer": optimizer,
            "lr": float(
                rng.choice(LEARNING_RATES_BY_OPTIMIZER[optimizer])
            ),
        }

        if not is_valid_configuration(configuration):
            continue

        signature = tuple(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in configuration.items()
        )
        if signature in signatures:
            continue

        signatures.add(signature)
        configuration["run_id"] = f"rs_{len(configurations) + 1:02d}"
        configurations.append(configuration)

    return configurations


def optimizer_parameters(configuration: dict) -> dict:
    parameters = {"lr": configuration["lr"]}

    if configuration["optimizer"] == "momentum":
        parameters["momentum"] = 0.9
    elif configuration["optimizer"] == "adam":
        parameters["beta1"] = 0.9
        parameters["beta2"] = 0.999

    return parameters


def train_model(
    configuration: dict,
    dataset: tuple,
    checkpoint_path: Path,
    max_training_time_sec: float | None,
) -> tuple[dict | None, list[dict], bytes]:
    """하나의 조합을 학습하거나 시간 제한 체크포인트에서 재개한다."""

    x_train, y_train, x_valid, y_valid = dataset
    optimizer_param = optimizer_parameters(configuration)
    iterations_per_epoch = int(
        np.ceil(len(y_train) / configuration["batch_size"])
    )

    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as file:
            checkpoint = pickle.load(file)

        if checkpoint["configuration"] != configuration:
            raise ValueError(
                f"Checkpoint configuration mismatch: {configuration['run_id']}"
            )

        network = checkpoint["network"]
        trainer = base.Trainer(
            network=network,
            x_train=x_train,
            t_train=y_train,
            x_test=x_valid,
            t_test=y_valid,
            epochs=base.EPOCHS,
            mini_batch_size=configuration["batch_size"],
            optimizer=configuration["optimizer"],
            optimizer_param=optimizer_param,
            evaluate_sample_num_per_epoch=None,
            verbose=False,
        )
        trainer.optimizer = checkpoint["optimizer"]
        trainer.current_iter = checkpoint["total_iterations"]
        np.random.set_state(checkpoint["rng_state"])

        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        best_val_loss = checkpoint["best_val_loss"]
        best_loss_epoch = checkpoint["best_loss_epoch"]
        best_model_val_acc = checkpoint["best_model_val_acc"]
        best_network_bytes = checkpoint["best_network_bytes"]
        best_val_acc = checkpoint["best_val_acc"]
        best_acc_epoch = checkpoint["best_acc_epoch"]
        wait = checkpoint["wait"]
        total_iterations = checkpoint["total_iterations"]
        accumulated_training_time = checkpoint["training_time_sec"]
        print(f"Resuming from Epoch {start_epoch}.")
    else:
        np.random.seed(base.SEED)
        network = base.MultiLayerNetExtend(
            input_size=base.INPUT_SIZE,
            hidden_size_list=configuration["hidden_sizes"],
            output_size=base.OUTPUT_SIZE,
            activation=configuration["activation"],
            weight_init_std=configuration["weight_init"],
            weight_decay_lambda=configuration["weight_decay"],
            use_dropout=configuration["use_dropout"],
            dropout_ration=configuration["dropout_ratio"] or 0.5,
            use_batchnorm=configuration["use_batchnorm"],
        )
        trainer = base.Trainer(
            network=network,
            x_train=x_train,
            t_train=y_train,
            x_test=x_valid,
            t_test=y_valid,
            epochs=base.EPOCHS,
            mini_batch_size=configuration["batch_size"],
            optimizer=configuration["optimizer"],
            optimizer_param=optimizer_param,
            evaluate_sample_num_per_epoch=None,
            verbose=False,
        )
        start_epoch = 1
        history = []
        best_val_loss = float("inf")
        best_loss_epoch = 0
        best_model_val_acc = 0.0
        best_network_bytes = b""
        best_val_acc = -1.0
        best_acc_epoch = 0
        wait = 0
        total_iterations = 0
        accumulated_training_time = 0.0

    stopped_early = False

    print(f"\n========== {configuration['run_id']} ==========")
    print(f"Hidden sizes: {configuration['hidden_sizes']}")
    print(
        "Settings: "
        f"activation={configuration['activation']}, "
        f"optimizer={configuration['optimizer']}, "
        f"lr={configuration['lr']}, "
        f"batch_size={configuration['batch_size']}"
    )

    start_time = time.perf_counter()
    for epoch in range(start_epoch, base.EPOCHS + 1):
        for _ in range(iterations_per_epoch):
            trainer.train_step()
            total_iterations += 1

        train_loss, train_acc = base.evaluate(network, x_train, y_train)
        val_loss, val_acc = base.evaluate(network, x_valid, y_valid)

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            history.append({
                "run_id": configuration["run_id"],
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            print(
                "Non-finite loss detected; "
                "ending this trial early."
            )
            stopped_early = True
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc_epoch = epoch

        if val_loss < best_val_loss - base.MIN_DELTA:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            best_model_val_acc = val_acc
            best_network_bytes = pickle.dumps(
                network,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            wait = 0
        else:
            wait += 1

        history.append({
            "run_id": configuration["run_id"],
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(
            f"Epoch {epoch:3d}/{base.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4%} | "
            f"Wait: {wait}/{base.PATIENCE}"
        )

        elapsed_time = time.perf_counter() - start_time
        if (
            max_training_time_sec is not None
            and elapsed_time >= max_training_time_sec
        ):
            checkpoint = {
                "configuration": configuration,
                "network": network,
                "optimizer": trainer.optimizer,
                "epoch": epoch,
                "history": history,
                "best_val_loss": best_val_loss,
                "best_loss_epoch": best_loss_epoch,
                "best_model_val_acc": best_model_val_acc,
                "best_network_bytes": best_network_bytes,
                "best_val_acc": best_val_acc,
                "best_acc_epoch": best_acc_epoch,
                "wait": wait,
                "total_iterations": total_iterations,
                "training_time_sec": accumulated_training_time + elapsed_time,
                "rng_state": np.random.get_state(),
            }
            with checkpoint_path.open("wb") as file:
                pickle.dump(checkpoint, file, protocol=pickle.HIGHEST_PROTOCOL)

            print(
                f"Time limit reached after Epoch {epoch}; "
                f"checkpoint saved to {checkpoint_path.name}."
            )
            return None, history, b""

        if wait >= base.PATIENCE:
            stopped_early = True
            break

    training_time = accumulated_training_time + (
        time.perf_counter() - start_time
    )
    summary = {
        **configuration,
        "hidden_sizes": "-".join(map(str, configuration["hidden_sizes"])),
        "num_params": base.count_parameters(network),
        "trained_epochs": epoch,
        "total_iterations": total_iterations,
        "final_train_acc": history[-1]["train_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "best_val_loss": best_val_loss,
        "best_loss_epoch": best_loss_epoch,
        "best_model_val_acc": best_model_val_acc,
        "best_val_acc": best_val_acc,
        "best_acc_epoch": best_acc_epoch,
        "stopped_early": stopped_early,
        "training_time_sec": round(training_time, 2),
    }
    return summary, history, best_network_bytes


def run_random_search(
    max_training_time_sec: float | None = MAX_TRAINING_TIME_SEC_PER_RUN,
) -> None:
    if max_training_time_sec is not None and max_training_time_sec <= 0:
        raise ValueError("max_training_time_sec must be positive or None")

    configurations = sample_configurations()
    result_dir = base.RESULTS_DIR / EXPERIMENT_NAME
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir = result_dir / "runs"
    run_dir.mkdir(exist_ok=True)
    summary_path = result_dir / f"{EXPERIMENT_NAME}_summary.csv"
    summary_fields = [
        "run_id", "hidden_sizes", "activation", "weight_init",
        "weight_decay", "use_dropout", "dropout_ratio", "use_batchnorm",
        "batch_size", "optimizer", "lr", "num_params", "trained_epochs",
        "total_iterations", "final_train_acc", "final_val_acc", "best_val_loss",
        "best_loss_epoch", "best_model_val_acc", "best_val_acc", "best_acc_epoch",
        "stopped_early", "training_time_sec",
    ]
    history_fields = [
        "run_id", "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
    ]
    summaries = {}

    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != summary_fields:
                raise ValueError(
                    "Existing random-search summary format mismatch. "
                    "Move or delete the old summary first."
                )
            summaries = {row["run_id"]: row for row in reader}

    dataset = base.load_training_and_validation()
    print("\n========== RANDOM SEARCH ==========")
    print(f"Trials: {N_TRIALS}")
    print(f"Search seed: {SEARCH_SEED}")
    if max_training_time_sec is None:
        print("Per-trial time limit: none")
    else:
        print(f"Per-trial time limit: {max_training_time_sec:g} seconds")
    print("===================================")

    for index, configuration in enumerate(configurations, start=1):
        run_id = configuration["run_id"]
        history_path = run_dir / f"{run_id}_history.csv"
        model_path = run_dir / f"{run_id}_best.pkl"
        checkpoint_path = run_dir / f"{run_id}_checkpoint.pkl"

        print(f"\n[{index}/{len(configurations)}] {run_id}")
        if (
            run_id in summaries
            and history_path.exists()
            and model_path.exists()
        ):
            print(f"SKIP: {run_id} (already completed)")
            continue

        summary, history, best_network_bytes = train_model(
            configuration,
            dataset,
            checkpoint_path,
            max_training_time_sec,
        )

        if summary is None:
            print(f"PAUSE: {run_id} (resume on the next execution)")
            continue

        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=history_fields)
            writer.writeheader()
            writer.writerows(history)

        with model_path.open("wb") as file:
            file.write(best_network_bytes)

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        summaries[run_id] = summary
        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=summary_fields)
            writer.writeheader()
            for saved_configuration in configurations:
                saved_run_id = saved_configuration["run_id"]
                if saved_run_id in summaries:
                    writer.writerow(summaries[saved_run_id])

    print("\n========== RANDOM SEARCH COMPLETED ==========")
    print(f"Summary: {summary_path}")
    print(f"Results directory: {result_dir}")
    print("=============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Fashion-MNIST random-search trials."
    )
    parser.add_argument(
        "--max-training-time",
        type=float,
        default=MAX_TRAINING_TIME_SEC_PER_RUN,
        metavar="SECONDS",
        help=(
            "Maximum seconds per trial before saving a checkpoint and moving "
            "on. Omit this option to run without a time limit."
        ),
    )
    arguments = parser.parse_args()
    run_random_search(arguments.max_training_time)
