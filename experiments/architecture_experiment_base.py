
from __future__ import annotations

import csv
import gzip
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# 1. 경로 설정
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# ============================================================
# 2. 공통 하이퍼파라미터
# ============================================================

SEED = 42

INPUT_SIZE = 784
OUTPUT_SIZE = 10

ACTIVATION = "relu"
WEIGHT_INIT = "he"

WEIGHT_DECAY = 0
USE_DROPOUT = False
USE_BATCHNORM = False

OPTIMIZER = "adam"
OPTIMIZER_PARAM = {
    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
}

BATCH_SIZE = 100
EPOCHS = 500
VALIDATION_SIZE = 10000

# Early Stopping 설정
PATIENCE = 10
MIN_DELTA = 0.001
MONITOR = "val_loss"


# ============================================================
# 3. Fashion-MNIST 데이터 로드
# ============================================================

def read_idx_gz(kind: str) -> tuple[np.ndarray, np.ndarray]:

    labels_path = DATA_DIR / f"{kind}-labels-idx1-ubyte.gz"
    images_path = DATA_DIR / f"{kind}-images-idx3-ubyte.gz"

    if not labels_path.is_file() or not images_path.is_file():
        raise FileNotFoundError(
            f"Dataset files missing in {DATA_DIR}. "
            "Run python download_dataset.py first."
        )

    with gzip.open(labels_path, "rb") as f:
        y = np.frombuffer(
            f.read(), dtype=np.uint8, offset=8
        ).copy()

    with gzip.open(images_path, "rb") as f:
        x = np.frombuffer(
            f.read(), dtype=np.uint8, offset=16
        ).reshape(len(y), INPUT_SIZE).copy()

    return x, y


def load_training_and_validation():

    x_all, y_all = read_idx_gz("train")
    x_all = x_all.astype(np.float32) / 255.0

    rng = np.random.RandomState(SEED)

    # 클래스별 비율을 유지하는 계층화 분할
    classes, counts = np.unique(
        y_all, return_counts=True
    )

    raw_sizes = counts * VALIDATION_SIZE / len(y_all)
    valid_sizes = np.floor(raw_sizes).astype(int)

    remaining = VALIDATION_SIZE - valid_sizes.sum()
    order = np.argsort(
        -(raw_sizes - valid_sizes),
        kind="stable",
    )
    valid_sizes[order[:remaining]] += 1

    train_indices = []
    valid_indices = []

    for label, valid_count in zip(classes, valid_sizes):

        indices = np.where(y_all == label)[0]
        rng.shuffle(indices)

        valid_indices.extend(indices[:valid_count])
        train_indices.extend(indices[valid_count:])

    train_indices = np.array(train_indices)
    valid_indices = np.array(valid_indices)

    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)

    return (
        x_all[train_indices],
        y_all[train_indices],
        x_all[valid_indices],
        y_all[valid_indices],
    )


# ============================================================
# 4. 모델 평가
# ============================================================

def evaluate(
    network: MultiLayerNetExtend,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1000,
) -> tuple[float, float]:

    total_loss = 0.0
    correct = 0

    for start in range(0, len(y), batch_size):

        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]

        # 평가 모드에서 손실 계산
        loss = network.loss(
            xb, yb, train_flg=False
        )

        total_loss += float(loss) * len(yb)

        # 평가 모드에서 정확도 계산
        logits = network.predict(
            xb, train_flg=False
        )

        predictions = np.argmax(logits, axis=1)
        correct += int(np.sum(predictions == yb))

    avg_loss = total_loss / len(y)
    accuracy = correct / len(y)

    return avg_loss, accuracy


def count_parameters(network: MultiLayerNetExtend) -> int:

    return sum(
        param.size for param in network.params.values()
    )


# ============================================================
# 5. 단일 모델 학습
# ============================================================

def train_model(
    run_id: str,
    hidden_sizes: list[int],
    dataset: tuple,
):

    # 모델마다 시드 재설정
    np.random.seed(SEED)

    x_train, y_train, x_valid, y_valid = dataset

    # --------------------------------------------------------
    # 6-1. Network 생성
    # --------------------------------------------------------

    network = MultiLayerNetExtend(
        input_size=INPUT_SIZE,
        hidden_size_list=hidden_sizes,
        output_size=OUTPUT_SIZE,
        activation=ACTIVATION,
        weight_init_std=WEIGHT_INIT,
        weight_decay_lambda=WEIGHT_DECAY,
        use_dropout=USE_DROPOUT,
        use_batchnorm=USE_BATCHNORM,
    )

    num_params = count_parameters(network)

    # --------------------------------------------------------
    # 6-2. Trainer 생성
    # --------------------------------------------------------

    trainer = Trainer(
        network=network,
        x_train=x_train,
        t_train=y_train,
        x_test=x_valid,
        t_test=y_valid,
        epochs=EPOCHS,
        mini_batch_size=BATCH_SIZE,
        optimizer=OPTIMIZER,
        optimizer_param=OPTIMIZER_PARAM,
        evaluate_sample_num_per_epoch=None,
        verbose=False,
    )

    iterations_per_epoch = int(
        np.ceil(len(y_train) / BATCH_SIZE)
    )

    # --------------------------------------------------------
    # 6-3. 학습 기록 및 Early Stopping 초기화
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_loss_epoch = 0
    best_model_val_acc = 0.0
    best_network_bytes = None

    best_val_acc = -1.0
    best_acc_epoch = 0

    wait = 0
    total_iterations = 0
    stopped_early = False

    print(f"\n========== {run_id} ==========")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Parameters: {num_params:,}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Early Stopping: {MONITOR}")
    print(f"Patience: {PATIENCE}")
    print(f"Min Delta: {MIN_DELTA}")
    print("================================")

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # 6-4. Epoch 단위 학습
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        # 한 Epoch에 해당하는 학습 수행
        for _ in range(iterations_per_epoch):
            trainer.train_step()
            total_iterations += 1

        # 전체 학습 데이터 평가
        train_loss, train_acc = evaluate(
            network, x_train, y_train
        )

        # 전체 검증 데이터 평가
        val_loss, val_acc = evaluate(
            network, x_valid, y_valid
        )

        # ----------------------------------------------------
        # 6-5. 최고 정확도 기록
        # ----------------------------------------------------

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc_epoch = epoch

        # ----------------------------------------------------
        # 6-6. Early Stopping
        # ----------------------------------------------------

        improved = (
            val_loss < best_val_loss - MIN_DELTA
        )

        if improved:

            best_val_loss = val_loss
            best_loss_epoch = epoch
            best_model_val_acc = val_acc

            # 검증 손실이 개선된 모델 저장
            best_network_bytes = pickle.dumps(
                network,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

            wait = 0

        else:
            wait += 1

        # ----------------------------------------------------
        # 6-7. Epoch 기록
        # ----------------------------------------------------

        history.append({
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4%} | "
            f"Wait: {wait}/{PATIENCE}"
        )

        # 조기 종료 조건
        if wait >= PATIENCE:

            stopped_early = True

            print(
                f"\nEarly Stopping triggered "
                f"at Epoch {epoch}"
            )

            break

    training_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # 6-8. 최종 결과 정리
    # --------------------------------------------------------

    summary = {
        "run_id": run_id,
        "hidden_sizes": "-".join(map(str, hidden_sizes)),
        "num_params": num_params,
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

    print(f"\nCompleted: {run_id}")
    print(f"Trained Epochs: {epoch}")
    print(f"Best Loss Epoch: {best_loss_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Saved Model Val Acc: {best_model_val_acc:.4%}")
    print(f"Best Val Acc: {best_val_acc:.4%}")
    print(f"Training Time: {training_time:.2f}s")

    return summary, history, best_network_bytes


# ============================================================
# 6. 모든 실험 조합 실행 및 저장
# ============================================================

def run_experiment(
    experiment_name: str,
    configurations: list[tuple[str, list[int]]],
) -> Path:

    result_dir = RESULTS_DIR / experiment_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # 실험별 통합 Summary 파일
    summary_path = (
        result_dir / f"{experiment_name}_summary.csv"
    )

    summary_fields = [
        "run_id",
        "hidden_sizes",
        "num_params",
        "trained_epochs",
        "total_iterations",
        "final_train_acc",
        "final_val_acc",
        "best_val_loss",
        "best_loss_epoch",
        "best_model_val_acc",
        "best_val_acc",
        "best_acc_epoch",
        "stopped_early",
        "training_time_sec",
    ]

    history_fields = [
        "run_id",
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
    ]

    # --------------------------------------------------------
    # 7-1. 기존 Summary 불러오기
    # --------------------------------------------------------

    summaries = {}

    if summary_path.exists():

        with summary_path.open(
            "r", newline="", encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            if reader.fieldnames != summary_fields:
                raise ValueError(
                    "Existing summary.csv format mismatch. "
                    "Move or delete the old summary first."
                )

            for row in reader:
                summaries[row["run_id"]] = row

    # 데이터셋은 한 번만 로드
    dataset = load_training_and_validation()

    print("\n========== EXPERIMENT ==========")
    print(f"Experiment: {experiment_name}")
    print(f"Total configurations: {len(configurations)}")
    print(f"Train samples: {len(dataset[0]):,}")
    print(f"Validation samples: {len(dataset[2]):,}")
    print("================================")

    # --------------------------------------------------------
    # 7-2. 모든 구조 반복 학습
    # --------------------------------------------------------

    for index, (run_id, hidden_sizes) in enumerate(
        configurations, start=1
    ):

        structure_name = "-".join(map(str, hidden_sizes))

        history_path = (
            result_dir
            / f"{experiment_name}_{structure_name}_history.csv"
        )

        model_path = (
            result_dir
            / f"{experiment_name}_{structure_name}_best.pkl"
        )

        print(
            f"\n[{index}/{len(configurations)}] "
            f"{run_id}"
        )

        # 기존 결과가 모두 있으면 건너뛰기
        if (
            run_id in summaries
            and history_path.exists()
            and model_path.exists()
        ):
            print(f"SKIP: {run_id} (already completed)")
            continue

        # ----------------------------------------------------
        # 7-3. 모델 학습
        # ----------------------------------------------------

        summary, history, best_network_bytes = train_model(
            run_id=run_id,
            hidden_sizes=hidden_sizes,
            dataset=dataset,
        )

        # ----------------------------------------------------
        # 7-4. History 저장
        # ----------------------------------------------------

        with history_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=history_fields,
            )

            writer.writeheader()
            writer.writerows(history)

        # ----------------------------------------------------
        # 7-5. 최고 모델 저장
        # ----------------------------------------------------

        with model_path.open("wb") as f:
            f.write(best_network_bytes)

        # ----------------------------------------------------
        # 7-6. 통합 Summary 갱신
        # ----------------------------------------------------

        summaries[run_id] = summary

        with summary_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=summary_fields,
            )

            writer.writeheader()

            for config_run_id, _ in configurations:
                if config_run_id in summaries:
                    writer.writerow(summaries[config_run_id])

        print(f"Saved: {run_id}")

    print("\n========== ALL COMPLETED ==========")
    print(f"Total configurations: {len(configurations)}")
    print(f"Summary: {summary_path}")
    print(f"Results directory: {result_dir}")
    print("===================================\n")

    return summary_path
