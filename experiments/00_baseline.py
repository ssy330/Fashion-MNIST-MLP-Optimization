
"""Fashion-MNIST Baseline
Based on: deep-learning-from-scratch/ch06/weight_init_compare.py

Baseline settings:
- Hidden layers: [100, 100, 100, 100]
- Activation: ReLU
- Weight initialization: He
- Optimizer: SGD (lr=0.01)
- Batch size: 128
- Iterations: 2000
- Seed: 42

The official test set is NOT used.
"""

from __future__ import annotations

import csv
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np

# ============================================================
# 1. 경로 설정
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# 현재 파이썬 파일명에서 실험 이름 자동 추출
EXPERIMENT_NAME = Path(__file__).stem

# 결과 파일 경로 자동 생성
HISTORY_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}_history.csv"
MODEL_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}_best.pkl"


# ============================================================
# 2. 하이퍼파라미터 설정
# ============================================================

SEED = 42

HIDDEN_SIZES = [100, 100, 100, 100]
ACTIVATION = "relu"
WEIGHT_INIT = "he"

LEARNING_RATE = 0.01
BATCH_SIZE = 128
MAX_ITERATIONS = 2000

WEIGHT_DECAY = 0
USE_DROPOUT = False
USE_BATCHNORM = False

VALIDATION_SIZE = 10000
LOG_EVERY = 100


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
        ).reshape(len(y), 784)

    return x, y


def load_training_and_validation():

    # 공식 학습 데이터 60,000개만 사용
    x_all, y_all = read_idx_gz("train")

    # 픽셀값 정규화
    x_all = x_all.astype(np.float32) / 255.0

    # 모든 팀원이 동일한 데이터 분할 사용
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(y_all))

    valid_idx = indices[:VALIDATION_SIZE]
    train_idx = indices[VALIDATION_SIZE:]

    return (
        x_all[train_idx],
        y_all[train_idx],
        x_all[valid_idx],
        y_all[valid_idx],
    )


# ============================================================
# 4. 검증 정확도 계산
# ============================================================

def accuracy_in_batches(
    network: MultiLayerNetExtend,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1000,
) -> float:

    correct = 0

    for start in range(0, len(y), batch_size):

        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]

        logits = network.predict(xb, train_flg=False)
        predictions = np.argmax(logits, axis=1)

        correct += int(np.sum(predictions == yb))

    return correct / len(y)


# ============================================================
# 5. 모델 학습
# ============================================================

def main():

    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_valid, y_valid = (
        load_training_and_validation()
    )

    # 신경망 생성
    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=HIDDEN_SIZES,
        output_size=10,
        activation=ACTIVATION,
        weight_init_std=WEIGHT_INIT,
        weight_decay_lambda=WEIGHT_DECAY,
        use_dropout=USE_DROPOUT,
        use_batchnorm=USE_BATCHNORM,
    )

    optimizer = SGD(lr=LEARNING_RATE)

    history = []
    best_val_acc = -1.0
    best_network_bytes = None

    print(f"\nExperiment: {EXPERIMENT_NAME}")
    print(f"Train: {len(y_train):,}")
    print(f"Validation: {len(y_valid):,}")
    print(f"Hidden layers: {HIDDEN_SIZES}")
    print(f"Optimizer: SGD")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Iterations: {MAX_ITERATIONS}\n")

    for step in range(1, MAX_ITERATIONS + 1):

        # 미니배치 추출
        batch_idx = np.random.choice(
            len(y_train), BATCH_SIZE
        )

        xb = x_train[batch_idx]
        yb = y_train[batch_idx]

        # 오차역전파
        grads = network.gradient(xb, yb)

        # 가중치 갱신
        optimizer.update(network.params, grads)

        # 학습 결과 기록
        if (
            step == 1
            or step % LOG_EVERY == 0
            or step == MAX_ITERATIONS
        ):

            loss = float(
                network.loss(xb, yb, train_flg=False)
            )

            val_acc = accuracy_in_batches(
                network, x_valid, y_valid
            )

            history.append({
                "iteration": step,
                "train_batch_loss": loss,
                "val_accuracy": val_acc,
            })

            print(
                f"Step {step:4d} | "
                f"Loss: {loss:.4f} | "
                f"Val Acc: {val_acc:.4%}"
            )

            # 최고 검증 정확도 모델 저장
            if val_acc > best_val_acc:

                best_val_acc = val_acc

                best_network_bytes = pickle.dumps(
                    network,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    # ========================================================
    # 6. 실험 결과 저장
    # ========================================================

    # CSV 저장
    with HISTORY_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "train_batch_loss",
                "val_accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    # 최고 성능 모델 저장
    with MODEL_PATH.open("wb") as f:
        f.write(best_network_bytes)

    print("\n========== RESULT ==========")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Best Validation Accuracy: {best_val_acc:.4%}")
    print(f"History saved: {HISTORY_PATH}")
    print(f"Model saved: {MODEL_PATH}")
    print("============================\n")


if __name__ == "__main__":
    main()
