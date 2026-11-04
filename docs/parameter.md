# MLP 및 학습 파라미터 정리

이 문서는 Fashion-MNIST 분류에 사용하는 `MultiLayerNetExtend`와 `Trainer`의 주요 파라미터를 정리한 참고 문서이다.

## 1. 모델 구조 파라미터

```python
network = MultiLayerNetExtend(
    input_size=784,
    hidden_size_list=[100, 100],
    output_size=10,
    activation="relu",
    weight_init_std="he",
    weight_decay_lambda=0,
    use_dropout=False,
    use_batchnorm=False,
)
```

| 파라미터 | 의미 | 기본 설정 | Random Search 후보 |
|---|---|---|---|
| `input_size` | 입력 벡터 차원 | `784` | `784` (고정) |
| `hidden_size_list` | 은닉층 수와 각 은닉층의 뉴런 수 | `[100, 100]` | 은닉층 수: `1~5`, 뉴런 수: `16`, `32`, `64`, `128`, `256`, `512` |
| `output_size` | 출력 클래스 수 | `10` | `10` (고정) |
| `activation` | 은닉층 활성화 함수 | `"relu"` | `"relu"`, `"sigmoid"` |
| `weight_init_std` | 가중치 초기화 방식 | `"he"` | `"he"` (activation=relu), `"xavier"` (activation=sigmoid)|
| `weight_decay_lambda` | L2 가중치 감쇠 계수 | `0` | `0`, `0.00001`, `0.0001`, `0.001`, `0.01` |
| `use_dropout` | Dropout 적용 여부 | `False` | `True`, `False` |
| `dropout_ration` | Dropout 비율 | 미사용 | `0.1`, `0.2`, `0.3`, `0.5` |
| `use_batchnorm` | Batch Normalization 적용 여부 | `False` | `True`, `False` |


## 2. 학습 파라미터

```python
trainer = Trainer(
    network=network,
    x_train=x_train,
    t_train=y_train,
    x_test=x_valid,
    t_test=y_valid,
    epochs=500,
    mini_batch_size=100,
    optimizer="adam",
    optimizer_param={
        "lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
    },
    evaluate_sample_num_per_epoch=None,
    verbose=False,
)
```

| 파라미터 | 의미 | 기본 설정 | Random Search 후보 |
|---|---|---|---|
| `epochs` | 최대 학습 에포크 수 | `500` | `500` (고정, 조기 종료 적용) |
| `mini_batch_size` | 미니배치 크기 | `100` | `64`, `128`, `256` |
| `optimizer` | 가중치 갱신 알고리즘 | `"adam"` | `"sgd"`, `"momentum"`, `"adagrad"`, `"adam"` |
| `optimizer_param["lr"]` | 학습률 | `0.001` | SGD/Momentum: `0.001`, `0.01`, `0.1`; AdaGrad: `0.001`, `0.01`; Adam: `0.0001`, `0.0003`, `0.001`, `0.003` |
| `optimizer_param["beta1"]` | Adam의 1차 모멘트 계수 | `0.9` | `0.9` (고정) |
| `optimizer_param["beta2"]` | Adam의 2차 모멘트 계수 | `0.999` | `0.999` (고정) |
| `evaluate_sample_num_per_epoch` | 에포크별 평가에 사용할 샘플 수 | `None` | `None` (고정) |
| `verbose` | 학습 로그 출력 여부 | `False` | `False` (고정) |

### Random Search 조합 제외 기준

제한된 탐색 횟수를 과도한 과소적합 가능성이 높은 조합에 사용하지 않기 위해, 다음 조합은 추출 단계에서 제외한다.

- 최소 은닉층 너비가 32 이하이면서 Dropout 비율이 0.5인 조합
- 가중치 감쇠 계수가 0.01이면서 Dropout 비율이 0.3 이상인 조합
- Batch Normalization 없이 Sigmoid 은닉층을 4개 이상 사용하는 조합

학습 중 손실이 `NaN` 또는 `inf`가 되면 해당 시행을 즉시 종료한다.
