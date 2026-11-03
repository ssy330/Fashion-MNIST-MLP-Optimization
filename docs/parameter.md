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

| 파라미터 | 의미 | 구조 실험 기본 설정 |
|---|---|---|
| `input_size` | 입력 벡터의 차원 | `784` |
| `hidden_size_list` | 은닉층 수와 각 은닉층의 뉴런 수 | 실험 구조에 따라 변경 |
| `output_size` | 분류 대상 클래스 수 | `10` |
| `activation` | 은닉층 활성화 함수 | `"relu"` |
| `weight_init_std` | 가중치 초기화 방식 | `"he"` |
| `weight_decay_lambda` | L2 가중치 감쇠 계수 | `0` |
| `use_dropout` | Dropout 적용 여부 | `False` |
| `dropout_ration` | Dropout 비율 | Dropout 미사용으로 적용하지 않음 |
| `use_batchnorm` | Batch Normalization 적용 여부 | `False` |

`hidden_size_list`는 모델 구조를 결정하는 핵심 파라미터이다. 예를 들어 `[128, 128, 128]`은 각 128개 뉴런으로 구성된 은닉층 3개를 의미한다.

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

| 파라미터 | 의미 | 구조 실험 기본 설정 |
|---|---|---|
| `epochs` | 최대 학습 에포크 수 | `500` |
| `mini_batch_size` | 한 번의 파라미터 갱신에 사용하는 샘플 수 | `100` |
| `optimizer` | 가중치 갱신 알고리즘 | `"adam"` |
| `optimizer_param["lr"]` | 학습률 | `0.001` |
| `optimizer_param["beta1"]` | Adam의 1차 모멘트 계수 | `0.9` |
| `optimizer_param["beta2"]` | Adam의 2차 모멘트 계수 | `0.999` |
| `evaluate_sample_num_per_epoch` | 에포크별 평가에 사용할 샘플 수 | `None` (전체 검증 데이터) |
| `verbose` | 학습 로그 출력 여부 | `False` |
