# Random Search 실험 결과 분석

## 1. 학습 시간 분석

전체 학습 시간은 **Epoch당 학습 시간 × 실제 학습 Epoch 수**에 의해 결정됨.

- **Epoch당 학습 시간:** 모델의 파라미터 수가 많을수록 증가하는 경향
- **전체 Epoch 수:** Optimizer와 Learning Rate의 영향을 크게 받는 경향
- **전체 학습 시간:** 파라미터 수가 많거나 수렴 속도가 느린 모델에서 증가

## 2. 성능 분포 분석

100개 모델의 검증 정확도를 기준으로 5개 그룹으로 분류함.

| 구간 | 정확도 범위 (%) | 모델 수 | 분류 |
|---|---|---:|---|
| Group 1 | 34.36 ~ 82.10 | 10 | 최하위 |
| Group 2 | 83.69 ~ 85.61 | 20 | 하위 |
| Group 3 | 85.83 ~ 88.07 | 40 | 중위 |
| Group 4 | 88.11 ~ 88.86 | 20 | 상위 |
| Group 5 | 88.89 ~ 90.26 | 10 | 최상위 |
| **전체** | **34.36 ~ 90.26** | **100** | - |

- 전체 정확도 범위: **34.36~90.26%**
- 60개 모델이 85.83~88.86% 구간에 분포
- 최고 검증 정확도: **90.26%**

## 3. 상위·하위 모델 파라미터 비교

### 3.1 상위 10개 vs 하위 10개

| 파라미터 | 상위 10개 | 하위 10개 |
|---|---:|---:|
| ReLU / He | 9개 | 2개 |
| Sigmoid / Xavier | 1개 | 8개 |
| Weight Decay ≤ 0.0001 | 8개 | 1개 |
| Weight Decay ≥ 0.001 | 2개 | 9개 |
| BatchNorm 적용 | 6개 | 0개 |
| AdaGrad / LR 0.01 | 5개 | 0개 |
| LR ≤ 0.001 | 2개 | 8개 |

### 3.2 상위 30개 vs 하위 30개

| 파라미터 | 상위 30개 | 하위 30개 |
|---|---:|---:|
| ReLU / He | 24개 (80.0%) | 9개 (30.0%) |
| Sigmoid / Xavier | 6개 (20.0%) | 21개 (70.0%) |
| Weight Decay ≤ 0.0001 | 19개 (63.3%) | 11개 (36.7%) |
| Weight Decay ≥ 0.001 | 11개 (36.7%) | 19개 (63.3%) |
| BatchNorm 적용 | 19개 (63.3%) | 10개 (33.3%) |
| AdaGrad / LR 0.01 | 11개 (36.7%) | 1개 (3.3%) |
| LR ≤ 0.001 | 12개 (40.0%) | 20개 (66.7%) |

### 3.3 주요 관찰 결과

- **Activation / Initialization:** ReLU/He 조합이 상위 모델에서 높은 빈도로 나타남.
- **Weight Decay:** 낮은 Weight Decay(≤0.0001)가 상위 모델에 상대적으로 많이 분포함.
- **BatchNorm:** 상위 모델에서 적용 비율이 높게 나타남.
- **Optimizer / Learning Rate:** AdaGrad와 LR=0.01 조합이 상위 모델에 집중됨.
- **낮은 Learning Rate:** LR≤0.001 설정은 하위 모델에서 상대적으로 많이 나타남.

## 4. 파라미터 상호작용 분석

### 4.1 Activation × BatchNorm

| Activation | BatchNorm | 실험 수 | 평균 정확도 | 중앙값 정확도 |
|---|---|---:|---:|---:|
| ReLU | True | 25 | 87.91% | 88.40% |
| ReLU | False | 26 | 86.64% | 87.55% |
| Sigmoid | True | 31 | 86.81% | 87.26% |
| Sigmoid | False | 18 | 79.14% | 84.25% |

**분석 결과**
- ReLU는 BatchNorm 적용 여부에 따른 성능 차이가 상대적으로 작음.
- Sigmoid는 BatchNorm 미적용 시 성능이 크게 저하되는 경향.
- BatchNorm은 특히 Sigmoid에서 성능 저하를 완화할 가능성이 있음.

### 4.2 Activation × BatchNorm × Weight Decay

| Activation | BatchNorm | 낮은 WD 중앙값 | 높은 WD 중앙값 | 차이 |
|---|---|---:|---:|---:|
| ReLU | False | 87.82% | 85.61% | +2.21%p |
| ReLU | True | 88.28% | 88.59% | -0.31%p |
| Sigmoid | False | 86.79% | 76.45% | +10.34%p |
| Sigmoid | True | 87.28% | 86.54% | +0.74%p |

**분석 결과**
- BatchNorm 미적용 시 높은 Weight Decay가 성능 저하와 관련되는 경향.
- 특히 Sigmoid + BatchNorm 미적용 조합에서 **10.34%p**의 차이 발생.
- BatchNorm 적용 시 Weight Decay에 따른 성능 차이가 감소함.
- Weight Decay의 효과는 Activation 및 BatchNorm 설정에 따라 달라질 가능성이 있음.

### 4.3 Activation × BatchNorm × Dropout

| Activation | BatchNorm | 미적용 중앙값 | 적용 중앙값 | 차이 |
|---|---|---:|---:|---:|
| ReLU | False | 88.00% | 86.18% | +1.82%p |
| ReLU | True | 88.44% | 87.70% | +0.74%p |
| Sigmoid | False | 85.09% | 81.24% | +3.85%p |
| Sigmoid | True | 87.32% | 86.54% | +0.78%p |

**분석 결과**
- 모든 조합에서 Dropout 미적용 모델의 중앙값이 더 높음.
- 특히 Sigmoid + BatchNorm 미적용 조합에서 **3.85%p**의 차이 발생.
- BatchNorm 적용 시 Dropout에 따른 성능 차이가 감소함.
- 현재 실험에서는 Dropout의 뚜렷한 성능 개선 효과가 관찰되지 않음.

## 5. 종합 결론

| 분석 항목 | 주요 결과 |
|---|---|
| 학습 시간 | 파라미터 수와 학습 Epoch 수의 영향 |
| Activation / Init | ReLU / He 조합이 상위 성능에 집중 |
| Weight Decay | 낮은 WD가 상대적으로 유리 |
| BatchNorm | 특히 Sigmoid에서 성능 저하 완화 경향 |
| Optimizer / LR | AdaGrad / 0.01 조합이 상위 모델에 집중 |
| Dropout | 적용 시 검증 정확도 중앙값이 낮아지는 경향 |
| 최고 성능 | 검증 정확도 90.26% |

### 후속 실험 방향

- 상위 성능 모델을 중심으로 파라미터 변경 실험 진행
- 다른 조건을 고정하고 하나의 파라미터만 변경하는 통제 실험
- ReLU/He, BatchNorm, 낮은 Weight Decay, AdaGrad/LR=0.01 조합 우선 검토
- 개별 파라미터의 성능 개선 효과 및 상호작용 검증

**주의:** Random Search 결과는 여러 파라미터가 동시에 달라지는 관찰적 비교이므로, 특정 설정이 성능 차이의 직접적인 원인이라고 단정할 수 없음.
