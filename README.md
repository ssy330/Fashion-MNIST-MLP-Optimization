# Fashion-MNIST MLP Optimization

《밑바닥부터 시작하는 딥러닝 1》의 4~6장 범위에서 **Fashion-MNIST 분류 성능을 개선할 다층 퍼셉트론(MLP)**을 탐색하는 3인 팀 프로젝트입니다.

단순히 최고 정확도를 보고하는 것뿐 아니라, **모델 구조·학습 알고리즘·일반화 기법을 통제 실험으로 비교**하고 최종 모델 선택의 근거를 제시합니다.

## 과제 조건

- 데이터: Fashion-MNIST (28×28 흑백 이미지, 10개 클래스, train 60,000장 / test 10,000장)
- 범위: 교재 **4, 5, 6장**에 소개된 방법만 사용; **6층 이하 신경망**
- 실험: 구조 결정 비교 실험, **Randomized Search** 등


## 저장소 구성

```text
fashion-mnist-mlp-optimization/
├── common/                      # 교재 공식 신경망 구현
├── data/                        # Fashion-MNIST
├── docs/
│   ├── 00_random_search_experiment.md # Random Search 실험
│   ├── 01_mlp_architecture_experiment.md # MLP 구조 실험
│   └── parameter.md             # 모델 및 학습 파라미터 참고
├── experiments/
│   ├── 00_random_search.py      # 하이퍼파라미터 Random Search
│   ├── 01A_width_depth.py       # 너비·깊이 구조 실험
│   ├── 01B_parameter_budget_depth.py # 파라미터 예산·깊이 실험
│   ├── 01C_neuron_layout.py     # 뉴런 배치 형태 실험
│   └── architecture_experiment_base.py # 구조 실험 공통 로직
├── results/
    ├── 00_random_search/
    │   ├── 00_random_search_summary.csv
    │   └── runs/                # 시행별 history, best model, checkpoint
    └── 01C_neuron_layout/
        ├── 01C_neuron_layout_summary.csv
        ├── 01C_neuron_layout_neuron_layout_accuracy.png
        └── runs/                # 구조별 history, best model, checkpoint
├── README.md                    # 프로젝트 개요 및 공통 사용 규칙
└── requirements.txt             # Python 의존성
```

## 참고 자료

- [《밑바닥부터 시작하는 딥러닝 1》 한국어 예제 코드](https://github.com/WegraLee/deep-learning-from-scratch)
- [원서 공식 예제 코드](https://github.com/oreilly-japan/deep-learning-from-scratch)
- [Fashion-MNIST 공식 GitHub](https://github.com/zalandoresearch/fashion-mnist)
