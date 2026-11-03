# Fashion-MNIST MLP Optimization

《밑바닥부터 시작하는 딥러닝 1》의 4~6장 범위에서 **Fashion-MNIST 분류 성능을 개선할 다층 퍼셉트론(MLP)**을 탐색하는 3인 팀 프로젝트입니다.

단순히 최고 정확도를 보고하는 것뿐 아니라, **모델 구조·학습 알고리즘·일반화 기법을 통제 실험으로 비교**하고 최종 모델 선택의 근거를 제시합니다.

## 과제 조건

- 데이터: Fashion-MNIST (28×28 흑백 이미지, 10개 클래스, train 60,000장 / test 10,000장)
- 범위: 교재 **4, 5, 6장**에 소개된 방법만 사용; **6층 이하 신경망**
- 실험: 구조 결정 비교 실험, 수동 탐색 이후 **Randomized Search 필수**
- 데이터 사용: 학습·튜닝 과정에서는 학습/검증 데이터만 사용하고 **테스트 데이터는 최종 평가에만 사용**
- 최종 제출: 팀별 최선의 모델 1개, 코드·PPT, Competition에서 실행 가능한 `network_Team{번호}.pkl` (정확한 파일명·실행 형식은 수업 공지에 맞춤)
- 팀원 **전원 발표**

## 저장소 구성

```text
fashion-mnist-mlp-optimization/
├── README.md                    # 프로젝트 개요 및 공통 사용 규칙
├── docs/
│   ├── 01_mlp_architecture_experiment.md # MLP 구조 실험 계획
│   └── parameter.md             # 모델 및 학습 파라미터 참고
├── requirements.txt             # Python 의존성
├── common/                      # 교재 공식 신경망 구현
├── data/                        # Fashion-MNIST
└── experiments/
    ├── architecture_experiment_base.py # 구조 실험 공통 로직
    ├── A01_*.py                 # A: 신경망 구조 실험
    ├── B01_*.py                 # B: 학습 알고리즘 실험
    └── C01_*.py                 # C: 일반화 기법 실험
```

## 참고 자료

- 과제 안내: 수업에서 배포한 「우리들의 신경망 모델」(2026 신경망 팀플)
- [《밑바닥부터 시작하는 딥러닝 1》 한국어 예제 코드](https://github.com/WegraLee/deep-learning-from-scratch)
- [원서 공식 예제 코드](https://github.com/oreilly-japan/deep-learning-from-scratch)
- [Fashion-MNIST 공식 GitHub](https://github.com/zalandoresearch/fashion-mnist)
