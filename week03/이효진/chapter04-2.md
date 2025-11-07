# 04-2 | 확률적 경사 하강법

1. **stocastic gradient descent** : train set에서 랜덤하게 하나의 샘플을 골라 gradient descent를 수행

**epoch** : train set을 한 번 모두 수행하는 것

1. **minibatch gradient descent** : train set에서 여러 개의 샘플을 골라 gradient descent를 수행
2. **batch gradient descent** : train set 전체 샘플을 사용

### 손실함수 ( **loss function )**

 : 어떤 문제에서 모델이 얼마나 틀렸는지 측정하는 기준. 미분 가능해야함.

1. **logistic loss function** : 이진 분류 손실함수. 
2. **cross-entropy loss function** : 다중 분류 손실함수

### **에포크와 과대/과소적합 관계**

에포크 횟수가 너무 적으면 모델이 train set을 덜 학습하는 과소적합

에포크 횟수가 너무 많으면 모델이 train set에 너무 잘 맞는 과대적합

조기 종료 : 과대적합이 시작되기 전에 훈련을 멈추는 것

SGDClassifier의 loss매개변수의 기본값은 hinge임.