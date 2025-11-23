# Chapter 7-3

### 검증손실

- 에포크가 늘어날 수록 손실은 줄어들지만 test set에 대해서만 확인하면 과대적합한 모델이 될 가능성이 크다.
- 아래 그래프를 확인해보면 에포크가 5이상부터는 validation set의 loss가 높아지는것을 확인할 수 있다.

<img width="652" height="436" alt="image" src="https://github.com/user-attachments/assets/44b0f090-9ad7-47f3-9da1-658abae1489f" />

- 이러한 과대적합을 막기 위해 옵티마이저 하이퍼파라미터를 조정하는 방법이 있다. ex) RMSprop, Adam

### 드롭아웃

- 훈련과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서 과대적합 방지(검증, 평가에는 X)
    
    → 특정 뉴런에 과대하게 의존하는것을 막고 모든 입력에 대해 주의를 기울여야함.
    
- 일부 뉴런의 출력을 0으로 만들지만 전체 출력 배열의 크기는 변하지 않음
- 텐서플로와 케라스는 모델을 평가와 예측에 사용할 때는 자동으로 드롭아웃을 적용하지 않는다.

<img width="988" height="410" alt="image 1" src="https://github.com/user-attachments/assets/aacb9736-f301-466a-9b36-bb2068680327" />


<img width="658" height="426" alt="image 2" src="https://github.com/user-attachments/assets/259c8174-3a21-4d0d-94d1-0f2e17b0b906" />

- dropout을 사용했을때 10번째 에포크까지 loss가 줄어들고 그 뒤로는 크게 변동되지 않음.

- 위처럼 10번의 에포크가 적절하다는것을 찾으면 20번의 에포크를 한 모델을 과대적합되지 않을 에포크만큼 다시 훈련해야한다.

### 콜백

- 훈련 과정 중간에 어떤 작업을 수행할 수 있게 하는 객체로 keras.callbacks 패키지 아래에 있는 클래스들
- ModelCheckpoint 콜백을 사용하여 에포크마다 모델을 저장하고 save_best_only=True 매개변수를 지정하여 가장 낮은 검증 점수를 만드는 모델을 저장할 수 있다.
- But 이대로만 한다면 저장만 하고 똑같이 20번의 에포크를 실시

### 조기종료

- 훈련 에포크 횟수를 제한하여 모델이 과대적합되는것을 방지한다.
- EarlyStopping 콜백을 사용하여 저장된 가장 낮은 검증점수보다 상승할때 훈련을 정지

[혼공머7_3 (1).ipynb](%ED%98%BC%EA%B3%B5%EB%A8%B87_3_(1).ipynb)
