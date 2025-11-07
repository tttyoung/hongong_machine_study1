# 과제

### Momentu

개념 : 경사하강법으로 이동할 때 관성을 부여해 최적화하는 기법. 이전 기울기의 크기를 고려하여 추가적으로 이동해, Local minimun에서 빠져나갈 수 있게함. 

해결하려는 문제 : 

1. Local minimum 문제 - 경사하강법은 시작위치에 따라 최적의 값이 달라져 Global minimun인지 Local minimum에 빠진건지 알 수 없음. 
2. saddle point (안장점) 문제 - 안장점은 기울기가 0이지만 극값이 아닌 지점을 의미하는데, 경사하강법은 미분이 0일 경우 파라미터 업데이트를 멈추기 때문에 이를 벗어나지 못하는 한계가 있음.

적용한 아이디어 : 외부에서 힘을 받지 않는 한 운동 상태를 지속하려는 성질인 ‘관성’ 을 활용함.

<img width="334" height="84" alt="image" src="https://github.com/user-attachments/assets/4f1a0556-8e55-4dcc-9c09-151b4bcee756" />

r : 관성계수 , n: 학습률

<img width="187" height="40" alt="image" src="https://github.com/user-attachments/assets/a0a861d2-db69-41b9-b446-9e94d742306c" />

### Adagrad

개념 : 특성별로 학습률을 다르게 조절하는 기법.

해결하려는 문제 : 각 특성마다 중요도가 다르기 때문에 모든 특성에 동일한 학습률을 적용하는 것이 비효율적임.

적용한 아이디어 : 가중치마다 기울기의 제곱합을 누적하여 학습률을 나눔.

<img width="381" height="177" alt="image" src="https://github.com/user-attachments/assets/d38f7085-6670-44ee-aee3-0d70a7a16234" />

### RMSProp

개념 : 특성별로 학습률을 다르게 조절하되, 최근 step의 기울기를 많이 반영하고 먼 과거의 step 기울기는 조금만 반영함.

해결하려는 문제 : Adagrad의 한계점인, 학습이 오래 진행될수록 학습률이 0에 가까워져 더 이상 학습이 진행되지 않는다는 점을 개선하기 위함.

적용한 아이디어 : 이전 step의 기울기를 지수이동평균을 활용해 업데이트해 먼 과거의 기울기 정보를 조금만 반영함.

<img width="405" height="120" alt="image" src="https://github.com/user-attachments/assets/8f9f03ce-823f-4b2a-986c-4a0926cd4653" />

gt : t번째 time step까지의 기울기 누적 크기, r : 지수이동평균의 업데이트 계수, e : 분모가 0이 되는 것을 방지하기 위한 아주작은 값 , n: 학습률

### Adam

개념 : Momentum + RMSProp 의 장점을 결합한 알고리즘.

해결할려는 문제 : RMSProp은 방향 정보가 없고 Momentum은 학습률 조정이 없는 단점을 상쇄함

적용한 아이디어 : 

<img width="426" height="283" alt="image" src="https://github.com/user-attachments/assets/6cfff3b0-a4b5-433f-b986-1268fc5c224c" />

### AdamW

개념 : Adam알고리즘의 L2규제 오류를 수정한 알고리즘

해결하려는 문제 : Adam의 L2규제가 학습률 조정에 섞여 올바른 효과를 내지 못함.

 적용한 아이디어 : 규제 항과 Adam업데이트 수식을 분리하여 적용함.
