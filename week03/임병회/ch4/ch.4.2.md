# 확률적 경사 하강법

## 점진적인 학습
점진적 학습 - 앞서 훈련한 모델을 버리지 않고 새로운 데이터에 대해서만 조금씩 더 훈련<br>

### 확률적 경사 하강법
훈련 세트에서 랜덤하게 하나의 샘플을 선택하여 가파른 경사를 조금 내려감 until 전체 샘플 모두 사용, 원하는 위치까지 반복<br>
에포크 - 확률적 경사 하강법에서 훈련 세트를 한 번 모두 사용하는 과정<br>
미니배치 경사 하강법 - 1개씩이 아닌 여러개 샘플을 사용해 경사 하강법 수행<br>
배치 경사 하강법 - 전체 샘플 이용해 한 번 경사로 이동, 데이터 많을 때 과부하<br>
신경망 알고리즘에서 이용된다. <br>
<img width="810" height="517" alt="image" src="https://github.com/user-attachments/assets/26624e30-8f4b-40c3-9ef9-439b8608c2a6" />

### 손실 함수
머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준<br>

### 로지스틱 손실 함수(=이진 크로스엔트로피 손실 함수)
<img width="511" height="217" alt="image" src="https://github.com/user-attachments/assets/b7952f69-27b5-42df-8ae2-e6911b2cd0b1" />

<img width="621" height="236" alt="image" src="https://github.com/user-attachments/assets/d89c0ffc-9eb1-4527-88ad-c367e77417c4" />

1에서 멀어질수록 손실<br>
크로스엔트로피 손실 함수 - 다중 분류에서 사용하는 손실 함수<br>
평균 제곱 오차 - 회귀에서 사용하는 손실 함수<br>


## SGDClassifier

```python
#확률적 경사 하강법 클래스 이용
from sklearn.linear_model import SGDClassifier
#점수 출력
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
#점수낮음->이어서 훈련
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

## 에포크와 적합
<img width="461" height="334" alt="image" src="https://github.com/user-attachments/assets/8a94d18b-b9f0-4ab2-996f-51011d34ed11" />

조기종료 - 과대적합 시작 전에 훈련을 멈추는 것<br>

```python
# 에포크 300회 반복
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
#그래프 그리기
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```
<img width="598" height="397" alt="image" src="https://github.com/user-attachments/assets/9ec42b50-1f53-4340-b413-664bb274ff9c" />

->100번째 에포크가 적절해보임<br>

```python
# 에포크 100으로 훈련
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

Q1. 자동으로 최적 에포크를 찾아 훈련 할 수는 없을까?<br>
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 조기 종료 설정: 검증 손실이 5번 연속 개선되지 않으면 훈련 중단
early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)

# 2. 최적 모델 자동 저장 설정: 검증 손실이 가장 낮을 때마다 모델을 파일에 저장
model_checkpoint_cb = ModelCheckpoint('best_model.h5', save_best_only=True)

# 3. 넉넉한 에포크로 훈련 시작
# max_iter 대신 epochs 사용, 충분히 큰 값(예: 1000)을 줌
model.fit(train_scaled, train_target, epochs=1000, 
          validation_data=(val_scaled, val_target),
          callbacks=[early_stopping_cb, model_checkpoint_cb])

# 훈련이 끝나면, model 객체는 restore_best_weights=True 덕분에
# 자동으로 최상의 성능을 냈던 시점의 가중치를 가지게 됩니다.
# best_model.h5 파일을 불러와도 동일합니다.
```


