# 07-1 | 인공 신경망

패션 MNIST 데이터 셋 불러오기

- 코드
    
    ```python
    import keras
    # 10종류의 패션 아이템으로 구성된 데이터 셋 다운로드
    (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
    # 28x28크기 60000개 이미지
    print(train_input.shape, train_target.shape)
    print(test_input.shape, test_target.shape)
    # 샘플 이미지 확인
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 10, figsize=(10,10))
    for i in range(10):
        axs[i].imshow(train_input[i], cmap='gray_r')
        axs[i].axis('off')
    plt.show()
    # 샘플 이미지 정답
    print(train_target[:10])
    # 레이블 당 샘플 개수
    import numpy as np
    print(np.unique(train_target, return_counts=True)) # 각 6000개 씩
    ```
    

### **로지스틱 회귀 (가장 간단한 인공 신경망) 로 패션 아이템 분류**

- 코드
    
    ```
    # 확률적 경사 하강법을 이용한 로지스틱 회귀 모델로 패션 아이템 분류
    # (=가장 간단한 인공 신경망)
    
    # 훈련 샘플 너무 많기 때문에 하나씩 꺼내서 모델 훈련하는 SGDClassifier 사용
    # SGD는 2차원 입력 불가, 1차원 배열로 변환
    train_scaled = train_input / 255.0
    train_scaled = train_scaled.reshape(-1, 28*28)
    # 784개의 픽셀(특성)로 이루어진 60000개의 샘플 준비 완료
    print(train_scaled.shape)
    # 교차 검증으로 성능 확인
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import SGDClassifier
    sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
    scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
    print(np.mean(scores['test_score']))
    # 반복 횟수 늘려도 성능 크게 향상되지않음.
    ```
    

로지스틱 회귀가 가장 간단한 인공 신경망인 이유는

로지스틱 회귀 구조가 인공 신경망과 동일하기 때문임.

( 가중치 × 입력(특성) + 절편을 계산해 그 결과를 소프트맥스(선형방정식계산함수)에 넣어 클래스를 예측 )

로지스틱 회귀 공식

<img width="415" height="44" alt="image" src="https://github.com/user-attachments/assets/db79ca0f-2a71-40c9-b2c7-f9effdbb46d3" />

선형 방정식

패션 MNIST의 레이블에 맞게 변형하면

<img width="389" height="26" alt="image 1" src="https://github.com/user-attachments/assets/aa3e28b2-db82-49ee-8323-3b3d2cf7213e" />
<img width="394" height="32" alt="image 2" src="https://github.com/user-attachments/assets/269651cb-ed78-426c-8f36-838459a9f3e8" />

10개의 클래스에 대한 선형 방정식을 계산한 다음 소프트맥스 함수를 통과해 각 클래스에 대한 확률을 얻음.

<img width="248" height="192" alt="image 3" src="https://github.com/user-attachments/assets/47a2ae31-b9ad-4505-97fb-eef6a3eafd14" />



### 인공 신경망

: 여러 개의 뉴런이 층 형태로 연결된 모델

입력층 : 데이터가 들어오는 곳. 특성 값 그 자체

출력층 : 최종 예측을 만드는 곳.

뉴런 : z값을 계산하는 단위

<img width="332" height="250" alt="image 4" src="https://github.com/user-attachments/assets/5bb19c2c-984d-4ca8-bc8b-3e83811297b9" />



**텐서플로** : 딥러닝 라이브러리. 

저수준 API(Low-level API) : 아주 기본적인 연산(텐서 만들기, 행렬곱, 미분, 그래프 구성)을 개발자가 직접 하나하나 코딩해야 하는 API ( ex. 엔진내부 )
고수준 API(Keras) : 모델을 간단한 명령어로 쉽게 만들도록 도와주는 API ( ex. 핸들, 엑셀 ) 

**케라스** : 텐서플로의 고수준 API. 인터페이스이기 때문에 연산은 백엔드(텐서플로)에게 맡긴다.

### **인공 신경망으로 모델 만들기 & 패션 아이템 분류하기**

- 코드
    
    ```python
    # 인공신경망으로 모델 만들기
    # 인공신경망에서는 검증 세트 사용함. (로지스틱회귀는 교차 검증)
    # train 80% -> train 60% + validation 20%
    from sklearn.model_selection import train_test_split
    train_scaled, val_scaled, train_target, val_target = train_test_split(
        train_scaled, train_target, test_size=0.2, random_state=42)
    print(train_scaled.shape, train_target.shape)
    print(val_scaled.shape, val_target.shape)
    
    # 입력층 정의, =784개의 픽셀 (특성)
    inputs = keras.layers.Input(shape=(784,))
    
    # 출력층(밀집층) 정의, 10개의 레이블 (정답)
    # 매개변수는 순서대로 '뉴런 개수' '뉴런 출력에 적용할 함수'
    # 활성화 함수인 소프트맥스 함수 적용
    dense = keras.layers.Dense(10, activation='softmax')
    # 신경망 모델
    model = keras.Sequential([inputs, dense])
    ```
    

<img width="330" height="206" alt="image 5" src="https://github.com/user-attachments/assets/bc8ac350-e54d-410f-a539-93ec34acd230" />



### 인공 신경망으로 패션 아이템 분류

- 코드
    
    ```python
    # 인공신경망으로 패션 아이템 분류하기
    
    # compile은 인공신경망을 훈련하기 전 설정 단계.
    # 손실 함수를 정수로 된 타겟값을 사용해 크로스 엔트로피 손실을 계산하는
    # 'sparse_categorical_crossentropy' 로 정한다.
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(train_target[:10])
    model.fit(train_scaled, train_target, epochs=10)
    model.evaluate(val_scaled, val_target)
    ```
    

다중 분류에서는 크로스 엔트로피 손실 함수 (categorical_crossentropy) 사용함.

sparse라는 단어가 붙은 이유는?

→다중 분류에서 크로스 엔트로피 손실 함수를 사용하려면 타겟값을 해당 클래스만 1이고 나머지는 모두 0으로 바꿔야하는 ‘원-핫 인코딩’ 형태로 바꿔야함. 

→그런데 케라스의 sparse_categorical_crossentropy 에서는 정수 타겟값을 그대로 사용할 수 있음.

ex.

타겟이 다음과 같다고 하자:

```
y = [2, 0, 1, 2]
```

클래스 개수는 3개라고 하자.

원-핫으로 바꾸면:

```
2 → [0,0,1]
0 → [1,0,0]
1 → [0,1,0]
2 → [0,0,1]
```

즉,

```
y_onehot =
[
 [0,0,1],
 [1,0,0],
 [0,1,0],
 [0,0,1]
]
```
