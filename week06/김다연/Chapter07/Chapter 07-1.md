# Chapter 07-1

1. **인공 신경망 (artificial neural network)** = 딥러닝
    1. 뉴런(유닛) : z 값을 계산하는 단위
    2. **출력층** : 신경망의 **최종 값**을 만듦
    3. **입력층** : **픽셀값** 자체, 특별한 계산 수행 x
    4. 교차 검증 잘 수행 x → 검증 세트 별도로 덜어내어 사용
        1. 딥러닝 분야의 데이터셋은 충분히 크기 때문에 검증 점수 안정적
        2. 교차 검증을 수행하기에 훈련 시간 너무 오래 걸림
    5. 사진
        
        ![image.png](image.png)
        
    6. 구조도
        
        ![image.png](image%201.png)
        

1. **텐서플로 (TensorFlow)**
    1. 구글이 오픈소스로 공개한 딥러닝 라이브러리
    2. 알파고 → 관심 높아짐
    3. **케라스 (Keras)**
        1. 고수준 API
        2. **GPU**를 사용하여 인공 신경망 훈련 → 행렬 연산 최적화
        3. **밀집층 (dense layer)** = 완전 연결층 : 말 그대로 빽빽함

1. **활성화 함수 (activation function)**
    1. 뉴런의 선형 방정식 **계산 결과에 적용되는 함수**
    2. ex) 소프트맥스

1. **one-hot encoding** : 타깃값을 **해당 클래스만 1**이고 **나머지는 모두 0**인 배열로 만드는 것

1. 실습
    - 코드
        
        ```python
        from tensorflow import keras
        
        (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
        ```
        
        ```python
        print(train_input.shape, train_target.shape)
        print(test_input.shape, test_target.shape)
        ```
        
        ```python
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(1, 10, figsize=(10,10))
        for i in range(10):
            axs[i].imshow(train_input[i], cmap='gray_r')
            axs[i].axis('off')
        plt.show()
        
        print([train_target[i] for i in range(10)])
        ```
        
        ```python
        import numpy as np
        
        print(np.unique(train_target, return_counts=True))
        ```
        
        ```python
        train_scaled = train_input / 255.0
        train_scaled = train_scaled.reshape(-1, 28*28)
        
        print(train_scaled.shape)
        ```
        
        ```python
        from sklearn.model_selection import cross_validate
        from sklearn.linear_model import SGDClassifier
        
        # sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
        sc = SGDClassifier(loss='log_loss', max_iter=20, random_state=42) # 반복 횟수 늘려도 딱히 성능 향상 x
        
        scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
        print(np.mean(scores['test_score']))
        ```
        
        ```python
        import tensorflow as tf
        from tensorflow import keras
        
        from sklearn.model_selection import train_test_split
        
        train_scaled, val_scaled, train_target, val_target = train_test_split(
            train_scaled, train_target, test_size=0.2, random_state=42)
        
        print(train_scaled.shape, train_target.shape)
        print(val_scaled.shape, val_target.shape)
        ```
        
        ```python
        dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
        model = keras.Sequential([dense])
        ```
        
        ```python
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print(train_target[:10])
        model.fit(train_scaled, train_target, epochs=5)
        model.evaluate(val_scaled, val_target)
        ```
        
    
    [07_1.ipynb](07_1.ipynb)