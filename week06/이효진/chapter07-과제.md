# 과제

### 딥러닝 훈련 기술을 조사하고 각각의 역할과 사용법을 정리하기.
(배치 정규화, 학습률 스케줄링, 데이터 증강, 가중치 초기화 등..)

문제점 : internal covariate shift현상으로 각 층마다 출력값의 데이터 분포가 달라져 신경망이 깊어질수록 변형이 누적되어 학습 성능이 떨어진다.

1. **배치 정규화** : 훈련 중 각 층에 들어가는 활성화 값을 미니배치 단위로 평균 0, 분산 1 이 되도록 정규화하는 기법. 
    
    역할
    
    - gradient의 크기, 초기값에 대한 의존도가 감소해서 학습률을 높게 사용할 수 있게 되어 빠르고 안정적인 학습이 가능해진다.
    - 과대적합을 줄여준다.
    
    사용법 
    
    → keras.layers.BatchNormalization()
    
    ```python
    model = keras.Sequential([
        keras.layers.Dense(100, use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
    ])
    ```
    

문제 : 훈련 내내 딱 맞는 하나의 학습률이라는게 거의 존재하지 않는다.

1. **학습률 스케줄링** : 훈련이 진행될수록 학습률(한 번 업데이트할 때 가중치를 얼마나 크게 움직일지)을 자동 조절하는 기법
    
    역할 
    
    - 초반에는 빠른 학습(학습률 크게), 후반에는 미세 조정(학습률 작게)하여 수렴할 수 있게 해준다.
    
    사용법
    
    1. keras.optimizers.schedules.ExponentialDecay : 일정 비율(지수적)로 감소
        
        ```python
        Exponential = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01, # 처음 학습률
            decay_steps=1000, # 몇 스텝마다
            decay_rate=0.9 # 어느정도 비율로 줄일지
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        ```
        
    2. keras.callbacks.ReduceLROnPlateau : 정체 시 학습률 감소
        
        ```python
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # 무엇을 기준으로 볼지 (검증 손실)
            factor=0.5,          # 학습률 절반으로 줄이기
            patience=3,          # 3 epoch 동안 개선 없으면 실행
            min_lr=1e-6          # 너무 작아지지 않게 하한 설정
        )
        model.fit(..., callbacks=[lr_reducer])
        ```
        
2. **데이터 증강** : 훈련 데이터에 랜덤 변형(회전, 이동, 뒤집기)을 적용해 데이터 양을 인위적으로 늘리는 기법
    
    역할
    
    - 훈련 데이터가 많아져 특정 패턴을 외울 수 없고 다양한 변형에도 강한 일반적인 패턴을 배워 과대적합을 크게 감소시킨다.
    
    사용법
    
    1. keras.preprocessing.image.ImageDataGenerator
        
        ```python
        
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,      # 0~20도 회전
            width_shift_range=0.1,  # 좌우 이동
            height_shift_range=0.1, # 상하 이동
            shear_range=0.2,        # 기울이기
            zoom_range=0.2,         # 확대/축소
            horizontal_flip=True,   # 좌우 반전
            fill_mode='nearest'     # 빈 공간 채우기
        )
        
        datagen.fit(train_images)
        model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                  epochs=20)
        ```
        
    2. keras.layers의 Augmentation Layer. 모델안에 층으로 바로 넣을 수 있음.
        
        ```cpp
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"), # 왼쪽/오른쪽으로 뒤집을지 랜덤으로 결정
            layers.RandomRotation(0.1), # 회전
            layers.RandomZoom(0.1) # 확대/축소
        ])
        
        model = keras.Sequential([
            data_augmentation, # 원본 이미지 변형 시킴
            layers.Conv2D(32, (3,3), activation='relu'),
            ...
        ])
        ```
        
3. **가중치 초기화** : 적절한 분포에서 가중치를 초기화
    
    역할
    
    - Gradient vanishing/exploding 방지
    - 가중치를 적절한 범위에서 시작시켜 효율적인 최적화와 빠른 수렴을 가능하게 한다.
    - 활성화 함수 반복으로 출력 분포가 한쪽으로 쏠리지 않도록 막아 안정적인 흐름을 유지한다.
    
    사용법
    
    1. He 초기화 ( 활성화 함수가 ReLU와 같을 때)
        
        ```python
        layers.Dense(128, activation='relu',
                     kernel_initializer='he_normal')
        ```
        
    2. Xavier 초기화 (활성화 함수가 시그모이드와 같을 때)
        
        ```cpp
        layers.Dense(128, activation='tanh',
                     kernel_initializer='glorot_uniform')
        ```