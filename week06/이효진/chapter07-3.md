# 07-3 | 신경망 모델 훈련

x-epoch , y-loss로 검증 세트 & 훈련 세트 손실 곡선을 그려보면

- 코드
    
    ```python
    # 손실 곡선
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    (train_input, train_target), (test_input, test_target) = \
        keras.datasets.fashion_mnist.load_data()
    train_scaled = train_input / 255.0
    train_scaled, val_scaled, train_target, val_target = train_test_split(
        train_scaled, train_target, test_size=0.2, random_state=42)
    # 모델을 함수로 정의 
    def model_fn(a_layer=None):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(28,28)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation='relu'))
        # 케라스 층을 추가하면 은닉층 뒤에 또 하나의 층을 추가하는 if문
        if a_layer:
            model.add(a_layer)
        model.add(keras.layers.Dense(10, activation='softmax'))
        return model
    model = model_fn()
    model.summary()
    # fit 메서드의 결과를 history 변수에 담음.
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
    # history 객체에 훈련 측정값(정확도,손실)이 담겨있는 딕셔너리가 들어 있음.
    print(history.history.keys())
    
    # 그래프로 표현
    # 손실
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    # 정확도
    plt.plot(history.history['accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    # 에포크 늘리기
    model = model_fn()
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
    plt.plot(history.history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    ```
    

<img width="307" height="186" alt="image" src="https://github.com/user-attachments/assets/1579411c-352f-49b4-ae54-e6cb89d1350d" />


train loss는 계속 감소하는 반면, validation loss는 어느 시점부터 증가하는 과대적합 모델이 되는 것을 확인할 수 있다.

**규제 방법** 

1. 옵티마이저를 Adam으로 변경 (Adam은 RMSProp의 단점을 보완한 개선 버전이므로)
2. **dropout (드롭아웃)** 
    
    : 훈련 과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서 출력을 0으로 만드는 기법
    
    <img width="413" height="189" alt="image 1" src="https://github.com/user-attachments/assets/3e04cda7-9a2e-4f2c-98b6-f98b751f7a93" />

    
    드롭아웃 비율은 하이퍼파라미터로 직접 정해줘야함.
    
    | 드롭아웃 비율 | 장점 | 단점 |
    | --- | --- | --- |
    | **높다 (0.5)** | 과대적합 강력 방지 | 학습 어려워짐, 성능 떨어질 수 있음 |
    | **중간 (0.3)** | 균형 좋음, 가장 많이 사용 | 상황 따라 부족/과함 |
    | **낮다 (0.1)** | 학습 잘됨, 빠름 | 과대적합 방지 약함 |
    
    왜 드롭아웃이 과대적합을 막냐?
    
    1. 뉴런이 랜덤하게 꺼지면 특정 뉴런, 패턴에 의존하는 것을 줄이고 좀 더 일반적으로 생각할 수 있도록 함.
    2. 앙상블 학습과 비슷함. 
        
        앙상블은 여러 모델을 뽑아서 그 예측들을 평균내서 성능을 좋게하는 방법인데, 드롭아웃을 보면 학습 중에 랜덤으로 여러 ‘작은 모델들’을 만들고, 테스트할 때는   그 모델들의 평균을 내기 때문에 앙상블 학습과 비슷하다고 할 수 있음.
        
    
    손실 곡선을 드롭아웃을 적용해서 다시 그려보면
    
    - 코드
        
        ```python
        # 2. dropout
        # 모델 층에 드롭아웃 추가 
        model = model_fn(keras.layers.Dropout(0.3)) # 꺼버릴 뉴런의 비율
        model.summary()
        model.compile(optimizer='adam', 
        							loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        history = model.fit(train_scaled, train_target, 
        										epochs=20, verbose=0,
                            validation_data=(val_scaled, val_target))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        ```
        
    
    <img width="301" height="195" alt="image 2" src="https://github.com/user-attachments/assets/e5efa857-e56b-4875-93b2-32f57551a180" />

    
    과대적합이 준 것을 확인할 수 있음.
    

### 모델 저장과 복원

- 코드
    
    ```python
    # 모델 저장과 복원
    # 에포크 11으로 재훈련
    model = model_fn(keras.layers.Dropout(0.3))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_scaled, train_target, epochs=11, verbose=0,
                        validation_data=(val_scaled, val_target))
    # 모델&파라미터 저장
    model.save('model-whole.keras')
    # 파라미터만 저장
    model.save_weights('model.weights.h5')
    !ls -al model*
    # 새로운 모델+파라미터만 저장된 파일
    model = model_fn(keras.layers.Dropout(0.3))
    model.load_weights('model.weights.h5')
    import numpy as np
    val_labels = np.argmax(model.predict(val_scaled), axis=-1)
    print(np.mean(val_labels == val_target))
    model = keras.models.load_model('model-whole.keras')
    model.evaluate(val_scaled, val_target)
    ```
    

### callback(콜백)

: 훈련 과정 중간에 어떤 작업을 수행할 수 있게 하는 객체. 

1. 최상의 모델을 자동으로 저장해주는 ModelCheckpoint
    - 코드
        
        ```python
        # best-model.keras에 최상의 검증 점수를 낸 모델이 저장됨.
        model = model_fn(keras.layers.Dropout(0.3))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # ModelCheckpoint에서 save_best_only=True로 설정
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras',
                                                        save_best_only=True)
        model.fit(train_scaled, train_target, epochs=20, verbose=0,
                  validation_data=(val_scaled, val_target),
                  # 콜백에 객체 넣어주기
                  callbacks=[checkpoint_cb])
        model = keras.models.load_model('best-model.keras')
        model.evaluate(val_scaled, val_target)
        
        ```
        
2. 과대적합이 시작되기 전 훈련을 미리 정지하는 조기종료 EarlyStopping
    - 코드
        
        ```python
        # 과대적합 시작 전 훈련 종료하는 '조기종료' 콜백도 같이 사용
        model = model_fn(keras.layers.Dropout(0.3))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras',
                                                        save_best_only=True)
        # EarlyStopping에서 
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                          restore_best_weights=True)
        history = model.fit(train_scaled, train_target, epochs=20, verbose=0,
                            validation_data=(val_scaled, val_target),
                            callbacks=[checkpoint_cb, early_stopping_cb])
        print(early_stopping_cb.stopped_epoch)
        # 훈련 손실과 검증 손실 그래프로 출력
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()
        ```
        

조기 종료로 에포크 횟수를 제한해 컴퓨터의 자원과 시간을 아낄 수 있고

최상의 모델을 자동으로 저장해 편리함.
