# Chapter 07-3

1. **검증 손실**
    1. 과대적합 / 과소적합 판단
        
        ![현재 과대적합 상태](https://github.com/user-attachments/assets/45f599e2-7a11-44f1-9c34-06e6921b1cc7)
        
        현재 과대적합 상태
        
    2. Adam 옵티마이저 적용
        
        ![과대적합 훨씬 줄어듦](https://github.com/user-attachments/assets/7bc14251-49f0-4587-8ec3-db1b87169cc6)

        
        과대적합 훨씬 줄어듦
        

1. **드롭아웃 (dropout)**
    1. 훈련 과정에서 층에 있는 **일부 뉴런을 랜덤하게 꺼서**(뉴런의 출력을 0으로 만들어서) **과대적합 방지** → 특정 뉴런에 과대하게 의존하는 것 방지
        
        ![image.png](https://github.com/user-attachments/assets/4b020011-e61b-4762-bd98-570348436d3e)

        
    2. 일부 뉴런의 출력을 0으로 만들지만 전체 출력 배열의 크기는 바꾸지 x
    3. **평가 및 예측 시 드롭아웃 적용 x**
        
        ![과대적합 확연히 줆](https://github.com/user-attachments/assets/64949edc-cdeb-4147-8235-8927ad80b564)

        
        과대적합 확연히 줆
        

1. **콜백 (callback)**
    1. 훈련 과정 중간에 어떤 작업을 수행할 수 있게 하는 객체

1. **조기 종료 (early stopping)**
    1. 과대적합이 시작되기 전에 **훈련을 미리 중지**하는 것
    2. 규제 방법 중 하나라고 생각할 수 있음

1. 실습
    - 코드
        
        ```python
        from tensorflow import keras
        from sklearn.model_selection import train_test_split
        
        (train_input, train_target), (test_input, test_target) = \
            keras.datasets.fashion_mnist.load_data()
        
        train_scaled = train_input / 255.0
        
        train_scaled, val_scaled, train_target, val_target = train_test_split(
            train_scaled, train_target, test_size=0.2, random_state=42)
        
        def model_fn(a_layer=None):
            model = keras.Sequential()
            model.add(keras.layers.Flatten(input_shape=(28, 28)))
            model.add(keras.layers.Dense(100, activation='relu'))
            if a_layer:
                model.add(a_layer)
            model.add(keras.layers.Dense(10, activation='softmax'))
            return model
        
        model = model_fn()
        
        model.summary()
        ```
        
        ```python
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
        
        print(history.history.keys())
        ```
        
        ```python
        import matplotlib.pyplot as plt
        
        plt.plot(history.history['loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        ```
        
        ```python
        plt.plot(history.history['accuracy'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()
        ```
        
        ```python
        model = model_fn()
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
        
        plt.plot(history.history['loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        ```
        
        ```python
        model = model_fn()
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(train_scaled, train_target, epochs=20, verbose=0,
                            validation_data=(val_scaled, val_target))
        
        print(history.history.keys())
        ```
        
        ```python
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()
        ```
        
        ```python
        model = model_fn()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        history = model.fit(train_scaled, train_target, epochs=20, verbose=0,
                            validation_data=(val_scaled, val_target))
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()
        ```
        
        ```python
        model = model_fn(keras.layers.Dropout(0.3))
        
        model.summary()
        ```
        
        ```python
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        history = model.fit(train_scaled, train_target, epochs=20, verbose=0,
                            validation_data=(val_scaled, val_target))
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()
        ```
        
        ```python
        model = model_fn(keras.layers.Dropout(0.3))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        history = model.fit(train_scaled, train_target, epochs=10, verbose=0,
                            validation_data=(val_scaled, val_target))
        ```
        
        ```python
        model.save('model-whole.keras')
        model.save_weights('model.weights.h5')
        ```
        
        ```python
        !ls -al model*
        ```
        
        ```python
        model = model_fn(keras.layers.Dropout(0.3))
        
        model.load_weights('model.weights.h5')
        ```
        
        ```python
        import numpy as np
        
        val_labels = np.argmax(model.predict(val_scaled), axis=-1)
        print(np.mean(val_labels == val_target))
        ```
        
        ```python
        model = keras.models.load_model('model-whole.keras')
        
        model.evaluate(val_scaled, val_target)
        ```
        
        ```python
        model = model_fn(keras.layers.Dropout(0.3))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras',
                                                        save_best_only=True)
        
        model.fit(train_scaled, train_target, epochs=20, verbose=0,
                  validation_data=(val_scaled, val_target),
                  callbacks=[checkpoint_cb])
        
        model = keras.models.load_model('best-model.keras')
        
        model.evaluate(val_scaled, val_target)
        ```
        
        ```python
        model = model_fn(keras.layers.Dropout(0.3))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras',
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                          restore_best_weights=True)
        
        history = model.fit(train_scaled, train_target, epochs=20, verbose=0,
                            validation_data=(val_scaled, val_target),
                            callbacks=[checkpoint_cb, early_stopping_cb])
        
        print(early_stopping_cb.stopped_epoch)
        ```
        
        ```python
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()
        ```
        
        ```python
        model.evaluate(val_scaled, val_target)
        ```
        
    
    [07_3.ipynb](07_3.ipynb)
