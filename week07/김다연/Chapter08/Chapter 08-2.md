# Chapter 08-2

1. 실습
    - 코드
        
        ```python
        from tensorflow import keras
        from sklearn.model_selection import train_test_split
        
        (train_input, train_target), (test_input, test_target) = \
            keras.datasets.fashion_mnist.load_data()
        
        train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
        
        train_scaled, val_scaled, train_target, val_target = train_test_split(
            train_scaled, train_target, test_size=0.2, random_state=42)
        ```
        
        ```python
        model = keras.Sequential()
        
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                      padding='same', input_shape=(28,28,1))) # 32개의 필터, 커널 : (3, 3)
        model.add(keras.layers.MaxPooling2D(2)) # 풀링 층 추가 (2, 2) -> 특성 맵 절반으로 줄어듦
        
        model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu',
                                      padding='same')) # 필터 개수 64개로
        model.add(keras.layers.MaxPooling2D(2))
        
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation='relu'))
        model.add(keras.layers.Dropout(0.4)) # 과대적합 방지
        model.add(keras.layers.Dense(10, activation='softmax'))
        
        model.summary()
        ```
        
        ```python
        keras.utils.plot_model(model)
        ```
        
        ```python
        keras.utils.plot_model(model, show_shapes=True) # 입력, 출력의 크기 표시
        ```
        
        ```python
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras',
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                          restore_best_weights=True)
        
        history = model.fit(train_scaled, train_target, epochs=20,
                            validation_data=(val_scaled, val_target),
                            callbacks=[checkpoint_cb, early_stopping_cb])
        ```
        
        ```python
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()
        
        model.evaluate(val_scaled, val_target)
        ```
        
        ```python
        plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
        plt.show()
        ```
        
        ```python
        preds = model.predict(val_scaled[0:1])
        print(preds) # 9번째 값이 1이고 나머지는 0에 가까움 -> 9번째 클래스라고 하는 것
        
        plt.bar(range(1, 11), preds[0])
        plt.xlabel('class')
        plt.ylabel('prob.')
        plt.show() # 막대그래프로 표현
        ```
        
        ```python
        classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
                   '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']
        
        import numpy as np
        print(classes[np.argmax(preds)])
        ```
        
        ```python
        test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
        model.evaluate(test_scaled, test_target)
        ```
        
    
    [08_2.ipynb](08_2.ipynb)