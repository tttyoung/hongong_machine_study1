# Chapter 09-3

1. **LSTM**
    1. Long Short-Term Memory
    2. **단기 기억을 오래 기억**하기 위해 고안
    3. 은닉 상태 만드는 법 : 입력, 이전 타임스텝의 은닉 상태를 가중치에 곱함 → 활성화 함수 통과 → 은닉 상태 만듦 (활성화 함수 : 시그모이드 함수)
    4. **셀 상태 (cell state)** : 다음 층으로 전달되지 않고 LSTM 셀에서 순환만 됨
        
        ![image.png](image.png)
        
    5. **삭제 게이트** : 셀 상태에 있는 정보 제거 / **입력 게이트** : 새로운 정보를 셀 상태에 추가 / **출력 게이트** : 셀 상태가 다음 은닉 상태로 출력

1. **GRU**
    1. Gated Recurrent Unit
    2. 은닉 상태 하나만 포함
        
        ![image.png](image%201.png)
        

1. 실습
    - 코드
        
        ```python
        from tensorflow.keras.datasets import imdb
        from sklearn.model_selection import train_test_split
        
        (train_input, train_target), (test_input, test_target) = imdb.load_data(
            num_words=500)
        
        train_input, val_input, train_target, val_target = train_test_split(
            train_input, train_target, test_size=0.2, random_state=42)
        
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        train_seq = pad_sequences(train_input, maxlen=100)
        val_seq = pad_sequences(val_input, maxlen=100)
        
        from tensorflow import keras
        
        model = keras.Sequential()
        
        model.add(keras.layers.Embedding(500, 16, input_shape=(100,)))
        model.add(keras.layers.LSTM(8))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        
        model.summary()
        ```
        
        ```python
        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model.compile(optimizer=rmsprop, loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.keras',
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                          restore_best_weights=True)
        
        history = model.fit(train_seq, train_target, epochs=100, batch_size=64,
                            validation_data=(val_seq, val_target),
                            callbacks=[checkpoint_cb, early_stopping_cb])
        ```
        
        ```python
        import matplotlib.pyplot as plt
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'val'])
        plt.show()
        ```
        
        ```python
        model2 = keras.Sequential()
        
        model2.add(keras.layers.Embedding(500, 16, input_shape=(100,)))
        model2.add(keras.layers.LSTM(8, dropout=0.3))
        model2.add(keras.layers.Dense(1, activation='sigmoid'))
        
        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model2.compile(optimizer=rmsprop, loss='binary_crossentropy',
                       metrics=['accuracy'])
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-dropout-model.keras',
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                          restore_best_weights=True)
        
        history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                             validation_data=(val_seq, val_target),
                             callbacks=[checkpoint_cb, early_stopping_cb])
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
        model3 = keras.Sequential()
        
        model3.add(keras.layers.Embedding(500, 16, input_shape=(100,)))
        model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
        model3.add(keras.layers.LSTM(8, dropout=0.3))
        model3.add(keras.layers.Dense(1, activation='sigmoid'))
        
        model3.summary()
        ```
        
        ```python
        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model3.compile(optimizer=rmsprop, loss='binary_crossentropy',
                       metrics=['accuracy'])
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-2rnn-model.keras',
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                          restore_best_weights=True)
        
        history = model3.fit(train_seq, train_target, epochs=100, batch_size=64,
                             validation_data=(val_seq, val_target),
                             callbacks=[checkpoint_cb, early_stopping_cb])
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
        model4 = keras.Sequential()
        
        model4.add(keras.layers.Embedding(500, 16, input_shape=(100,)))
        model4.add(keras.layers.GRU(8))
        model4.add(keras.layers.Dense(1, activation='sigmoid'))
        
        model4.summary()
        ```
        
        ```python
        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model4.compile(optimizer=rmsprop, loss='binary_crossentropy',
                       metrics=['accuracy'])
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-gru-model.keras',
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                          restore_best_weights=True)
        
        history = model4.fit(train_seq, train_target, epochs=100, batch_size=64,
                             validation_data=(val_seq, val_target),
                             callbacks=[checkpoint_cb, early_stopping_cb])
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
        test_seq = pad_sequences(test_input, maxlen=100)
        
        rnn_model = keras.models.load_model('best-2rnn-model.keras')
        
        rnn_model.evaluate(test_seq, test_target)
        ```