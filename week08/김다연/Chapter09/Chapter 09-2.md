# Chapter 09-2

1. IMDB 리뷰 데이터셋
    1. 영화 리뷰를 감상평에 따라 긍정 / 부정으로 분류해놓은 데이터셋
    2. **자연어 처리 (NLP)** : 컴퓨터를 사용해 인간의 언어를 처리
    3. **말뭉치 (corpus)** : 자연어 처리 분야에서의 훈련 데이터
    4. 텍스트 자체를 신경망에 전달 x → 컴퓨터에서 처리하는 것은 어떤 숫자 데이터
        
        ![image.png](image.png)
        
    5. **토큰 (token)** : 모두 소문자로, 구둣점 삭제, 공백 기준 분리한 단어
    6. 어휘 사전 : 훈련 세트에서 고유한 단어를 뽑아 만든 목록

1. **단어 임베딩 (word embedding)** : 각 단어를 고정된 크기의 실수 벡터로 바꿔줌
    
    ![image.png](image%201.png)
    
    1. 입력을 정수 데이터로 받음 → **메모리 효율적 사용**

1. 실습
    - 코드
        
        ```python
        from tensorflow.keras.datasets import imdb
        
        (train_input, train_target), (test_input, test_target) = imdb.load_data(
            num_words=200)
        
        print(train_input.shape, test_input.shape)
        print(len(train_input[0]))
        print(len(train_input[1]))
        print(train_input[0])
        print(train_target[:20]) # 0 : 부정, 1 : 긍정
        ```
        
        ```python
        from sklearn.model_selection import train_test_split
        
        train_input, val_input, train_target, val_target = train_test_split(
            train_input, train_target, test_size=0.2, random_state=42)
        
        import numpy as np
        
        lengths = np.array([len(x) for x in train_input])
        
        print(np.mean(lengths), np.median(lengths))
        
        import matplotlib.pyplot as plt
        
        plt.hist(lengths)
        plt.xlabel('length')
        plt.ylabel('frequency')
        plt.show() # 한 쪽에 치우침
        ```
        
        ```python
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        train_seq = pad_sequences(train_input, maxlen=100) # 토큰 길이 100으로
        
        print(train_seq.shape)
        print(train_seq[0])
        print(train_input[0][-10:])
        print(train_seq[5])
        
        val_seq = pad_sequences(val_input, maxlen=100) # 검증 세트의 길이도 100으로
        ```
        
        ```python
        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model.compile(optimizer=rmsprop, loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.keras',
                                                        save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                          restore_best_weights=True)
        
        history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                            validation_data=(val_oh, val_target),
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
        model2 = keras.Sequential()
        
        model2.add(keras.layers.Embedding(200, 16, input_shape=(100,)))
        model2.add(keras.layers.SimpleRNN(8))
        model2.add(keras.layers.Dense(1, activation='sigmoid'))
        
        model2.summary()
        ```
        
        ```python
        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model2.compile(optimizer=rmsprop, loss='binary_crossentropy',
                       metrics=['accuracy'])
        
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.keras',
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