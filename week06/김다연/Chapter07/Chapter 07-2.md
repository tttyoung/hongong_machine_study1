# Chapter 07-2

1. **은닉층 (hidden layer)**
    1. **입력층, 출력층 사이**에 있는 모든 층
    2. 출력층에 적용하는 활성화 함수 : 제한적 but 은닉층에 적용하는 활성화 함수 : 비교적 자유로움
    3. 많이 사용하는 활성화 함수
        1. 시그모이드 함수
        2. 렐루 함수
    4. 은닉층의 뉴런 → 적어도 **출력층의 뉴런보다는 많게** 만들어야 함

1. **배치 차원**
    1. 신경망 층에 입력되거나 출력되는 **배열의 첫 번째 차원**
    2. 미니배치 경사 하강법 이용
    3. 샘플 개수를 **고정하지 않고** 어떤 배치 크기에도 **유연하게** 대응 → **None으로** 설정

1. **렐루(ReLU) 함수**
    1. 초창기 은닉층에 많이 사용된 활성화 함수 → 시그모이드 함수 but 함수의 끝으로 갈수록 그래프가 누워 있기 때문에 올바른 출력에 신속한 대응 x
    2. 입력이 **양수일 경우 입력 통과**시킴 / **음수일 경우 0으로** 만듦
    3. **max(0, z)**
        
        ![image.png](image.png)
        

1. **옵티마이저 (optimizer)**
    1. 다양한 종류의 경사 하강법 알고리즘
        
        ![image.png](image%201.png)
        
    2. 하이퍼파라미터 중 하나
    3. **모멘텀 최적화 (momentum optimization)**
        1. momentum 매개변수를 **0보다 큰 값**으로 지정할 때
        2. 보통 momentum 매개변수는 0.9 이상을 지정
    4. **네스테로프 모멘텀 최적화 (nesterov momentum optimization)**
        1. nesterov 매개변수를 **기본값 False → True**
        2. **모멘텀 최적화를 2번** 반복하여 구현
    5. **적응적 학습률**
        1. 모델이 최적점에 가까이 갈수록 학습률 낮춤
        2. 사용하는 대표적인 옵티마이저 : Adagrad, RMSprop

1. 실습
    - 코드
        
        ```python
        from tensorflow import keras
        
        (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
        
        from sklearn.model_selection import train_test_split
        
        train_scaled = train_input / 255.0
        train_scaled = train_scaled.reshape(-1, 28*28)
        
        train_scaled, val_scaled, train_target, val_target = train_test_split(
            train_scaled, train_target, test_size=0.2, random_state=42)
        ```
        
        ```python
        dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
        dense2 = keras.layers.Dense(10, activation='softmax')
        
        model = keras.Sequential([dense1, dense2])
        model.summary()
        ```
        
        ```python
        model = keras.Sequential([
            keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
            keras.layers.Dense(10, activation='softmax', name='output')
        ], name='패션 MNIST 모델')
        model.summary()
        ```
        
        ```python
        model = keras.Sequential()
        model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
        model.summary()
        ```
        
        ```python
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(train_scaled, train_target, epochs=5)
        ```
        
        ```python
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28))) # Flatten은 학습하는 층 x -> 깊이 3인 신경망 x
        model.add(keras.layers.Dense(100, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
        model.summary()
        ```
        
        ```python
        (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
        
        train_scaled = train_input / 255.0
        
        train_scaled, val_scaled, train_target, val_target = train_test_split(
            train_scaled, train_target, test_size=0.2, random_state=42)
        
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(train_scaled, train_target, epochs=5)
        model.evaluate(val_scaled, val_target)
        ```
        
        ```python
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # sgd = keras.optimizers.SGD()
        # model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # 위 코드와 완전히 동일
        
        sgd = keras.optimizers.SGD(learning_rate=0.1)
        sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)
        ```
        
        ```python
        adagrad = keras.optimizers.Adagrad()
        model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        rmsprop = keras.optimizers.RMSprop()
        model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        ```
        
        ```python
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))
        model.add(keras.layers.Dense(100, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(train_scaled, train_target, epochs=5)
        model.evaluate(val_scaled, val_target)
        ```
        
    
    [07_2.ipynb](07_2.ipynb)