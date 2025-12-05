# Chapter 08-3

1. 함수형 API (functional API)
    1. 조금 더 복잡한 모델이 많음 → 입력이 2개일 수도 있고 출력이 2개일 수도 있고…
    2. 첫 번째 층의 출력을 두 번째 층의 입력으로
    3. 이때 inputs → InputLayer 클래스의 출력값
        
        ![image.png](image.png)
        

1. 실습
    - 코드
        
        ```python
        from tensorflow import keras
        
        !wget https://github.com/rickiepark/hg-mldl/raw/master/best-cnn-model.keras
        ```
        
        ```python
        model = keras.models.load_model('best-cnn-model.keras')
        model.layers
        ```
        
        ```python
        conv = model.layers[0]
        
        print(conv.weights[0].shape, conv.weights[1].shape)
        ```
        
        ```python
        conv_weights = conv.weights[0].numpy()
        
        print(conv_weights.mean(), conv_weights.std())
        
        import matplotlib.pyplot as plt
        
        plt.hist(conv_weights.reshape(-1, 1))
        plt.xlabel('weight')
        plt.ylabel('count')
        plt.show() # 0을 중심으로 종 모양
        ```
        
        ```python
        fig, axs = plt.subplots(2, 16, figsize=(15,2))
        
        for i in range(2):
            for j in range(16):
                axs[i, j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
                # imshow() : 배열에 있는 최댓값, 최솟값을 이용해 픽셀의 강도 표현
                axs[i, j].axis('off')
        
        plt.show()
        ```
        
        ```python
        no_training_model = keras.Sequential()
        
        no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                                  padding='same', input_shape=(28,28,1)))
        
        no_training_conv = no_training_model.layers[0]
        
        print(no_training_conv.weights[0].shape)
        ```
        
        ```python
        no_training_weights = no_training_conv.weights[0].numpy()
        
        print(no_training_weights.mean(), no_training_weights.std())
        
        plt.hist(no_training_weights.reshape(-1, 1))
        plt.xlabel('weight')
        plt.ylabel('count')
        plt.show() # 대부분의 가중치가 -0.15~0.15 사이, 비교적 고른 분포
        ```
        
        ```python
        fig, axs = plt.subplots(2, 16, figsize=(15,2))
        
        for i in range(2):
            for j in range(16):
                axs[i, j].imshow(no_training_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
                axs[i, j].axis('off')
        
        plt.show() # 가중치가 전체적으로 밋밋해짐 (색 차이 많이 x)
        ```
        
        ```python
        print(model.inputs)
        
        conv_acti = keras.Model(model.inputs, model.layers[0].output)
        ```
        
        ```python
        (train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
        
        plt.imshow(train_input[0], cmap='gray_r')
        plt.show()
        ```
        
        ```python
        inputs = train_input[0:1].reshape(-1, 28, 28, 1)/255.0
        
        feature_maps = conv_acti.predict(inputs)
        
        print(feature_maps.shape)
        ```
        
        ```python
        fig, axs = plt.subplots(4, 8, figsize=(15,8))
        
        for i in range(4):
            for j in range(8):
                axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
                axs[i, j].axis('off')
        
        plt.show()
        ```
        
        ```python
        conv2_acti = keras.Model(model.inputs, model.layers[2].output)
        
        feature_maps = conv2_acti.predict(train_input[0:1].reshape(-1, 28, 28, 1)/255.0)
        
        print(feature_maps.shape)
        ```
        
        ```python
        fig, axs = plt.subplots(8, 8, figsize=(12,12))
        
        for i in range(8):
            for j in range(8):
                axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
                axs[i, j].axis('off')
        
        plt.show() # 합성곱 신경망 앞부분에 있는 합성곱 층은 이미지의 시작적인 정보를 감지
        # 뒤쪽에 있는 합성곱 층은 감지한 시각적인 정보를 바탕으로 추상적인 정보 학습
        ```
        
    
    [08_3.ipynb](08_3.ipynb)