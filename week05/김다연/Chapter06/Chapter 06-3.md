# Chapter 06-3

1. **차원 (dimension)**
    1. 데이터가 가진 속성 (**특성**)
        1. 10,000개의 특성은 곧 10,000개의 차원이라는 것

1. **차원 축소 (dimesionality reduction)**
    1. 데이터를 가장 잘 나타내는 **일부 특성**을 선택 → 데이터 **크기 줄임**, 모델의 **성능 향상**

1. **주성분 분석 (PCA)**
    1. 데이터에 있는 **분산이 큰 방향**을 찾는 것 (분산 : 데이터가 퍼져 있는 정도)
        
        ![image.png](image.png)
        
    2. **주성분 벡터**
        1. 원본 데이터에 있는 어떤 방향
        2. 주성분 벡터의 원소 개수 = 원본 데이터셋의 특성 개수
            
            ![2차원 데이터 S(4, 2) → 1차원 데이터 P(4.5)로 투영 (이 데이터가 주성분 방향으로 얼마나 떨어져 있는지 숫자로 표현)](image%201.png)
            
            2차원 데이터 S(4, 2) → 1차원 데이터 P(4.5)로 투영 (이 데이터가 주성분 방향으로 얼마나 떨어져 있는지 숫자로 표현)
            
        3. **투영** : 원래 데이터를 주성분 방향으로 옮기는 것
        4. 주성분이 가장 분산이 큰 방향이기 때문에 주성분에 투영하여 바꾼 데이터는 **원본이 가지고 있는 특성 가장 잘 나타냄**
        5. 일반적으로 주성분 개수 = 원본 특성의 개수
    3. **설명된 분산 (explained variance)**
        1. 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값

1. 실습
    - 코드
        
        ```python
        !wget https://bit.ly/fruits_300_data -O fruits_300.npy
        
        import numpy as np
        
        fruits = np.load('fruits_300.npy')
        fruits_2d = fruits.reshape(-1, 100*100)
        ```
        
        ```python
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=50)
        pca.fit(fruits_2d)
        
        print(pca.components_.shape) # 50, 10000 -> 차원 = 주성분 : 50, 10000
        ```
        
        ```python
        import matplotlib.pyplot as plt
        
        def draw_fruits(arr, ratio=1):
          n = len(arr)
        
          rows = int(np.ceil(n/10))
          cols = n if rows < 2 else 10
        
          fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
        
          for i in range(rows):
            for j in range(cols):
              if i * 10 + j < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
              axs[i, j].axis('off')
        
          plt.show()
        ```
        
        ```python
        draw_fruits(pca.components_.reshape(-1, 100, 100)) # 주성분을 그림으로 -> 데이터의 특징 잡아냄
        ```
        
        ```python
        print(fruits_2d.shape)
        
        fruits_pca = pca.transform(fruits_2d) # 300, 100 -> 300, 50
        print(fruits_pca.shape) # 차원 축소
        ```
        
        ```python
        fruits_inverse = pca.inverse_transform(fruits_pca) # 300, 50 -> 300, 10000
        print(fruits_inverse.shape) # 원본 데이터 복원
        
        fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
        
        for start in [0, 100, 200]:
          draw_fruits(fruits_reconstruct[start:start+100])
          print("\n") # 거의 모든 과일이 잘 복원됨 (조금 흐리고 번지긴 함)
        ```
        
        ```python
        print(np.sum(pca.explained_variance_ratio_)) # 92%의 분산 유지 -> 복원 시 원본 이미지의 품질 높음
        
        plt.plot(pca.explained_variance_ratio_)
        plt.show() # 처음 10개의 분산이 대부분의 분산 표현
        ```
        
        ```python
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression()
        target = np.array([0] * 100 + [1] * 100 + [2] * 100) # 사과 : 0, 파인애플 : 1, 바나나 : 2
        
        from sklearn.model_selection import cross_validate
        
        scores = cross_validate(lr, fruits_2d, target)
        print(np.mean(scores['test_score']))
        print(np.mean(scores['fit_time']))
        
        scores = cross_validate(lr, fruits_pca, target)
        print(np.mean(scores['test_score']))
        print(np.mean(scores['fit_time'])) # 시간이 엄청나게 감소
        ```
        
        ```python
        pca = PCA(n_components=0.5)
        pca.fit(fruits_2d)
        
        print(pca.n_components_) # 2개의 특성만으로 원본 데이터에 있는 분산의 50% 표현 가능
        
        fruits_pca = pca.transform(fruits_2d)
        print(fruits_pca.shape)
        
        scores = cross_validate(lr, fruits_pca, target)
        print(np.mean(scores['test_score'])) # 특성 2개만 사용해도 99%의 정확도
        print(np.mean(scores['fit_time']))
        ```
        
        ```python
        from sklearn.cluster import KMeans
        
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(fruits_pca)
        print(np.unique(km.labels_, return_counts=True))
        
        for label in range(0, 3):
          draw_fruits(fruits[km.labels_ == label])
          print("\n")
        ```
        
        ```python
        for label in range(0, 3):
          data = fruits_pca[km.labels_ == label]
          plt.scatter(data[:, 0], data[:, 1])
        
        plt.legend(['apple', 'pineapple', 'banana'])
        plt.show() # 클러스터의 산점도로 볼 수 있음 -> 아주 잘 구분되어 있음
        ```
        
    
    [06_3.ipynb](06_3.ipynb)