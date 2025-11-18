# Chapter 06-2

1. **k-평균 군집 알고리즘 (k-means clustering)**
    1. 타깃을 모를 때 **평균값**을 구하는 방법
    2. 평균값이 클러스터의 중심에 위치함 → **centroid**
    3. 알고리즘 작동 방식
        1. 무작위로 k개의 클러스터 중심을 정함
        2. 각 샘플에 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정
        3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심 변경
        4. 클러스터 중심에 변화가 없을 때까지 반복
            
            ![image.png]("https://github.com/user-attachments/assets/7844a069-5ca1-4d89-a0be-608e91d2af86")

            

1. **엘보우 (elbow)**
    1. 적절한 클러스터 개수를 찾기 위한 대표적인 방법
    2. **이너셔 (inertia)**
        1. (클러스터 중심과 샘플 사이의 거리) ** 2의 합
        2. 일반적으로 클러스터 개수가 늘어나면 클러스터 개개의 크기 줄어듦 → 이너셔 줄어듦
        3. 클러스터 개수를 늘려가면서 **이너셔의 변화를 관찰** → 최적의 클러스터 개수 찾음
        4. 그래프로 그리면 감소하는 **속도가 꺾이는 지점**이 있는데 이 지점부터는 클러스터에 잘 밀집된 정도가 크게 개선 x
            
            ![image.png](https://github.com/user-attachments/assets/6e3bb5dc-8b6d-4575-a4d3-3689956916c7)

            

1. 실습
    - 코드
        
        ```python
        !wget https://bit.ly/fruits_300_data -O fruits_300.npy
        ```
        
        ```python
        import numpy as np
        
        fruits = np.load('fruits_300.npy')
        fruits_2d = fruits.reshape(-1, 100*100) # (샘플 개수, 너비, 높이) 3차원 -> (샘플 개수, 너비 x 높이) 2차원
        
        from sklearn.cluster import KMeans
        
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(fruits_2d)
        
        print(km.labels_) # 레이블 : 0, 1, 2
        print(np.unique(km.labels_, return_counts=True))
        ```
        
        ```python
        import matplotlib.pyplot as plt
        
        def draw_fruits(arr, ratio=1):
          n = len(arr)
        
          # 한 줄에 10개씩 그림
          rows = int(np.ceil(n/10)) # 10으로 나누어 전체 행 개수 계산
          cols = n if rows < 2 else 10 # 행이 1개면 열의 개수 = 샘플 개수 / 그렇지 않으면 10개
        
          fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
        
          for i in range(rows):
            for j in range(cols):
              if i * 10 + j < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
              axs[i, j].axis('off')
        
          plt.show()
        ```
        
        ```python
        draw_fruits(fruits[km.labels_==2]) # 책에서는 0이지만 지금 자료에서 사과가 2
        draw_fruits(fruits[km.labels_==1])
        draw_fruits(fruits[km.labels_==0]) # 완벽하게 구분 x
        ```
        
        ```python
        draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3) # 중심을 이미지로 출력
        
        print(km.transform(fruits_2d[100:101])) # 각 클러스터 중심까지의 거리
        print(km.predict(fruits_2d[100:101])) # 중심 예측
        draw_fruits(fruits[100:101]) # 파인애플 그림
        print(km.n_iter_) # 알고리즘 반복 횟수
        ```
        
        ```python
        inertia = []
        
        for k in range(2, 7):
          km = KMeans(n_clusters=k, random_state=42)
          km.fit(fruits_2d)
          inertia.append(km.inertia_)
        
        plt.plot(range(2, 7), inertia)
        plt.xlabel('k')
        plt.ylabel('Inertia')
        
        plt.show() # k = 3에서 그래프의 기울기가 조금 바뀜 -> 명확하지는 않음
        ```
        
    
    [06_2.ipynb](06_2.ipynb)
