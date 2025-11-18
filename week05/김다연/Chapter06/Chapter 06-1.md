# Chapter 06-1

1. **비지도 학습 (unsupervised learning)**
    1. **타깃(정답)이 없을 때** 사용하는 알고리즘 

1. **군집 (clustering)**
    1. **비슷한 샘플**끼리 그룹으로 모으는 작업
    2. **cluster** : 군집 알고리즘에서 만든 그룹

1. 실습
    1. imshow() : 넘파이 배열로 저장된 이미지를 그리는 함수
    2. 원래는 흰 바탕에 짙은 물체 → 컴퓨터는 255에 가까운 흰 바탕에 집중할 것 → 반전시켜 바탕을 짙게 만들고 물체를 밝게
    3. subplots()
        1. 여러 개의 그래프를 배열처럼 쌓을 수 있도록 도와줌
        2. subplots(1, 2) → 1개의 행, 2개의 열
    4. axis
        1. 배열의 축
        2. axis=1 → 열 방향 계산 / axis=0 → 행 방향 계산
    5. np.argsort() : 작은 것에서 큰 순서대로 나열한 abs_mean 배열의 인덱스 반환
    - 코드
        
        ```python
        !wget https://bit.ly/fruits_300_data -O fruits_300.npy
        ```
        
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        
        fruits = np.load('fruits_300.npy')
        print(fruits.shape)
        print(fruits[0, 0, :])
        
        # plt.imshow(fruits[0], cmap='gray') -> 흑백 이미지여서 cmap을 그레이로
        plt.imshow(fruits[0], cmap='gray_r') # 이렇게 하면 좀 더 보기 좋아짐 -> 밝은 부분 : 0에 가까움, 짙은 부분 : 255에 가까움
        plt.show() # 0에 가까울수록 검게 나타나고 높은 값을 밝게 표시
        ```
        
        ```python
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(fruits[100], cmap='gray_r') # 파인애플
        axs[1].imshow(fruits[200], cmap='gray_r') # 바나나
        plt.show()
        ```
        
        ```python
        apple = fruits[0:100].reshape(-1, 100 * 100) # 첫 번째 차원 : 샘플 개수 (-1이라 자동으로 남은 차원 할당)
        # 두 번째, 세 번째 차원 10,000으로 합침 -> 결국 이미지를 한 줄로 펴주는 거임
        # 머신러닝은 이미지 자체를 학습하지 못해서 이렇게 변환해주는 것
        pineapple = fruits[100:200].reshape(-1, 100 * 100)
        banana = fruits[200:300].reshape(-1, 100 * 100)
        print(apple.shape)
        print(apple.mean(axis=1))
        ```
        
        ```python
        # 샘플의 평균값
        plt.hist(apple.mean(axis=1), alpha=0.8)
        plt.hist(pineapple.mean(axis=1), alpha=0.8)
        plt.hist(banana.mean(axis=1), alpha=0.8)
        
        plt.legend(['apple', 'pineapple', 'banana'])
        plt.show()  # 바나나 사진의 평균값은 40 아래에 집중 / 사과, 파인애플의 평균은 90~100 사이
        # 바나나가 사진에서 차지하는 영역이 작아서 평균값이 작은 듯...
        ```
        
        ```python
        # 픽셀의 평균값
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].bar(range(10000), apple.mean(axis=0))
        axs[1].bar(range(10000), pineapple.mean(axis=0))
        axs[2].bar(range(10000), banana.mean(axis=0))
        plt.show() # 순서대로 사과, 파인애플, 바나나 그래프
        # 사과 -> 아래쪽으로 갈수록 값이 높아짐 / 파인애플 -> 비교적 고르면서 높음 / 바나나 -> 중앙의 픽셀값이 높음
        ```
        
        ```python
        apple_mean = apple.mean(axis=0).reshape(100, 100)
        pineapple_mean = pineapple.mean(axis=0).reshape(100, 100)
        banana_mean = banana.mean(axis=0).reshape(100, 100)
        
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].imshow(apple_mean, cmap='gray_r')
        axs[1].imshow(pineapple_mean, cmap='gray_r')
        axs[2].imshow(banana_mean, cmap='gray_r')
        plt.show()
        ```
        
        ```python
        abs_diff = np.abs(fruits - apple_mean)
        abs_mean = abs_diff.mean(axis=(1, 2)) # 이 이미지가 사과와 얼마나 다른가 (사과인지 알기 위함)
        print(abs_mean.shape)
        
        apple_index = np.argsort(abs_mean)[:100] # 사과 이미지에 가장 가까운 이미지 100개 찾기
        fig, axs = plt.subplots(10, 10, figsize=(10, 10)) # 100개의 서브 그래프 생성
        
        for i in range(10):
            for j in range(10): # 반복문으로 서브 그래프의 위치 지정
                axs[i, j].imshow(fruits[apple_index[i * 10 + j]], cmap='gray_r')
                axs[i, j].axis('off') # 좌표축 생략
        
        plt.show() # apple_mean과 가장 가까운 사진 100개 -> 모두 사과
        ```
        
    
    [06_1.ipynb](06_1.ipynb)