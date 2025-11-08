# Chapter 04-2

1. **확률적 경사 하강법 (stochastic gradient descent)**
    1. 전체 샘플을 사용하지 않고 **하나의 샘플**을 훈련 세트에서 랜덤하게 골라 가장 가파른 길 찾음
        1. 전체 샘플 모두 사용할 때까지
        2. 모든 샘플을 다 사용했는데 내려오지 못했다면 다시 처음부터

1. **미니배치 경사 하강법 (minibatch gradient descent)**
    1. **여러 개의 샘플**을 사용해 경사 하강법을 수행하는 방식
    2. 전체 샘플 사용 → **배치 경사 하강법 (batch gradient descent)**
        1. 데이터가 너무 많으면 **느림**

1. **손실 함수 (loss function)**
    1. 머신러닝 알고리즘이 **얼마나 틀렸는지** 측정
    2. 손실 함수의 값이 **작을수록 좋음**
    3. but 어떤 값이 최솟값인지는 알지 못함 (**local minimum**)
    4. **연속적**이어야 gradient descent 사용 가능

1. **logistic loss function (= binary cross-entropy loss function)**
    1. 타깃이 1일 때 손실은 **-log(예측 확률)**
        1. 확률이 1에서 멀어질수록 손실은 아주 큰 양수 됨
    2. 타깃이 0일 때 손실은 **-log(1 - 예측 확률)**
        1. 확률이 0에서 멀어질수록 손실은 아주 큰 양수 됨
    3. 다중 분류라면 → **cross-entropy loss function**

1. **에포크 (epoch)** : 훈련 세트 한 번을 모두 사용하는 과정
    1. epoch 횟수 너무 **적으면** 학습 덜함 → **과소적합**
    2. epoch 횟수 너무 **많으면** → **과대적합**
    3. **조기 종료 (early stopping)**
        1. 테스트 세트 점수가 감소하는 지점에서 훈련 멈춤 (**과대적합 방지**)

1. 힌지 손실 (hinge loss = support vector machine)

1. 실습
    - 코드
        
        ```python
        import pandas as pd
        fish = pd.read_csv('https://bit.ly/fish_csv_data')
        
        fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
        fish_target = fish['Species'].to_numpy()
        
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
        
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(train_input)
        
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)
        ```
        
        ```python
        from sklearn.linear_model import SGDClassifier
        sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42) # log : 로지스틱, max_iter : epoch
        sc.fit(train_scaled, train_target)
        
        print(sc.score(train_scaled, train_target))
        print(sc.score(test_scaled, test_target))
        ```
        
        ```python
        sc.partial_fit(train_scaled, train_target)
        print(sc.score(train_scaled, train_target))
        print(sc.score(test_scaled, test_target))
        ```
        
        ```python
        import numpy as np
        
        sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
        train_score = []
        test_score = []
        
        classes = np.unique(train_target)
        
        for _ in range(0, 300):
          sc.partial_fit(train_scaled, train_target, classes=classes)
        
          train_score.append(sc.score(train_scaled, train_target))
          test_score.append(sc.score(test_scaled, test_target))
        
        import matplotlib.pyplot as plt
        plt.plot(train_score)
        plt.plot(test_score)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()
        ```
        
        ```python
        sc = SGDClassifier(loss='log_loss', max_iter=1000, tol=None, random_state=42) # None : 자동으로 멈추는 것 방지
        sc.fit(train_scaled, train_target)
        
        print(sc.score(train_scaled, train_target))
        print(sc.score(test_scaled, test_target))
        ```
        
        ```python
        sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
        sc.fit(train_scaled, train_target)
        
        print(sc.score(train_scaled, train_target))
        print(sc.score(test_scaled, test_target))
        ```
        
    
    [04_2.ipynb](04_2.ipynb)