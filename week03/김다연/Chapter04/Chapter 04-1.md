# Chapter 04-1

1. **다중 분류 (multi-class classification)** : 타깃 데이터에 **2개 이상의 클래스**가 포함된 문제
    1. ↔ 이진 분류 but 모델을 만들고 훈련하는 방식은 동일
    2. **소프트맥스(softmax)** 함수를 사용하여 n개의 z 값을 확률로 변환
        1. 이진 분류와 달리 여러 개의 선형 방정식의 출력값을 0~1 사이로 압축, 합이 1이 되도록 함
2. **로지스틱 회귀 (logistic regression)**
    1. 이름은 회귀지만 **분류** 모델
    2. **선형 방정식** 학습
  
        ![image.png](https://github.com/user-attachments/assets/c1c2700e-d913-40ec-ae57-ee1130755b74)
        
        a, b, c, d, e는 가중치 or 계수 → 특성은 늘어났지만 선형 방정식
        
    3. z는 어떤 값도 가능하지만 확률이 되려면 0~1 사이 값이어야 함 → **시그모이드 함수(sigmoid function)** 사용
        
        ![image.png](https://github.com/user-attachments/assets/e049901a-ee07-4df5-b3c8-d0369a57acc3)
        
    4. 이진 분류일 때 : 0.5보다 크면 양성 클래스, 작으면 음성 클래스
    5. 규제 제어 매개변수 : **C**
        1. C가 작을수록 규제 커짐
        2. C의 기본값 : 1
3. **불리언 인덱싱 (boolean indexing)**
    1. 넘파이 배열에서 **true, false 값을 전달**하여 행을 선택하는 것
        
        ```python
        char_arr = np.array[['A', 'B', 'C', 'D', 'E'])
        print(char_arr[[True, False, True, False, False]]) # 이러면 A, C만 출력됨
        ```

4. 실습
    1. to_numpy() : 데이터프레임 → 넘파이로 변환
    2. predict_proba() : 클래스별 확률값
        
        ![image.png](https://github.com/user-attachments/assets/8d3e963c-6f47-458f-883b-7a650a443fa8)
        
    - 코드
        
        ```python
        import pandas as pd
        
        fish = pd.read_csv('https://bit.ly/fish_csv_data')
        # fish.head()
        
        print(pd.unique(fish['Species']))
        ```
        
        ```python
        fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
        print(fish_input[:5])
        
        fish_target = fish['Species'].to_numpy()
        ```
        
        ```python
        from sklearn.model_selection import train_test_split
        
        # 훈련 세트, 테스트 세트 분리
        train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
        
        from sklearn.preprocessing import StandardScaler
        
        # 전처리
        ss = StandardScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)
        ```
        
        ```python
        from sklearn.neighbors import KNeighborsClassifier
        
        kn = KNeighborsClassifier(n_neighbors=3)
        kn.fit(train_scaled, train_target)
        
        print(kn.score(train_scaled, train_target))
        print(kn.score(test_scaled, test_target))
        ```
        
        ```python
        print(kn.classes_)
        
        print(kn.predict(test_scaled[:5]))
        ```
        
        ```python
        import numpy as np
        
        proba = kn.predict_proba(test_scaled[:5])
        print(np.round(proba, decimals=4)) # 소수 네 번째 자리까지
        ```
        
        ```python
        distances, indexes = kn.kneighbors(test_scaled[3:4])
        print(train_target[indexes]) # Roach 1개(0.3333), Perch 2개(0.6667)
        ```
        
        ```python
        import matplotlib.pyplot as plt
        
        z = np.arange(-5, 5, 0.1)
        phi = 1 / (1 + np.exp(-z))
        plt.plot(z, phi)
        plt.xlabel('z')
        plt.ylabel('phi')
        plt.show() # 시그모이드 함수
        ```
        
        ```python
        bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
        # 도미, 빙어일 경우 true / 그 외는 모두 false
        train_bream_smelt = train_scaled[bream_smelt_indexes]
        target_bream_smelt = train_target[bream_smelt_indexes]
        
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression()
        lr.fit(train_bream_smelt, target_bream_smelt)
        
        print(lr.predict(train_bream_smelt[:5]))
        print(lr.predict_proba(train_bream_smelt[:5]))
        # 첫 번째 열 : 0에 대한 확률, 두 번째 열 : 1에 대한 확률 -> 이때 smelt가 0, bream이 1
        print(lr.coef_, lr.intercept_)
        ```
        
        ```python
        decisions = lr.decision_function(train_bream_smelt[:5])
        print(decisions)
        
        from scipy.special import expit
        print(expit(decisions)) # z : predict_proba()의 두 번째 열(0)과 값 동일
        ```
        
        ```python
        lr = LogisticRegression(C=20, max_iter=1000)
        lr.fit(train_scaled, train_target)
        
        print(lr.score(train_scaled, train_target))
        print(lr.score(test_scaled, test_target))
        print(lr.predict(test_scaled[:5]))
        
        proba = lr.predict_proba(test_scaled[:5])
        print(np.round(proba, decimals=3))
        
        print(lr.coef_.shape, lr.intercept_.shape)
        ```
        
        ```python
        decisions = lr.decision_function(test_scaled[:5])
        print(np.round(decisions, decimals=3))
        
        from scipy.special import softmax
        proba = softmax(decisions, axis=1) # axis 매개변수를 지정하지 않음녀 배열 전체에 대해 소프트맥스 계산함
        print(np.round(proba, decimals=3))
        ```
        
    
    [04_1.ipynb](04_1.ipynb)
