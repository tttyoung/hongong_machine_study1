# Chapter 05-2

1. **검증 세트 (validation set)**
    1. 테스트 세트를 사용하지 않고 이를 측정하기 위해 **훈련 세트를 또 나눴**을 때의 데이터
        
        ![image.png](image.png)
        
    2. **매개변수를 정할 때** 사용

1. **교차 검증 (cross validation)**
    1. 검증 세트 때문에 훈련 세트 줄음 → 많은 데이터 훈련에 사용할수록 좋은 모델이지만 검증 세트를 너무 조금 남겨 놓으면 검증 점수 불안정
    2. 검증 세트를 떼어 내어 **평가하는 과정 여러 번** 반복 → 이 점수를 **평균**하여 최종 검증 점수
        
        ![3-폴드 교차 검증](image%201.png)
        
        3-폴드 교차 검증
        
    3. 회귀 모델 → KFold 분할기, 분류 모델 → StratifiedKFold

1. 하이퍼파라미터 튜닝
    1. **하이퍼파라미터** : **사용자가 지정**해야 하는 파라미터
    2. 매개변수가 여러 개라면 모두 동시에 바꿔가며 최적의 값을 찾아야 함 → 하나 맞추고 하나 하고 이거 안 됨
    3. **그리드 서치 (Grid Search)**
        1. 하이퍼파라미터 탐색, 교차 검증 한 번에 수행
    4. **랜덤 서치**
        1. 매개변수 값의 목록 전달 x 매개변수 샘플링할 수 있는 **확률 분포** 객체 전달

1. 실습
    1. cross_validate()
        1. 교차 검증 함수
        2. fit_time, score_time, test_score 반환 (모델 훈련 시간, 검증 시간, 점수)
        3. 각 키마다 5개의 숫자 담겨 있음 → 기본적으로 5-폴드 교차 검증 수행
    2. argmax() : 가장 큰 값의 인덱스 추출
    - 코드
        
        ```python
        import pandas as pd
        wine = pd.read_csv('https://bit.ly/wine_csv_data')
        
        data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
        target = wine['class'].to_numpy()
        
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
        
        sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
        
        print(sub_input.shape, val_input.shape)
        ```
        
        ```python
        from sklearn.tree import DecisionTreeClassifier
        
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(sub_input, sub_target)
        
        print(dt.score(sub_input, sub_target))
        print(dt.score(val_input, val_target))
        ```
        
        ```python
        from sklearn.model_selection import cross_validate
        scores = cross_validate(dt, train_input, train_target)
        print(scores)
        
        import numpy as np
        
        print(np.mean(scores['test_score']))
        ```
        
        ```python
        from sklearn.model_selection import StratifiedKFold
        
        scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
        print(np.mean(scores['test_score']))
        
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # 10-폴드 교차 검증
        scores = cross_validate(dt, train_input, train_target, cv=splitter)
        print(np.mean(scores['test_score']))
        ```
        
        ```python
        from sklearn.model_selection import GridSearchCV
        
        params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
        gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1) # 값을 바꿔가며 총 5번 실행
        # n_jobs -> 병렬 실행에 사용할 CPU 코어 수 지정 (-1 : 모든 코어 사용)
        gs.fit(train_input, train_target)
        
        dt = gs.best_estimator_
        print(dt.score(train_input, train_target))
        print(gs.best_params_) # 최적의 파라미터
        print(gs.cv_results_['mean_test_score'])
        
        best_index = np.argmax(gs.cv_results_['mean_test_score'])
        print(gs.cv_results_['params'][best_index])
        ```
        
        ```python
        params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001), # 총 9개
                            'max_depth' : range(5, 20, 1), # 총 15개
                            'min_samples_split' : range(2, 100, 10)} # 총 10개
        
        gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
        gs.fit(train_input, train_target)
        
        print(gs.best_params_)
        print(np.max(gs.cv_results_['mean_test_score']))
        ```
        
        ```python
        from scipy.stats import uniform, randint
        
        rgen = randint(0, 10) # 정수 뽑음
        rgen.rvs(10)
        
        np.unique(rgen.rvs(1000), return_counts=True)
        ```
        
        ```python
        ugen = uniform(0, 1) # 실수 뽑음
        ugen.rvs(10)
        ```
        
        ```python
        params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
                            'max_depth' : randint(20, 50),
                            'min_samples_split' : randint(2, 25),
                            'min_samples_leaf' : randint(1, 25)}
        
        from sklearn.model_selection import RandomizedSearchCV
        
        gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
        gs.fit(train_input, train_target)
        
        print(gs.best_params_)
        print(np.max(gs.cv_results_['mean_test_score']))
        
        dt = gs.best_estimator_
        print(dt.score(test_input, test_target))
        ```
        
    
    [05_2.ipynb](05_2.ipynb)