# Chapter 05-3

1. **정형 데이터 (structed data)**

1. csv 파일 같은 거에 가지런히 정리되어 있는 데이터 → **어떤 구조**로 되어 있음
2. csv, 데이터베이스, 엑셀에 저장하기 쉬움
3. 가장 뛰어난 성과를 가진 알고리즘 → **앙상블 학습 (ensemble learning)**

1. **비정형 데이터 (unstructed data)**
    1. 데이터베이스나 엑셀로 **표현하기 어려운** 것들
    2. 텍스트, 사진, 음악 등
    3. 신경망 알고리즘

1. **랜덤 포레스트 (Random Forest)**
    1. **결정 트리를 랜덤**하게 만들어 숲을 만듦 → 각 결정 트리의 예측을 사용해 최종 예측 만듦
    2. **부트스트랩 샘플 (bootstrap sample)**
        1. **중복 허용**인 랜덤 샘플 뽑기
        2. 포함되지 않고 남는 샘플 → **OOB(out of bag) 샘플**
    3. 훈련 세트에 과대적합 되는 것 막아주고 안정적인 성능 얻을 수 있음

1. **엑스트라 트리 (Extra Tree)**
    1. 부트스트랩 샘플 사용 x → 결정 트리를 만들 때 **전체 훈련 세트 사용**
    2. 노드 분할 시 가장 좋은 분할 x **무작위 분할** o
    3. 무작위성이 조금 더 큼 → 더 많은 결정 트리 훈련 필요, 빠른 계산 속도

1. **그레이디언트 부스팅 (Gradient Boosting)**
    1. **깊이가 얇은** 결정 트리를 사용하여 이전 트리의 오차 보완
    2. 깊이가 얕은 결정 트리 사용 → 과대적합에 강함, 높은 일반화 성능
    3. **경사하강법** 사용하여 트리를 앙상블에 추가
    4. **subsample : 훈련 세트의 비율 정함**
    5. 성능이 높지만 순서대로 트리를 추가하기 때문에 훈련 속도 느림

1. **히스토그램 기반 그레이디언트 부스팅 (Histogram-based Gradient Boosting)**
    1. 입력 특성 256개의 **구간으로 나눔** (최적 분할 빨리 찾을 수 있음, 누락된 값 있어도 전처리 필요 x)
    2. HistGradientBoostingRegressor뿐만 아니라 XGBoost, LightGBM에도 구현되어 있음

1. 실습
    - 코드
        
        ```python
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        wine = pd.read_csv('https://bit.ly/wine_csv_data')
        data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
        target = wine['class'].to_numpy()
        train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
        
        from sklearn.model_selection import cross_validate
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_jobs=-1, random_state=42)
        scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
        print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        
        rf.fit(train_input, train_target)
        print(rf.feature_importances_)
        ```
        
        ```python
        rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
        rf.fit(train_input, train_target)
        print(rf.oob_score_)
        ```
        
        ```python
        from sklearn.ensemble import ExtraTreesClassifier
        
        et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
        scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
        print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        
        et.fit(train_input, train_target)
        print(et.feature_importances_)
        ```
        
        ```python
        from sklearn.ensemble import GradientBoostingClassifier
        
        gb = GradientBoostingClassifier(random_state=42)
        scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
        print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        
        gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
        scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
        print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        
        gb.fit(train_input, train_target)
        print(gb.feature_importances_)
        ```
        
        ```python
        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingClassifier
        
        hgb = HistGradientBoostingClassifier(random_state=42)
        scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
        print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        
        from sklearn.inspection import permutation_importance
        
        hgb.fit(train_input, train_target)
        result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
        print(result.importances_mean)
        
        result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
        print(result.importances_mean)
        
        hgb.score(test_input, test_target)
        ```
        
        ```python
        from xgboost import XGBClassifier
        
        xgb = XGBClassifier(tree_method='hist', random_state=42)
        scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
        print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        
        from lightgbm import LGBMClassifier
        
        lgb = LGBMClassifier(random_state=42)
        scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
        print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        ```
        
    
    [05_3.ipynb](05_3.ipynb)