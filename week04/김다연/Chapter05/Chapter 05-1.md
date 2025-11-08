# Chapter 05-1

1. **결정 트리 (Decision Tree)**
    1. 기본 틀
        
        ![image.png](image.png)
        
    2. **root node** : 맨 위의 노드 (부모 x)
    3. **leaf node** : 맨 아래 노드 (자식 x)
    4. 특성값의 스케일이 계산에 영향 x → **전처리 과정 필요 x**
    5. **특성 중요도**
        1. 어떤 특성이 **가장 유용한지** 나타냄
        2. 이걸 활용하면 결정 트리 모델을 특성 선택에 활용할 수 있음

1. **지니 불순도 (Gini impurity)**
    1. 데이터의 불순도 → 한 노드가 **얼마나 섞여 있는지**
    2. 수식
        
        ![image.png](image%201.png)
        
    3. **클래스가 하나**라면 지니 불순도 : 0 → **순수 노드**
    4. **정보 이득 (information gain)**
        1. 부모, 자식 노드 사이의 불순도 차이
        2. 결정 트리 모델은 **정보 이득이 가능한 크도록** 트리 성장시킴
    5. **entropy 불순도**
        1. 수식
            
            ![image.png](image%202.png)
            
        2. 보통 두 불순도의 결과 차이는 크지 않음

1. 실습
    - 코드
        
        ```python
        import pandas as pd
        wine = pd.read_csv('https://bit.ly/wine_csv_data')
        wine.head() # 0이면 레드, 1이면 화이트wine.info() # 누락값 확인
        ```
        
        ```python
        wine.info() # 누락값 확인
        ```
        
        ```python
        wine.describe()
        ```
        
        ```python
        data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
        target = wine['class'].to_numpy()
        
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
        
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)
        ```
        
        ```python
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(train_scaled, train_target)
        print(lr.score(train_scaled, train_target))
        print(lr.score(test_scaled, test_target))
        ```
        
        ```python
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(train_scaled, train_target)
        
        print(dt.score(train_scaled, train_target))
        print(dt.score(test_scaled, test_target))
        ```
        
        ```python
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        plt.figure(figsize=(10,7))
        plot_tree(dt, max_depth=2, filled=True, feature_names=['alchol', 'sugar', 'pH'])
        plt.show()
        ```
        
        ```python
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(train_input, train_target)
        
        print(dt.score(train_input, train_target))
        print(dt.score(test_input, test_target))
        
        plt.figure(figsize=(20,15))
        plot_tree(dt, filled=True, feature_names=['alchol', 'sugar', 'pH'])
        plt.show()
        ```
        
        ```python
        print(dt.feature_importances_) # 당도가 특성 중요도 가장 높음
        ```
        
    
    [05_1.ipynb](05_1.ipynb)