# 결정 트리

## 와인 분류 by 로지스틱 회귀
```python
# 와인데이터
import pandas as pd
wine = pd.read_csv('http://bit.ly/wine_csv_data')
wine.head()
# 넘파이 배열로 변경
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
# 세트분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
# 표준화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
# 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

훈련 및 테스트 세트 모두 점수 낮음=과소적합<br>

### 설명 쉬운 모델과 어려운 모델
<img width="765" height="541" alt="image" src="https://github.com/user-attachments/assets/e7708db2-cb9b-43fd-96c3-70fa4c132c40" />
설명 어려움->어떻게 쉽게 설명가능할까?<br>

## 결정 트리
<img width="366" height="216" alt="image" src="https://github.com/user-attachments/assets/78634af2-65b9-4030-b18a-4cf9ddb01eb8" />

```python
# 결정트리 불러오기
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
# 결정 트리 제한 출력
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```
<img width="543" height="346" alt="image" src="https://github.com/user-attachments/assets/30165bd7-88a0-4139-b44c-157c64d705a9" />

가장 아래 있는 리프 노드에서 가장 많은 클래스가 예측 클래스가 된다.<br>

### 불순도(gini)
<img width="551" height="45" alt="image" src="https://github.com/user-attachments/assets/9ecacd3c-11d0-45a2-a284-47217de28880" />

<img width="688" height="119" alt="image" src="https://github.com/user-attachments/assets/9f416d0e-a9c8-4f1c-bf96-7ee67c64454c" />
정보 이득 - 부모와 자식 노드 사이의 불순도 차이, 최대화 시켜야함<br>

### 가지치기

```python
#3개 가지치기
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```
설명 어려움->결정 트리의 정확도는 표준화와 무관->표준화 전처리 하지 말고 결정 트리
만들면 설명 쉬움.<br>


