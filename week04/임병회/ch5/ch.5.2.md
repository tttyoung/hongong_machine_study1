# 교차 검증과 그리드 서치

## 검증 세트

<img width="730" height="192" alt="image" src="https://github.com/user-attachments/assets/f217c6fa-7144-4bc8-b4ad-73a6f6328ad9" />

```python
#데이터
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
#특성 배치
data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()
#훈련, 테스트 세트 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
# 검증 세트 분리
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
#결정트리 만들고 평가
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
```

-> 과대적합<br>

## 교차 검증

<img width="1019" height="468" alt="image" src="https://github.com/user-attachments/assets/dc61a820-ecf3-43e5-92bb-17edda1f55ce" />

```python
#교차검증함수
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
#교차검증평균
import numpy as np
print(np.mean(scores['test_score']))
```

## 하이퍼파라미터 튜닝

하이퍼파라미터 - 사용자 지정 파라미터<br>
그리드 서치 - 하이퍼 파라미터 탐색+교차검증 한번에 수행<br>


```python
#그리드서치
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
#그리드서치 객체
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
#훈련
gs.fit(train_input, train_target)
#그리드 훈련 중 가장 높은 훈련 점수 추출
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
#검증 점수 출력
print(gs.cv_results_['mean_test_score'])
#최상의 검증 점수일 때 매개변수 추출
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
```

<img width="797" height="200" alt="image" src="https://github.com/user-attachments/assets/cb06b397-0270-4adb-bbd2-85154915616d" />

### 랜덤 서치
매개 변수 값의 간격 정하지 말고 샘플링 가능한 확률 분포 객체를 전달<br>

```python
#2개의 확률분포 클래스 임포트
from scipy.stats import uniform, randint
#10개 숫자 샘플링
rgen = randint(0,10)
rgen.rvs(10)
#1000개 샘플링
np.unique(rgen.rvs(1000),return_counts=True)
#0~1사이 10개 실수 추출
unif = uniform(0,1)
unif.rvs(10)
#매개변수 범위
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20,50),
          'min_samples_split':randint(2,25),
          'min_samples_leaf':randint(1,25),}
#샘플링 횟수 지정
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input,train_target)
#최적 매개변수 추출
print(gs.best_estimator_)
#최고 교차 검증 점수 확인
print(np.max(gs.cv_results_['mean_test_score']))
#테스트
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
```









