# 트리의 앙상블

## 정형, 비정형 데이터

정형 데이터 - 구조로 되어 있는 데이터<br>
비정형 데이터 - 구조적이지 않아 데이터베이스나 엑셀로 표현하기 어려운 데이터<br>
앙상블 학습 - 정형 데이터를 다루는 데 가장 뛰어난 성과를 내는 알고리즘, 결정 트리 기반<br>

## 랜덤 포레스트

앙상블 학습의 대표 주자 중 하나<br>
랜덤하게 샘플, 특성 사용->과대적합 방지리의 앙상블

## 정형, 비정형 데이터

정형 데이터 - 구조로 되어 있는 데이터<br>
비정형 데이터 - 구조적이지 않아 데이터베이스나 엑셀로 표현하기 어려운 데이터<br>
앙상블 학습 - 정형 데이터를 다루는 데 가장 뛰어난 성과를 내는 알고리즘, 결정 트리 기반<br>

## 랜덤 포레스트

앙상블 학습의 대표 주자 중 하나<br>
랜덤하게 샘플, 특성 사용->과대적합 방지

<img width="395" height="290" alt="image" src="https://github.com/user-attachments/assets/d90e25cd-f261-4474-bc91-83bd4f0c6097" />

부트스트랩 샘플 - 훈련 데이터에서 랜덤하게 복원 추출한 샘플, 훈련 세트의 크기와 같게 만듦<br>
OOB 샘플 - 부트스트랩 샘플에 포함되지 않고 남는 샘플(검증세트 역할 가능)<br>

```python
# 교차 검증
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 특성 중요도 출력
rf.fit(train_input, train_target)
print(rf.feature_importances_)
#OOB 점수 출력
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_)
```

## 엑스트라 트리


랜덤 포레스트에서 부투스트랩 샘플 대신 전체 훈련 세트 사용<br>
특성의 경계값을 무작위로 정함->속도빠<br>


```python
#엑스트라 트리
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

## 그레이디언트 부스팅

깊이가 얕은 결정 트리 사용->이전 트리 오차 보완(직렬적으로 오차 보완)<br>
경사 하강법 사용<br>
과대적합 방지<br>
느리지만 고성능<br>

```python
# 그레이디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 결정트리 개수 늘리기
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```


## 히스토그램 기반 그레이디언트 부스팅

입력 특성 256개 구간으로 분리(최적의 분할 빠르게 찾기위해), 그리에디언트 부스팅에 속도 보완<br>

```python
# 히스토그램 기반 그레이디언트 부스팅
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

