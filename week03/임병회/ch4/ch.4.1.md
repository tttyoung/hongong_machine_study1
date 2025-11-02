# 로지스틱 회귀
## 럭기백의 확률
k-최근접 이웃 클래스 비율=확률?
<img width="548" height="340" alt="image" src="https://github.com/user-attachments/assets/97eff7ec-16a5-4b77-8aff-7a36a5cbbfe9" />
### 데이터 준비
```python
import pandas as pd
fish=pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
```
<img width="465" height="214" alt="image" src="https://github.com/user-attachments/assets/1a97bf58-9bbc-4fc7-ad99-fa4c11a72c0e" />

```pyhton
# species 제외 열선택, 넘파이 배열로 변환
fish_input=fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
#타겟데이터 만들기
fish_target=fish['Species'].to_numpy()
# 훈련 세트와 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target=train_test_split(fish_input,fish_target,random_state=42)
# 테스트 세트 변환 by 훈련세트 통계값
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)
```

### 확률 에측
다중분류 - 타깃 데이터에 2개 이상의 클래스가 포함된 문제<br>
```python
#훈련 후 점수 확인
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled,train_target)
print(kn.score(train_scaled,train_target))
print(kn.score(test_scaled,test_target))
#타깃값 그대로 사이킷런에 전달 시 순서 알파벳 순으로 변경
print(kn.classes_)
#테스트 세트 처음 5개 샘플 예측
print(kn.predict(test_scaled[:5]))
#확률값 반환
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=4))
```
<img width="407" height="97" alt="image" src="https://github.com/user-attachments/assets/b463dfaf-2e01-4a8b-94d7-738af72f4663" />
<img width="514" height="215" alt="image" src="https://github.com/user-attachments/assets/69af4b44-1ab8-42f1-9e56-b2d890c82e16" />

가능한 확률이 적어서 개선 필요<br>

##로지스틱 회귀
<img width="726" height="102" alt="image" src="https://github.com/user-attachments/assets/a5b1d29c-164d-449c-92a7-68301f9542a8" />
z는 확률->0~1이 되어야함.<br>
시그모이드 함수(로지스틱 함수)를 통해 z는 변환가능<br>
<img width="553" height="316" alt="image" src="https://github.com/user-attachments/assets/6e81fabe-0367-459e-9688-d2625e9161ca" />
```python
#시그모이드 함수 출력
import numpy as np
import matplotlib.pyplot as plt
z=np.arrange(-5,5,0.1)
phi=1/(1+np.exp(-z))
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
```

### 로지스틱 회귀->이진 분류
불리언 인덱싱 - T or F로 행선택<br>
```python
# bream or smelt -> T
brema_smelt_indexes=(train_target=='Bream')|(train_target=='Smelt')
train_bream_smelt=train_scaled[brema_smelt_indexes]
target_bream_smelt=train_target[brema_smelt_indexes]
#로지스텍 회귀 훈련
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)
#예측
print(lr.predict(train_bream_smelt[:5]))
#확률
print(lr.predict_proba(train_bream_smelt[:5]))
#계수 확인
print(lr.coef_,lr.intercept_)
#z 출력
decisions=lr.decision_function(train_bream_smelt[:5])
print(decisions)
#변환 by시그모이드 함수
from scipy.special import expit
print(expit(decisions))
```

### 로지스틱 회귀->다중 분류
소프트맥스함수 - 다중 분류 시 z값 확률로 변환<br>
<img width="502" height="41" alt="image" src="https://github.com/user-attachments/assets/c4645a75-bf31-431c-a638-02cc4b79491d" />
<img width="472" height="130" alt="image" src="https://github.com/user-attachments/assets/320c949f-994b-432e-808c-6abd25f303d5" />

```python
# 반복횟수, 규제 고려 데이터
lr=LogisticRegression(C=20,max_iter=1000)
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))
#테스트 세트 첫 5개 샘플 예측 출력
print(lr.predict(test_scaled[:5]))
#예측 확률 출력
proba=lr.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=3))
#클래스 확인
print(lr.clasees_)
# z구하기
decision=lr.decision_function(test_scaled[:5])
print(np.round(decision,decimals=2))
#소프트맥스 함수로 변환
from scipy.special import softmax
proba=softmax(decision,axis=1)
print(np.round(proba,decimals=3))
```
