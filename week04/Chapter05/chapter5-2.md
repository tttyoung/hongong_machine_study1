
# Chapter 5-2

[혼공머5_2.ipynb](%ED%98%BC%EA%B3%B5%EB%A8%B85_2.ipynb)

### 검증세트(validation set)

<img width="648" height="148" alt="image" src="https://github.com/user-attachments/assets/11afbe42-0da3-43ee-ae72-dbd341bb15a2" />


- 테스트세트만 사용하면 반복되는 학습을 통해 학습된 모델은 과대적합한 방향의 모델이 된다. → 테스트세트를 가능한 사용하지 말아야한다.
- 테스트세트의 일부를 검증세트로 분리하여 검증세트를 통한 학습의 결과로 모델의 방향을 조정해나간다.

### 교차검증

<img width="860" height="426" alt="image 1" src="https://github.com/user-attachments/assets/4dd0f46a-44e8-4da4-9dd8-8d28cbff7127" />


- 훈련세트를 여러 폴드로 나눈 후, 하나의 폴드는 검증세트 나머지 폴드들은 테스트세트의 역할로 모델이 학습한다.
- 몇개의 폴드로 나누는지에 따라 k-겹 교차검증이라고 부른다. (보통은 5 / 10폴드를 사용한다)

```
from sklearn.model_selectionimport StratifiedKFold
scores= cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
```

- StratifiedKFold()를 통해 교차검증 시 훈련세트를 섞는 분할기를 지정할 수 있다.

### 하이퍼파라미터 튜닝

- 하이퍼파라미터: 모델이 학습 할 수 없어 사용자가 지정해야하는 파라미터 → 클래스/ 메서드의 매개변수로 표현

### 그리드 서치

- 하이퍼파리미터 탐색을 자동화해주는 도구로 탐색할 매개변수를 나열하면 교차검증을 수행하여 최적의 매개변수 조합을 선택

1. 탐색할 매개변수를 지정
2. 훈련세트에서 그리드 서치를 수행하여 최상의 평균 검증 점수가 나오는 매개변수 조합을 찾고 이 조합을 그리드 서치 객체에 저장
3. 그리드 서치는 최상의 매개변수에서 전체 훈련세트를 사용해 최종 모델 훈련

- 그리드 서치에서 탐색할 파라미터 넘겨주기

```
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001), #0.0001~0.001까지 0.0001씩 더한 원소가 9개인 배열
          'max_depth': range(5, 20, 1), #정수만
          'min_samples_split': range(2, 100, 10) #정수만 
          }
```

### 랜덤서치

- 탐색할 값을 직접 나열하는것이 아닌 탐색값을 샘플링 할 수 있는 확률분포 객체를 전달
- 위 코드와 같이 특정 범위내에 특정 간격을 기준으로 넘겨주는것이 아닌 scipy를 사용하여 random한 값을 넘겨준다.
- 매개변수가 수치형이고 연속적인 실숫값일때 유용하다.
