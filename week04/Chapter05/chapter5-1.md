
# Chapter 5-1

[혼공머5_1.ipynb](%ED%98%BC%EA%B3%B5%EB%A8%B85_1.ipynb)

### 결정트리(decision tree)

- 예/아니오에 대한 질문을 이어나가며 정답을 찾아 학습하는 알고리즘
- 노드: 결정트리를 구성하는 핵심요소로 훈련데이터의 특성에 대한 테스트를 표현
- 노드에 대한 의사결정의 방향은 “가지”로 나타낸다.

- img) 결정트리의 예시

<img width="794" height="559" alt="image" src="https://github.com/user-attachments/assets/00d85bde-50a1-4955-951f-faf57f5449bc" />


- max_depth를 조정하면 아래와 같이 하나의 노드를 더 확장하여 만들 수 있다.
- 아래의 노드에는 테스트 조건, 불순도, 총 샘플수, 클래스별 샘플수를 표현한다.
- 분기를 통해 나눈 각각의 노드에는 filled = True를 사용하여 데이터의 클래스 비율에 따른 색상표현이 가능하여 보기 간편해진다.

### 지니 불순도 (gini)

- 지니 불순도는 주어진 데이터 집합의 불확실성 또는 순도를 수치화한 것으로 한 데이터 집합에 다양한 클래스(또는 레이블)가 얼마나 섞여 있는지를 나타낸다.
    
<img width="524" height="54" alt="image 1" src="https://github.com/user-attachments/assets/e56a9aea-cffb-404c-ab73-29856b167d91" />

    

- 부모노드와 자식노드의 불순도 차이가 가능한 크도록 트리를 성장시켜야한다.

<img width="904" height="102" alt="image 2" src="https://github.com/user-attachments/assets/d659fb81-d43d-4107-8c45-fa39a07896ce" />


- 위와 같은 부모노드와 자식노드 사이의 불순도 차이를 “정보이득”이라고 한다.

### 엔트로피 불순도

<img width="1022" height="60" alt="image 3" src="https://github.com/user-attachments/assets/fddc1dc8-714c-44eb-9a98-d0710d78a6dc" />


### 가지치기

- decision tree를 만들때 계속 tree가 생성되면 test set에만 잘 맞는 과대적합 모델이 생성 될 수 있으므로 적당한 깊이의 tree를 만들어주기 위해 가지치기를 해야한다.
- tree의 max depth를 지정하여 가지치기를 할 수 있다.

### 결정트리의 전처리과정 불필요성

- 결정트리는 각 노드에서 특성값을 기준으로 임계값을 정하고, 그 임계값을 기준으로 분기하기 때문에 스케일(값의 단위나 크기)이 달라도 구조나 판단에 영향을 주지 않는다.
