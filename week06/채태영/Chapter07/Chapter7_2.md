# Chapter 7-2

### 은닉층

- 입력층과 출력층 사이에 있는 모든 층
- 은닉층에는 활성화함수가 포함되어있으며 활성화함수가 출력층에 적용될때는 sigmoid함수나 softmax함수로 종류가 제한되었지만 은닉층에서는 활성화함수가 비교적 자유롭다.

<img width="800" height="546" alt="image" src="https://github.com/user-attachments/assets/51986bbd-0a17-4438-89ba-b803b73a8472" />


### 심층신경망

- 2개 이상의 층을 포함하는 신경망, 즉 여러개의 은닉층을 가지고있는 신경망을 의미한다.

```python
model.summary()
```

- 우측 layer에는 입력층을 제외한 층이 순서대로 나열되며 층마다 이름, 클래스, 출력크기, 모델파라미터 개수가 출력된다.

<img width="642" height="374" alt="image 1" src="https://github.com/user-attachments/assets/8c0e6002-cd8b-400a-beff-b1766f37453c" />


### 층 추가하는 다른 방법

- sequential 클래스의 생성자 안에서 클래스의 객체를 바로 만들어 층을 추가할 수 있지만 층이 많아지면 생성자가 매우 길어짐.
- add() 매서드를 사용하여 층을 추가할 수 있음

### 렐루함수(RELU)

- 시그모이드 함수는 양쪽 끝으로 갈수록 그래프가 누워있는 모습이기 때문에 올바른 출력을 만드는데 신속한 대응을 하지 못한다. → 심층신경망일 경우 효과가 누적되어 학습이 더 어려움
- 입력이 양수일때는 입력 통과, 음수일때는 0으로 만들어 위의 단점을 해소함

 

<img width="430" height="314" alt="image 2" src="https://github.com/user-attachments/assets/320cae1e-9945-4078-b26b-96c1dad3073b" />


### 옵티마이저

- 신경망의 가중치와 절편을 학습하기 위한 알고리즘
- 케라스→ 다양한 경사하강법 알고리즘이 구현되어있음 (SGD, 네스테로프 모멘텀, RMSprop, Adam)
- 모델이 최적점에 가까이 갈수록 학습률을 낮출 수 있어 최적점에 수렴할 가능성이 높음. 이런 학습률을 ‘적응형 학습률’이라고 한다.

[혼공머7_2.ipynb](%ED%98%BC%EA%B3%B5%EB%A8%B87_2.ipynb)
