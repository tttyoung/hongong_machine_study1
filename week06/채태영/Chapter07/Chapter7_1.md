# Chapter 7-1

### MNIST 데이터셋

- 딥러닝에서 많이 사용하는 데이터

### 인공신경망

<img width="788" height="560" alt="image" src="https://github.com/user-attachments/assets/0f15a1c3-930d-4197-8440-e47dc15b286c" />


- 생물학적 신경망에서 영감을 받아 만들어진 머신러닝 알고리즘
- 입력층: 원본 데이터(픽셀값 자체)
- 출력층: 아래 그림에 z값으로, z를 계산하고 이를 바탕으로 클래스를 예측하기 때문에 신경망의 최종값을 만듦
- 뉴런(유닛): z값을 계산하는 단위 → 선형계산만 함

- 각 입력층마다 다른 가중치와 절편값을 선형계산하여 target의 class개수만큼의 출력층을 만들어냄
- ex) 784개의 픽셀(특성)을 입력층으로 가지며 각자 다른 w, b를 계산하여(밀집층) z1-z10의 출력층을 생성함

- 텐서플로(tensorflow): 딥러닝 라이브러리
- 케라스(keras): 텐서플로의 고수준 API

- 밀집층: 인공신경망에서 뉴런들이 모두 연결되어있는 완전연결층 → 출력층에 밀집층을 사용할때는 분류하려는 클래스와 동일한 개수의 뉴런을 사용
- 활성화 함수: 소프트맥수 함수와 같이 뉴런의 선형방정식 계산 결과에 적용되는 함수

<img width="826" height="514" alt="image 1" src="https://github.com/user-attachments/assets/d3fff590-1f6e-46f7-afd4-48f53b6ee9aa" />


```python
inputs = keras.layers.Input(shape=(784,)) #입력층 생성
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,)) #밀집층 생성
#10은 뉴런개수, activation은 뉴런 출력에 사용할 함수, input_shape은 입력의 크기

model = keras.Sequential(dense) #입력층과 밀집층을 사용하여 모델 만들기
```

### 원-핫 인코딩

- 다중분류에서는 타킷에 해당하는 확률만 남기기 위해 나머지 타깃에는 0을 해당 타깃에는 1을 곱해줘야하므로 출력층의 활성화 값의 배열과 같은 수의 배열을 만들어주는 원-핫 인코딩을 해야함.
- 텐서플로에서는 ‘sparse_categorical_entropy’ 손실을 지정해주어 이러한 변환과정을 생략할 수 있음.

<img width="500" height="414" alt="image 2" src="https://github.com/user-attachments/assets/b3b0733a-d013-4b42-adc2-e33bfb3a88fa" />


<img width="630" height="386" alt="image 3" src="https://github.com/user-attachments/assets/96da1442-fbf1-410d-9472-07603550c680" />


[혼공머7_1.ipynb](%ED%98%BC%EA%B3%B5%EB%A8%B87_1.ipynb)
