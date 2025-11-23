# Chapter 6-1

### 비지도 학습

- 훈련 데이터에 타깃값이 없기 때문에 정답(레이블)이 없는 데이터에서 스스로 패턴이나 구조를 학습하는 기계 학습 방법

<img width="800" height="416" alt="image" src="https://github.com/user-attachments/assets/49cbabf8-c48c-40f7-a535-0f87ae68e1fb" />

### 군집

- 데이터 내에서 비슷한 샘플끼리 그룹으로 모으는 작업
- 목표는 같은 군집 내의 데이터는 서로 유사하고, 다른 군집의 데이터는 서로 유사하지 않도록 데이터를 분류하는 것
- 클러스터: 군집 알고리즘에서 만든 그룹

### 실습

- 사진 데이터 사용
    - 컴퓨터는 왜 255에 가까운 바탕에 집중하는가?
        
        알고리즘이 어떤 출력을 만들기 위해 곱셈, 덧셈을 한다. 픽셀값이 0이면 출력도 0이 되어 의미가 없다. 픽셀값이 높으면 출력값도 커지기 때문에 의미를 부여하기 좋다.
        

```python
plt.imshow(fruits[0], cmap='gray')
plt.show()
 
plt.imshow(fruits[0], cmap='gray_r')
plt.show()
```
<img width="426" height="420" alt="image 1" src="https://github.com/user-attachments/assets/2cd3ec6b-6b34-4d31-82b3-befa6a873ccf" />

<img width="760" height="422" alt="image 2" src="https://github.com/user-attachments/assets/3cc9ad69-516a-4edd-b61c-89287f5dde4a" />

- **axis인수**: 배열의 축
    
    axis = 1일때는 열방향, axis = 0일때는 행방향으로 계산
    

### 픽셀 평균값과 가까운 샘플을 통해 그룹 만들기

- 실습코드를 보면 apple_mean과 샘플 픽셀값의 오차를 계산하여 만든 배열을 통해 절댓값 오차가 작은 샘플들을 apple그룹으로 만들려고 한다.

```python
apple_index = np.argsort(abs_mean)[:100]
```

- np.argsort를 통해 정렬된 절댓값 오차 배열에서 100개만큼만 슬라이싱하여 받는다.

[혼공머6_1.ipynb](%ED%98%BC%EA%B3%B5%EB%A8%B86_1.ipynb)
