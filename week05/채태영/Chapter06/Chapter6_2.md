# Chapter 6-2

1장에서는 어떤 과일이 몇개 있는지 알았지만 일반적인 비지도 학습에서는 알 수가 없다.

### K-평균 알고리즘

- 처음에 무작위 k개의 클러스터 중심을 정한 뒤 점차 가장 가까운 샘플의 중심으로 이동하는 알고리즘
- 작동방식
    1. 무작위로 k개의 클러스터 중심을 정한다.
    2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정한다.
    3.  클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경한다.
    4.  클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복한다

<img width="1032" height="298" alt="image" src="https://github.com/user-attachments/assets/4955ddcb-117f-437a-89aa-be5f85d42c14" />


<img width="794" height="944" alt="image 1" src="https://github.com/user-attachments/assets/e69c0027-1e6b-4ba4-b504-1cc0398d7e5c" />


<img width="794" height="790" alt="image 2" src="https://github.com/user-attachments/assets/6c510745-4b13-409d-b62f-6f6876ffe4f3" />


<img width="794" height="713" alt="image 3" src="https://github.com/user-attachments/assets/5ffd9eb6-4f87-445c-83d6-7d9177162e11" />


### km.labels_의 값 조정에 따른 분류

- k-평균 알고리즘을 통해 분류하였을때 완벽하게 분류되지 않음 → 최적의 k값이 아니기 때문

### 최적의 k찾기

- **이너셔**: k-평균 알고리즘에서 클러스터 중심과 클러스터에 속한 샘플 사이의 거리의 제곱합
    
    → 샘플이 얼마나 가깝게 모였는가를 확인 할 수 있음
    

- **엘보우**
    - 클러스터의 개수가 늘어날 수록 한 클러스터 내에 있는 샘플의 수가 줄어들기 때문에 이너셔 값이 작아짐
    - 클러스터 개수를 늘려가면서 이너셔의 변화를 관찰하여 최적의 클러스터 개수를 찾는 방법

<img width="744" height="434" alt="image 4" src="https://github.com/user-attachments/assets/1f2e0fdf-5134-4bcf-9cd0-42f9d3e23bda" />


