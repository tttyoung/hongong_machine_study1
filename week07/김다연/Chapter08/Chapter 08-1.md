# Chapter 08-1

1. **합성곱 (convolution)**
    1. 입력 데이터 전체에 가중치를 적용하는 것이 아니라 **일부에 가중치** 곱함
    2. **한 칸씩 아래로 이동**하면서 출력 만듦
        
        ![image.png](https://github.com/user-attachments/assets/94bc426b-b497-45ce-9b49-bb90b3fdfe68)

        
        ![image.png](https://github.com/user-attachments/assets/a9f41d77-ca2c-4bd1-94df-b12d54110ec2)

        

1. **합성곱 신경망 (CNN)**
    1. 1개 이상의 합성곱 층을 쓴 인공 신경망
    2. 뉴런 → **필터(filter)** or **커널(kernel)**
    3. **특성 맵 (feature map)** : 합성곱 계산을 통해 얻은 출력
        
        ![image.png](https://github.com/user-attachments/assets/c3b940cf-2596-4801-a28f-47fddfd707e8)
        
2. **패딩 (padding)**
    1. 출력 크기 늘리기 위해 **입력 배열 주위를 가상의 원소**로 채우는 것

       ![image.png](https://github.com/user-attachments/assets/5b86b78b-ed58-4f4b-a6db-3ae3655f05e6)
        
    3. 실제 입력값이 아니기 때문에 **패딩은 0으로** 채움
    4. **세임 패딩 (same padding)**
        1. 입력과 특성 맵의 크기를 동일하게 만들기 위해 입력 주위에 0으로 패딩하는 것
        2. 입력과 특성 맵의 크기를 동일하게 만듦
    5. **밸리드 패딩 (valid padding)**
        1. 패딩 없이 순수한 입력 배열에서만 합성곱하여 특성 맵 만드는 경우
        2. 특성 맵의 크기가 줄어듦
    6. 패딩 사용 이유
        1. 특성 맵에서 가운데에 있는 정보는 2번 이상 커널과 계산 → 두드러지게 표현
        2. 모서리에 있는 것은 1번만 계산 → **중요한 정보 특성 맵으로 잘 전달되지 않을** 수 있음
            
            ![image.png](https://github.com/user-attachments/assets/90d9afb9-5429-4953-b637-340410e79f22)

            

1. **스트라이드 (stride)**
    1. 합성곱 연산의 이동 크기
    2. 기본값 : 1 (잘 수정하지 x)

1. **풀링 (pooling)**
    1. 합성곱 층에서 만든 특성 맵의 **가로세로 크기를 줄이는** 역할 (특성 맵의 개수는 줄이지 x)
    2. **최대 풀링 (max pooling)** : 도장을 찍은 영역에서 가장 큰 값
    3. **평균 풀링 (average pooling)** : 도장을 찍은 영역에서의 평균값
    4. 가중치, 패딩 x
    5. 풀링 크기 = 스트라이드 크기

1. 합성곱 신경망의 전체 구조
    
    ![image.png](https://github.com/user-attachments/assets/3740a681-1c29-48fa-b6ce-e2324e697936)
