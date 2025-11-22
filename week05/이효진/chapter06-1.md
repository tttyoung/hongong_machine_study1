# 06-1 | 군집 알고리즘

**비지도 학습** : 정답(label)이 없는 데이터에서 스스로 패턴이나 구조를 찾아 학습하는 것

**군집** : 비슷한 샘플끼리 그룹으로 모으는 것

**클러스터** : 군집 알고리즘에서 만든 그룹

---

### 픽셀 값 분석

1. **각 샘플의 픽셀 평균값** : 하나의 이미지 전체를 보고 모든 픽셀값의 평균을 계산. 한장의 이미지를 대표하는 단일 숫자로 요약하는 것

아래 그림을 보면 바나나는 사진에서 차지하는 영역이 작아 평균값이 작음. 사진에서 차지하는 영역이 작다는 건 어두운곳( =배경,작은값)이 넓고 밝은곳(=과일,큰값)이 적다는 것

<img width="418" height="308" alt="06-1-1" src="https://github.com/user-attachments/assets/6a0ec333-8023-4147-a2f0-6c6158ff8eeb" />


                                                            x : 샘플 하나의 픽셀 평균값  y : 샘플 개수

사과와 파인애플 구분하기 힘듦.

2. **픽셀의 평균값** : 샘플 100장의 같은 위치 픽셀값을 평균낸 것

<img width="547" height="149" alt="06-1-2" src="https://github.com/user-attachments/assets/de73d9eb-3d8c-44ee-ae6a-f3feacbcc9a3" />


<img width="428" height="128" alt="06-1-3" src="https://github.com/user-attachments/assets/5bfd5a6c-d63d-4826-aac2-f700f577dc3e" />


세 과일로 구분 가능함.
