---
layout: post
title: "추천 시스템 기본 - 협업 필터링"
---
출처: https://kmhana.tistory.com/31

# 추천 알고리즘의 기본
> 1. 협업 필터링(Collaborative Filtering)
>> • Memory Based Approach
>>> - User-based Filtering
>>> - Item-based Filtering
>> • Model Based Approach
>>> - 행렬 분해(Matrix Factorization)
> 2. 콘텐츠 필터링(Contents-Based Filtering)

# 협업 필터링 이란 ?

#### 영화를 예시로 들때, 볼만한 영화를 어떻게 찾을까요?
> 1. 내가 좋아하는 감독, 장르, 키워드의 영화를 찾아본다
>> * Content Based Filtering
> 2. 나랑 성향이 비슷한 친구들이 본 영화를 찾아본다
>> * 협업 필터링(Collaborative Filetering)

# 협업 필터링(Collaborative Filtering) 특징
### 가정 : 나와 비슷한 취향의 사람들이 좋아하는 것은 나도 좋아할 가능성이 높다
> - 많은 사용자로 부터 얻은 취향 정보를 활용
* 핵심 포인트 : "많은 사용자들"로 부터 얻은 취향 정보를 활용
> - 사용자의 취향 정보 = 집단 지성 
> - 축적된 사용자들의 집단 지성을 기반으로 추천
> - 예를 들어 : A 상품을 구매한 사용자가, 함께 구매한 다른 상품들