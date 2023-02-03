---
layout: posts
title: "월간 데이콘 TV 손동작 제어 인식 AI 경진대회"
categories: Project
tag: [python, ML, Project]
toc: true
---

> <데이터셋> - hhttps://dacon.io/competitions/official/236050/overview/description

# DACON

---

## 배경

---

[DACON]
안녕하세요 여러분! 😀 월간 데이콘 TV 손동작 제어 인식 AI 경진대회에 오신 것을 환영합니다.
이번 월간 데이콘에서는 사용자가 리모컨을 사용하지 않고 TV를 제어할 수 있도록 사용자가 수행하는 5가지 손동작을 인식할 수 있는 스마트 TV의 기능을 개발하려고 합니다.
데이커 여러분들의 AI 모델로 스마트 TV의 손동작 인식 제어 기능 개발에 기여해주세요.

**TV를 제어하는 사용자의 손동작을 분류하는 AI 모델 개발**

# 프로젝트 진행 과정

1. 참고자료

2. 영상 속 mediapipe를 이용하여 손 좌표 추출

3. 각 좌표간 손가락 각도를 계산하여 시퀀스 데이터로 생성

4. 해당 시퀀스 데이터로 LSTM을 사용하여 학습 진행

5. 학습된 모델을 불러와 테스트 데이터에 라벨링 작업 시행

6. 도출된 결과물 제출하여 점수 측정

# 참고

- [논문] Fast Hand-Gesture Recognition Algorithm For Embedded System
- [영상] 가위바위보 기계 만들기 - 손가락 인식 인공지능 [
  빵형의 개발도상국] - https://www.youtube.com/watch?v=udeQhZHx-00
- [영상] 손 제스처 인식 딥러닝 인공지능 학습시키기 [
  빵형의 개발도상국] - https://www.youtube.com/watch?v=eHxDWhtbRCk&t=4s
- [논문] Dynamic Hand Gesture Recognition Using 3DCNN and LSTM with FSM Context-Aware Model
- [블로그] [🔥포스🔥] Multi-Hand Gesture Recognition(1) (LSTM, 과적합 방지) - https://dacon.io/codeshare/4956

# 데이터 소개

TV를 제어하는 사용자의 손동작을 분류하기 위해서 30프레임의 1초 분량의 동영상(mp4)들이 입력 데이터로 주어지며,
동영상을 입력으로 받아 사용자의 손동작을 5가지의 Class로 분류하는 AI 모델을 개발해야합니다.

Class 0 : 스마트 TV 볼륨을 높입니다.
Class 1 : 스마트 TV 볼륨을 낮춥니다.
Class 2 : 스마트 TV의 재생 영상을 10초 전으로 점프합니다.
Class 3 : 스마트 TV의 재생 영상을 10초 앞으로 점프합니다.
Class 4 : 스마트 TV의 재생 영상을 중지합니다.

# 코드
