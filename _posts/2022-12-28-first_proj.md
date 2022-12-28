---
layout: posts
title: "CNN을 활용한 감정분석 + Open Pose를 활용한 감정 캐릭터 얼굴 트래킹"
categories: Project
tag: [python, ML, Project]
toc: true
---

## 참고 자료

> 참고 - https://www.youtube.com/watch?v=tpWVyJqehG4 [얼굴인식 스노우 카메라 쉽게 따라만들기 - Python]
> 참고 - https://medium.com/analytics-vidhya/facial-expression-detection-using-machine-learning-in-python-c6a188ac765f [Facial expression detection using Machine Learning in Python]
> 참고 - https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/ [Face Recognition using Principal Component Analysis]
> 참고 - A Comparative Study on Facial Recognition Algorithms [논문]
> 참고 - https://www.geeksforgeeks.org/cropping-faces-from-images-using-opencv-python/ [Cropping Faces from Images using OpenCV – Python]
> 참고 - https://medium.com/@jsflo.dev/training-a-tensorflow-model-to-recognize-emotions-a20c3bcd6468 [Training a TensorFlow model to recognize emotions]
> 참고 - https://bskyvision.com/entry/python-cv2imread-%ED%95%9C%EA%B8%80-%ED%8C%8C%EC%9D%BC-%EA%B2%BD%EB%A1%9C-%EC%9D%B8%EC%8B%9D%EC%9D%84-%EB%AA%BB%ED%95%98%EB%8A%94-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95 [cv2.imread 한글 경로 인식 문제 해결법]


# 데이터 출처

[AIHub]

# 데이터 압축

원본 데이터의 크기가 약 500기가되기에 ssd작업공간에서 작업하기 힘들것으로 파악되어 open cv를 활용하여 JPG파일을 JPEG 파일 형식으로 압축하였다. 이때 CNN 분석에서 JPEG Quality를 20으로 낮춰도 분류 모델에만 사용한다는 가정하에 문제없다는 논문을 읽어 Quality 40으로 압축 진행(데이터 크기가 거의 1/8사이즈로 줄었다)

```{python}
import cv2
import os
import numpy as np

# upset: 당황, pleasure: 기쁨, anger: 분노, unrest: 불안, wound: 상처, sadness: 슬픔, neutrality: 중립

# 데이터 압축 함수
def convert_file(number,save_start, trial, feeling, filename):
    for a in range(number):
        if a is not 0:
            img = "C:/Users/Administrator/Desktop/proj/Training/{}_TRAIN_{}/{} ({}).jpg".format(feeling, trial, filename, str(a+1))
            print(img)
            img_ori = cv2.imread(img)
            img_name = "Training\\{}\\{} ({}).jpg".format(feeling, feeling, str(a+save_start))
            cv2.imwrite(img_name, img_ori, [cv2.IMWRITE_JPEG_QUALITY, 40])
# 자동화를 위한 기초 포멧
save_start = 1
number = 0

# 각 폴더에 있는 사진개수 리스트
numbers = [9913,5727,11579,3873,11017,4757,7941,3727]
# 인코딩할 감정
feeling = "sadness"

save_start = save_start + number
number = numbers[0]
filename = "{}1".format(feeling)
trial = "01"
convert_file(number, save_start, trial, feeling, filename)

save_start = save_start + number
number = numbers[1]
filename = "{}2".format(feeling)
trial = "01"
convert_file(number, save_start, trial, feeling, filename)

save_start = save_start + number
number = numbers[2]
filename = "{}1".format(feeling)
trial = "02"
convert_file(number, save_start, trial, feeling, filename)

save_start = save_start + number
number = numbers[3]
filename = "{}2".format(feeling)
trial = "02"
convert_file(number, save_start, trial, feeling, filename)

save_start = save_start + number
number = numbers[4]
filename = "{}1".format(feeling)
trial = "03"
convert_file(number, save_start, trial, feeling, filename)

save_start = save_start + number
number = numbers[5]
filename = "{}2".format(feeling)
trial = "03"
convert_file(number, save_start, trial, feeling, filename)

save_start = save_start + number
number = numbers[6]
filename = "{}1".format(feeling)
trial = "04"
convert_file(number, save_start, trial, feeling, filename)

save_start = save_start + number
number = numbers[7]
filename = "{}2".format(feeling)
trial = "04"
convert_file(number, save_start, trial, feeling, filename)
```

# 데이터 크롭

감정 분석에 있어 노이즈 제거를 위해 데이터 크롭을 진행하였다. 머리카락, 옷, 악세서리, 배경등은 감정 분석에 큰 영향을 끼치지 않는다 판단하고 오로지 눈코입 얼굴 표정만으로 학습을 진행하기로 결정.

> [참고] https://www.tutorialspoint.com/how-to-crop-and-save-the-detected-faces-in-opencv-python [How to crop and save the detected faces in OpenCV Python?]

```
import cv2

def crop_file(emotion, number):
    for n in range(number):
        route = 'Training\\{}\\{}1 ({}).jpg'.format(emotion, emotion, n+1)
        img = cv2.imread(route)
        print(route)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            faces = gray[y:y + h, x:x + w]
            #cv2.imshow("face",faces)
            cv2.imwrite('crop\\{}\\{}{}.jpg'.format(emotion, emotion, n+1), faces)


crop_file("anger", 53591)
crop_file("neutrality", 57935)
crop_file("pleausre", 49198)
crop_file("sadness", 58526)
crop_file("unrest", 46495)
crop_file("upset", 47016)
crop_file("wound", 54992)
```
이때 이미지 크롭을 흑백 사진으로 진행하였는데 이유는 색의 유무에 따라 차원수가 달라지고 차원이 늘어나면 날수록 처리해야하는 데이터 양이 기하급수적으로 늘어나기에 흑백으로 진행하였다.
(또한 감정에 있어 색은 특수한 경우를 제외하고는 (연극이나 영화의 연출) 분석에 큰 영향을 끼치지 않을꺼라 판단)

<img src="/images/2022-12-28-first_proj/크롭이미지화면.png" alt="크롭이미지화면" style="zoom:80%;" />

또한 크롭한 이미지 중에 위 그림처럼 얼굴을 제대로 추출하지 못한 사진들이 존재한다.

<img src="/images/2022-12-28-first_proj/anger114.jpg" alt="anger114" style="zoom:200%;" />



이렇게 크롭된 이미지인데 원본이 궁금하여 밑에 업로드하였다.



![잘못크롭된사진](/images/2022-12-28-first_proj/잘못크롭된사진.png)

자세히 보면 잘못 크롭된 이미지에 눈코입의 윤곽선이 살짝 보이는것을 확인할 수 있다. 처음 opencv로 얼굴 위치를 인식할때 하나의 얼굴만이 아닌 복수의 얼굴좌표를 인식하게 한 다음 잘못 크롭된 이미지들만 삭제하는 방법이 있지만 데이터양이 이미 5만장을 넘어가고 대다수의 이미지가 제대로 크롭되는 것을 확인하였기에 진행하지 않았습니다.