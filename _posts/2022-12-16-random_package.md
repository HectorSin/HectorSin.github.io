---
layout: posts
title: "랜덤(random) 모듈"
categories: Python
tag: [python, coding, package]
toc: true
---

# 랜덤(random) 모듈

출처: https://wikidocs.net/79 [왕초보를 위한 Python: 쉽게 풀어 쓴 기초 문법과 실습]

파이썬에서의 랜덤(random) 함수

주사위를 던지는 상황을 생각해봅시다. 주사위의 각 면에는 1개에서 6개까지의 눈이 새겨져 있어서, 주사위를 던질 때마다 그 중 하나의 숫자가 선택됩니다.

주사위를 직접 던져보기 전에는 다음번에 어떤 숫자가 나올지 알 수가 없죠.

그런데 주사위를 600번 정도 던져보면 각 숫자가 대략 100번 정도는 나오기는 합니다.

이런 것이 바로 **난수(random number)**입니다.

난수의 예가 될 만한 것으로 주사위 외에 또 어떤 것들이 있을까요? 복권 추첨, 음악 CD의 재생 순서 섞기...

그럼 파이썬으로 난수를 만들어봅시다.

```
>>> import random
>>> random.random()
# 0.90389642027948769
```
random 모듈의 random() 함수를 호출했더니 복잡한 숫자를 돌려주네요. random() 함수는 **0 이상 1 미만**의 숫자 중에서 아무 숫자나 하나 뽑아서 돌려주는 일을 한답니다.

주사위처럼 1에서 6까지의 **정수 중 하나를 무작위**로 얻으려면 어떻게 해야 할까요? 이럴 때 편리하게 쓸 수 있는 randrange()라는 함수가 있습니다.

```
>>> random.randrange(1,7)
# 6
>>> random.randrange(1,7)
# 2
```
여기에서 randrange(1,6)이 아니라 randrange(1,7)이라고 썼다는 점에 주의하세요.

"1 이상 7 미만의 난수"라고 생각하시면 이해가 쉽습니다.

내장함수인 range()를 되새겨보는 것도 좋겠군요.

```
>>> range(1,7)
[1, 2, 3, 4, 5, 6]
```

**shuffle()**이라는 재미있는 함수도 있군요. 시퀀스를 뒤죽박죽으로 섞어놓는 함수입니다.

```
abc = ['a', 'b', 'c', 'd', 'e']
random.shuffle(abc)
abc
# ['a', 'd', 'e', 'b', 'c']
```

```
random.shuffle(abc)
abc
# ['e', 'd', 'a', 'c', 'b']
```
아무 원소나 하나 뽑아주는 **choice()** 함수도 있네요.

```
abc
# ['e', 'd', 'a', 'c', 'b']
random.choice(abc)
# 'a'
random.choice(abc)
# 'd'
```
```
menu = '쫄면', '육계장', '비빔밥'
random.choice(menu)
# '쫄면'
```
참과 거짓 중에 하나를 뽑고 싶다면

```
random.choice([True, False])
# True
random.choice([True, False])
# False
```