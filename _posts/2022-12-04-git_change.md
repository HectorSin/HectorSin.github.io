---
layout: post
title: "[git] 삭제된 폴더, 파일 반영하기"
---

### 분명히 로컬에서 삭제한 파일인데 원격에 반영되지 않는 경우가 있다.
### git status로 했을 때 삭제 됐다고 뜨는데 add를 해도 안먹고 commit을 해도 반영이 안되는 것이다...

### 이럴 때 유용하게 쓸 수 있는 게 바로 -u 옵션이다.
> add에 -u 옵션을 붙여주면 수정되거나 삭제된 파일을 반영할 수 있다.
```
git add -u
```
> commit 할 때 -a 옵션을 붙이면 삭제된 파일만 반영된다.
```
git commit -a -m "massage"
```
이렇게 진행하고 push까지 진행하면 git에서 삭제한 파일까지 깔끔하게 반영되는 것을 볼 수 있다.


### 출처: https://studyingfox.tistory.com/12