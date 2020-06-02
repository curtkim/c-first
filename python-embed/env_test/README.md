## 개요
python embedding이 pyenv로 설치한 경우에 안되는 문제를 탐구하기 위한 프로젝트

## 결론
pyenv install할때 PYTHON_CONFIGURE_OPTS="--enable-shared"을 주어야 한다. ㅠㅠ 

## 참고
https://github.com/pyenv/pyenv/wiki#how-to-build-cpython-with---enable-shared

## Howto

    docker build .
    docker build . -f Dockerfile_pyenv

    