## compile hooking code

    gcc -shared -fPIC mallochook.c -o mallochook.so -ldl

## RUN with hook

    LD_PRELOAD=./mallochook.so ./program

## reference
- https://sjp38.github.io/ko/post/hooking_library_calls/
