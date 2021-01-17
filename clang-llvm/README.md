https://engineering.linecorp.com/ko/blog/code-obfuscation-compiler-tool-ork-1/

## howto

    clang -E test.c               # 전처리       
    clang -emit-ast test.c        # .ast
    clang -S -emit-llvm test.c    # .ll
    clang -S test.c               # .s
    clang -c test.c               # .o 


## CUDA

    /usr/bin/clang++-11 axpy.cu -o axpy --cuda-gpu-arch=sm_35 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread
