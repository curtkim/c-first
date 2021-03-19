## reference
https://cs.lmu.edu/~ray/notes/nasmtutorial/

## howto

    nasm -felf64 hello.asm && ld hello.o && ./a.out

    nasm -felf64 hola.asm && gcc -no-pie hola.o && ./a.out

    nasm -felf64 fibo.asm && gcc -no-pie fibo.o && ./a.out