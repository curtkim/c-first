#include <stdio.h>
#include <stdlib.h>
 
int main (int argc, char *argv[]) {
 
    if (3 != argc) {
        return 0;
    }
 
    const int Question = atoi( argv[1] );
    if (Question <= 0) {
        return 0;
    }
 
    const int Answer = atoi( argv[2] );
     
    // Calculate Fibonacci number
    int Prev = 0, Fibo = 1, Temp = 0;
    for ( int Index = 1 ; Index < Question ; Index++ ) {
        Temp = Fibo;
        Fibo = Prev + Fibo;
        Prev = Temp;
    }
 
    if (Answer == Fibo) {
        printf("Correct : %d \n", Answer);
    } else {
        printf("Incorrect : %d != (%d) \n", Answer, Fibo);
    }
 
    return 0;
}

