#include <stdio.h>
#include <math.h>
#include <unistd.h>

#define __USE_GNU

#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>

double waste_time(long n) {
    double res = 0;
    long i = 0;
    while (i < n * 200000) {
        i++;
        res += sqrt(i);
    }
    return res;
}

void *thread_func(unsigned long mask) {
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), (cpu_set_t *) &mask) < 0) {
        perror("pthread_setaffinity_np");
    }

    unsigned cpu, node;
    syscall(SYS_getcpu, &cpu, &node, NULL);
    printf("This program is running on CPU core %u and NUMA node %u.\n", cpu, node);

    /* waste some time so the work is visible with "top" */
    printf("processor=%ld result: %f\n", mask, waste_time(2000));
    return 0;
}

int main(int argc, char *argv[]) {
    pthread_t my_thread1;
    if (pthread_create(&my_thread1, NULL, thread_func, 1) != 0) {
        perror("pthread_create");
    }
    pthread_t my_thread2;
    if (pthread_create(&my_thread2, NULL, thread_func, 2) != 0) {
        perror("pthread_create");
    }

    //pthread_exit(NULL);
    pthread_join(my_thread1, NULL);
    pthread_join(my_thread2, NULL);
    return 0;
}
