#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>

#define __USE_GNU

#include <sched.h>


void print_using_cpu() {
    unsigned cpu, node;

    // Get current CPU core and NUMA node via system call
    // Note this has no glibc wrapper so we must call it directly
    syscall(SYS_getcpu, &cpu, &node, NULL);

    // Display information
    printf("This program is running on CPU core %u and NUMA node %u.\n\n", cpu, node);
}

int main() {
    print_using_cpu();

    unsigned long mask = 1; /* processor 0 */
    if (sched_setaffinity(0, sizeof(mask), (cpu_set_t *) &mask) < 0) {
        perror("sched_setaffinity");
    }

    print_using_cpu();

    return 0;
}