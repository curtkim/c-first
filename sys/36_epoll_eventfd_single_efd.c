// epfd, efd를 파라미터로 넘김
// thread가 sleep해서 0,1,2을 efd에 write한다.
// main thread가 epoll_wait, read해서 value를 출력한다.
// epoll_create, epoll_ctl, epoll_wait
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>

struct timespec tp;

#define log_error(msg, ...)                                                 \
    do {                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &tp);                                \
        if (errno > 0) {                                                    \
            fprintf(stderr, "[%ld.%ld] [ERROR](%s:%d) "msg" [%s:%d]\n",     \
                tp.tv_sec, tp.tv_nsec, __FILE__, __LINE__, ##__VA_ARGS__,   \
                strerror(errno), errno);                                    \
        }                                                                   \
        else {                                                              \
            fprintf(stderr, "[%ld.%ld] [ERROR](%s:%d)"msg"\n",              \
                tp.tv_sec, tp.tv_nsec, __FILE__, __LINE__, ##__VA_ARGS__);  \
        }                                                                   \
        fflush(stdout);                                                     \
    } while (0)


#define exit_error(...)                                                     \
    do { log_error(__VA_ARGS__); exit(EXIT_FAILURE); } while (0)


#define log_debug(msg, ...)                                                 \
    do {                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &tp);                                \
        fprintf(stdout, "[%ld.%ld] [DEBUG] "msg"\n",                        \
            tp.tv_sec, tp.tv_nsec, ##__VA_ARGS__);                          \
        fflush(stdout);                                                     \
    } while (0)


#define MAX_EVENTS_SIZE 10 //1024

typedef struct thread_info {
  pthread_t thread_id;
  int epfd;
  int efd;
} thread_info_t;


static void do_task() {
  return;
}

static void *producer_routine(void *data) {
  struct thread_info *p = (struct thread_info *) data;
  int ret;
  uint64_t i = 0;

  while (1) {
    uint64_t v = i++ % 3 + 1;
    ret = write(p->efd, &v, sizeof(uint64_t));
    log_debug("[producer] write: value=%d", v);
    if (ret != 8)
      log_error("[producer] failed to write eventfd");

    sleep(1);
  }
}

int main(int argc, char *argv[]) {
  int ret;

  // create epoll fd
  int epfd = epoll_create1(EPOLL_CLOEXEC);
  if (epfd == -1)
    exit_error("epoll_create1: %s", strerror(errno));

  int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd == -1)
    exit_error("eventfd create: %s", strerror(errno));

  struct epoll_event event;
  event.data.fd = efd;
  event.events = EPOLLIN | EPOLLET;
  ret = epoll_ctl(epfd, EPOLL_CTL_ADD, efd, &event);
  if (ret != 0)
    exit_error("epoll_ctl");

  thread_info_t threadInfo;
  threadInfo.epfd = epfd;
  threadInfo.efd = efd;

  // start consumers (as task worker)
  ret = pthread_create(&threadInfo.thread_id, NULL, producer_routine, &threadInfo);
  if (ret != 0)
    exit_error("pthread_create");


  struct epoll_event events[MAX_EVENTS_SIZE];
  uint64_t v;

  for (;;) {
    int nfds = epoll_wait(epfd, events, MAX_EVENTS_SIZE, 2000);
    log_debug("[consumer] nfds=%d", nfds);

    for (int i = 0; i < nfds; i++) {
      if (events[i].events & EPOLLIN) {
        ret = read(events[i].data.fd, &v, sizeof(v));
        if (ret < 0) {
          log_error("[consumer] failed to read eventfd");
          continue;
        }
        log_debug("[consumer] tasks done: value=%d", v);
      }
    }
  }

  return EXIT_SUCCESS;
}