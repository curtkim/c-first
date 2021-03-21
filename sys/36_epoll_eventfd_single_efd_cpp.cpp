// efd를 파라미터로 넘김
// thread가 sleep해서 0,1,2을 efd에 write한다.
// main thread가 epoll_wait, read해서 value를 출력한다.
// epoll_create, epoll_ctl, epoll_wait

#include <unistd.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>

#include <thread>
#include <iostream>

#define MAX_EVENTS_SIZE 10


void producer_routine(int efd) {
  uint64_t i = 0;

  while (1) {
    uint64_t v = i++ % 3 + 1;
    int ret = write(efd, &v, sizeof(uint64_t));
    std::cout << "[producer] write: value=" << v << "\n";
    if (ret != 8)
      std::cout << "[producer] failed to write eventfd\n";
    sleep(1);
  }
}

int main(int argc, char *argv[]) {
  int ret;

  int epfd = epoll_create1(EPOLL_CLOEXEC);
  if (epfd == -1) {
    exit(1);
  }

  int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd == -1) {
    exit(1);
  }

  struct epoll_event event;
  event.data.fd = efd;
  event.events = EPOLLIN | EPOLLET;
  ret = epoll_ctl(epfd, EPOLL_CTL_ADD, efd, &event);
  if (ret != 0) {
    std::cout << "epoll_ctl\n";
    exit(1);
  }

  std::thread producer_thread = std::thread(producer_routine, efd);

  struct epoll_event events[MAX_EVENTS_SIZE];

  for (;;) {
    int nfds = epoll_wait(epfd, events, MAX_EVENTS_SIZE, 2000);
    std::cout << "[consumer] nfds=" << nfds << "\n";

    for (int i = 0; i < nfds; i++) {
      uint64_t v;
      if (events[i].events & EPOLLIN) {
        ret = read(events[i].data.fd, &v, sizeof(v));
        if (ret < 0) {
          std::cout << "[consumer] failed to read eventfd\n";
          continue;
        }
        std::cout << "[consumer] tasks done: value=" << v << "\n";
      }
    }
  }

  return 0;
}