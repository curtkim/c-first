#include <liburing.h>
#include <unistd.h>
#include <sys/signalfd.h>
#include <sys/epoll.h>
#include <sys/poll.h>
#include <stdio.h>

int setup_signal() {
  sigset_t mask;
  sigemptyset(&mask);
  sigaddset(&mask, SIGINT);

  sigprocmask(SIG_BLOCK, &mask, NULL);
  int sfd = signalfd(-1, &mask, SFD_NONBLOCK);
  return sfd;
}

void test_uring(int sfd) {
  struct io_uring ring;
  io_uring_queue_init(32, &ring, 0);

  struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
  io_uring_prep_poll_add(sqe, sfd, POLLIN);
  io_uring_submit(&ring);

  struct io_uring_cqe *cqe;
  io_uring_wait_cqe(&ring, &cqe);
  io_uring_cqe_seen(&ring, cqe);
  io_uring_queue_exit(&ring);
}


void test_epoll(int sfd) {
  int epfd = epoll_create(1);

  struct epoll_event ev, ret;
  ev.data.fd = sfd;
  ev.events = EPOLLIN | EPOLLET | EPOLLONESHOT;
  epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &ev);
  epoll_wait(epfd, &ret, 1, -1);
  close(epfd);
}

void test_workaround(int sfd) {
  int epfd = epoll_create(1);

  struct epoll_event ev;
  ev.data.fd = sfd;
  ev.events = EPOLLIN | EPOLLET | EPOLLONESHOT;
  epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &ev);

  struct io_uring ring;
  io_uring_queue_init(32, &ring, 0);

  struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
  io_uring_prep_poll_add(sqe, epfd, POLLIN);
  io_uring_submit(&ring);

  struct io_uring_cqe *cqe;
  io_uring_wait_cqe(&ring, &cqe);
  io_uring_cqe_seen(&ring, cqe);
  io_uring_queue_exit(&ring);

  close(epfd);
}

int main() {
  int sfd = setup_signal();
  test_uring(sfd);
//    test_epoll(sfd);
//    test_workaround(sfd);
//    test_aio(sfd);
  puts("OK");
  close(sfd);
}