#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/eventfd.h>

#include <thread>
#include <iostream>

#include <liburing.h>


void handle_error(char* reason){
  printf("%s", reason);
  exit(1);
}

void produce(int efd){
  for (uint64_t j = 1; j < 10; j++) {
    sleep(1);

    // io_uring을 사용하지 않고, old style
    ssize_t s = write(efd, &j, sizeof(uint64_t));
    if (s != sizeof(uint64_t))
      handle_error("write");
  }
  printf("Child completed write loop\n");
}


int main(){
  const int QUEUE_DEPTH=1;

  int efd = eventfd(0, EFD_CLOEXEC); //EFD_SEMAPHORE
  if (efd == -1)
    exit(1);

  std::thread t1(produce, efd);

  struct io_uring ring;
  io_uring_queue_init(QUEUE_DEPTH, &ring, 0);

  uint64_t value;
  // submit
  struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
  io_uring_prep_read(sqe, efd, &value, sizeof(value), 0);
  io_uring_submit(&ring);

  while (1) {
    // completion
    struct io_uring_cqe *cqe;
    int ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
      perror("io_uring_wait_cqe");
      return 1;
    }
    if (cqe->res < 0) {
      fprintf(stderr, "Async readv failed.\n");
      return 1;
    }

    std::cout << "value=" << value << ", length=" << cqe->res << std::endl;
    io_uring_cqe_seen(&ring, cqe);

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    io_uring_prep_read(sqe, efd, &value, sizeof(value), 0);
    io_uring_submit(&ring);
  }

  io_uring_queue_exit(&ring);
  close(efd);
}