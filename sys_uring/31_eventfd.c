// https://unixism.net/loti/tutorial/register_eventfd.html?highlight=io_uring_register_eventfd
// This capability enables
// processes that are multiplexing I/O using poll(2) or epoll(7)
// to add a io_uring registered eventfd instance file descriptor to the interest list
// so that poll(2) or epoll(7) can notify them
// when a completion via io_uring occurred
#include <sys/eventfd.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <liburing.h>
#include <fcntl.h>

#define BUFF_SZ   512

char buff[BUFF_SZ + 1];
struct io_uring ring;

void error_exit(char *message) {
  perror(message);
  exit(EXIT_FAILURE);
}

void* listener_thread(void* data) {
  struct io_uring_cqe *cqe;
  int efd = (int) data;
  eventfd_t v;
  printf("%s: Waiting for completion event...\n", __FUNCTION__);

  // 무엇이 efd를 trigger한 것인가?
  int ret = eventfd_read(efd, &v);
  // int ret = read(efd, &v, sizeof(uint64_t)); 동일한 효과가 있다.
  if (ret < 0)
    error_exit("eventfd_read");

  printf("%s: Got completion event. v=%d\n", __FUNCTION__, v); // 1이 찍히는 것은 cqe가 1개라는 의미인가?

  int ret2 = io_uring_wait_cqe(&ring, &cqe);
  if (ret2 < 0) {
    fprintf(stderr, "Error waiting for completion: %s\n", strerror(-ret));
    return NULL;
  }

  /* Now that we have the CQE, let's process it */
  if (cqe->res < 0) {
    fprintf(stderr, "Error in async operation: %s\n", strerror(-cqe->res));
  }
  printf("Result of the operation: %d\n", cqe->res);
  io_uring_cqe_seen(&ring, cqe);

  printf("Contents read from file:\n");
  printf("========================\n");
  printf("%s\n", buff);
  return NULL;
}

int main() {
  pthread_t t;
  int efd;

  // 1. Create an eventfd instance
  efd = eventfd(0, 0);  // count?, flag?
  if (efd < 0)
    error_exit("eventfd");
  printf("efd=%d\n", efd);

  // Create the listener thread
  pthread_create(&t, NULL, listener_thread, (void *)efd);

  sleep(2);

  // 2. Setup io_uring instance and register the eventfd
  int ret = io_uring_queue_init(8, &ring, 0);
  if (ret) {
    fprintf(stderr, "Unable to setup io_uring: %s\n", strerror(-ret));
    return 1;
  }

  // 3. register_eventfd
  // it is possible to get notified of completion events on an io_uring instance
  // eventfd를 사용하는 기존 로직이 동작하게 해준다.
  // io_uring_submit이 있으면 write처럼 동작하는 것이 아닐까?
  io_uring_register_eventfd(&ring, efd);

  // 4. init sqe
  struct io_uring_sqe *sqe;
  sqe = io_uring_get_sqe(&ring);
  if (!sqe) {
    fprintf(stderr, "Could not get SQE.\n");
    return 1;
  }

  // 5. io_uring_prep_read
  int fd = open("/etc/passwd", O_RDONLY);
  printf("fd=%d\n", fd);
  io_uring_prep_read(sqe, fd, buff, BUFF_SZ, 0);
  io_uring_submit(&ring);


  // Wait for th listener thread to complete
  pthread_join(t, NULL);

  /* All done. Clean up and exit. */
  io_uring_queue_exit(&ring);
  return EXIT_SUCCESS;
}