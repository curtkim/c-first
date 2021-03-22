#include <stdio.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <liburing.h>


#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (1024 * (EVENT_SIZE + 16))
#define QUEUE_DEPTH 1


void print_inotify_event(char* buffer, int length) {
  printf("==================================\n");

  int i = 0;
  while (i < length) {
    struct inotify_event *event = (struct inotify_event *) &buffer[i];
    printf("wd=%d, event->len = %d, length= %d, i=%d \n", event->wd, event->len, length, i);

    if (event->len) {
      if (event->mask & IN_CREATE) {
        printf("The file %s was created.\n", event->name);
      } else if (event->mask & IN_DELETE) {
        printf("The file %s was deleted.\n", event->name);
      } else if (event->mask & IN_MODIFY) {
        printf("The file %s was modified.\n", event->name);
      }
      else {
        printf("mask = %d", event->mask);
      }
    }

    i += EVENT_SIZE + event->len;
  }
  printf("----------------------------------\n");
}

int main(int argc, char **argv) {
  int length, i = 0;
  int fd;
  int wd;
  char buffer[BUF_LEN];

  fd = inotify_init();
  printf("after init\n");

  if (fd < 0) {
    perror("inotify_init");
  }

  wd = inotify_add_watch(fd, ".", IN_MODIFY | IN_CREATE | IN_DELETE);
  printf("after add watch\n");



  struct io_uring ring;
  io_uring_queue_init(QUEUE_DEPTH, &ring, 0);

  for(int i = 0;i < 3; i++) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    io_uring_prep_read(sqe, fd, buffer, BUF_LEN, 0);
    io_uring_submit(&ring);


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

    length = cqe->res;
    printf("length %d\n", length);
    if (length < 0) {
      perror("read");
    }
    print_inotify_event(buffer, length);
    io_uring_cqe_seen(&ring, cqe);
  }

  io_uring_queue_exit(&ring);

  (void) inotify_rm_watch(fd, wd);
  (void) close(fd);

  return 0;
}