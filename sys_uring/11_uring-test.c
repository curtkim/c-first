// BLOCK_SIZE만큼 4번 읽는다.
// io_uring_prep_readv를 4번 호출한다.??

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "liburing.h"

#define QD  4
#define MY_BLOCK_SIZE 4096

int main(int argc, char *argv[]) {
  struct io_uring ring;
  int i, fd, ret, pending, done;
  struct io_uring_sqe *sqe;
  struct io_uring_cqe *cqe;
  struct iovec *iovecs;
  off_t offset;
  void *buf;

  if (argc < 2) {
    printf("%s: file\n", argv[0]);
    return 1;
  }

  // 1. io_uring_queue_init
  ret = io_uring_queue_init(QD, &ring, 0);
  if (ret < 0) {
    fprintf(stderr, "queue_init: %s\n", strerror(-ret));
    return 1;
  }

  printf("filename= %s\n", argv[1]);

  fd = open(argv[1], O_RDONLY); //  | O_DIRECT
  if (fd < 0) {
    perror("open");
    return 1;
  }
  printf("fid= %d\n", fd);

  // calloc(size_t num, size_t size)
  iovecs = calloc(QD, sizeof(struct iovec));
  for (i = 0; i < QD; i++) {
    if (posix_memalign(&buf, MY_BLOCK_SIZE, MY_BLOCK_SIZE))
      return 1;
    iovecs[i].iov_base = buf;
    iovecs[i].iov_len = MY_BLOCK_SIZE;
  }


  offset = 0;
  i = 0;
  do {
    sqe = io_uring_get_sqe(&ring);
    if (!sqe)
      break;
    io_uring_prep_readv(sqe, fd, &iovecs[i], 1, offset);
    offset += iovecs[i].iov_len;
    i++;
  } while (1);

  ret = io_uring_submit(&ring);
  if (ret < 0) {
    fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
    return 1;
  }

  done = 0;
  pending = ret;
  for (i = 0; i < pending; i++) {
    ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
      fprintf(stderr, "io_uring_wait_cqe: %s\n", strerror(-ret));
      return 1;
    }

    done++;
    printf("cqe->res= %d\n", cqe->res);
    ret = 0;
    if (cqe->res != MY_BLOCK_SIZE) {
      fprintf(stderr, "ret=%d, wanted 4096\n", cqe->res);
      ret = 1;
    }
    io_uring_cqe_seen(&ring, cqe);
    if (ret)
      break;
  }

  printf("Submitted=%d, completed=%d\n", pending, done);
  close(fd);
  io_uring_queue_exit(&ring);
  return 0;
}

