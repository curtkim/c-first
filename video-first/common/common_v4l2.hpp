//
// Created by curt on 20. 12. 15..
//

#ifndef VIDEO_FIRST_COMMON_V4L2_CUH
#define VIDEO_FIRST_COMMON_V4L2_CUH

#include <liburing.h>

void waitByIOUring(struct io_uring& ring, int fd){
  struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
  io_uring_prep_poll_add(sqe, fd, POLLIN);
  io_uring_sqe_set_data(sqe, &fd);
  io_uring_submit(&ring);

  struct io_uring_cqe *cqe;
  io_uring_wait_cqe(&ring, &cqe);
  int * pFd = (int*)io_uring_cqe_get_data(cqe);
  printf("fd=%d, *pFd=%d\n", fd, *pFd);
  io_uring_cqe_seen(&ring, cqe);
}


#endif //VIDEO_FIRST_COMMON_V4L2_CUH
