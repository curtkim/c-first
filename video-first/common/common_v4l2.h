#ifndef COMMON_V4L2_H
#define COMMON_V4L2_H

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/types.h>

#include <libv4l2.h>
#include <linux/videodev2.h>

#define COMMON_V4L2_CLEAR(x) memset(&(x), 0, sizeof(x))
#define COMMON_V4L2_DEVICE "/dev/video0"

typedef struct {
  void *start;
  size_t length;
} CommonV4l2_Buffer;

typedef struct {
  int fd;
  CommonV4l2_Buffer *buffers;
  struct v4l2_buffer buf;
  unsigned int n_buffers;
} CommonV4l2;

void CommonV4l2_xioctl(int fh, unsigned long int request, void *arg) {
  int r;
  do {
    r = v4l2_ioctl(fh, request, arg);
  } while (r == -1 && ((errno == EINTR) || (errno == EAGAIN)));
  if (r == -1) {
    fprintf(stderr, "error %d, %s\n", errno, strerror(errno));
    exit(EXIT_FAILURE);
  }
}

// VIDIOC_S_FMT : set the data format
// VIDIOC_REQBUFS : Initiate Memory Mapping, User Pointer I/O or DMA buffer I/O
//    VIDIOC_QUERYBUF : Query the status of a buffer
//    VIDIOC_QBUF : Exchange a buffer with the driver
// VIDIOC_STREAMON
void CommonV4l2_init(CommonV4l2 *that, char *dev_name, unsigned int x_res, unsigned int y_res) {
  enum v4l2_buf_type type;
  struct v4l2_format fmt;
  struct v4l2_requestbuffers req;
  unsigned int i;

  that->fd = v4l2_open(dev_name, O_RDWR | O_NONBLOCK, 0);
  if (that->fd < 0) {
    perror("Cannot open device");
    exit(EXIT_FAILURE);
  }

  COMMON_V4L2_CLEAR(fmt);
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = x_res;
  fmt.fmt.pix.height = y_res;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
  fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
  CommonV4l2_xioctl(that->fd, VIDIOC_S_FMT, &fmt);
  if ((fmt.fmt.pix.width != x_res) || (fmt.fmt.pix.height != y_res))
    printf("Warning: driver is sending image at %dx%d\n", fmt.fmt.pix.width, fmt.fmt.pix.height);

  COMMON_V4L2_CLEAR(req);
  req.count = 2;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  CommonV4l2_xioctl(that->fd, VIDIOC_REQBUFS, &req);

  that->buffers = (CommonV4l2_Buffer *) calloc(req.count, sizeof(*that->buffers));
  for (that->n_buffers = 0; that->n_buffers < req.count; ++that->n_buffers) {
    COMMON_V4L2_CLEAR(that->buf);
    that->buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    that->buf.memory = V4L2_MEMORY_MMAP;
    that->buf.index = that->n_buffers;
    CommonV4l2_xioctl(that->fd, VIDIOC_QUERYBUF, &that->buf);

    that->buffers[that->n_buffers].length = that->buf.length;
    that->buffers[that->n_buffers].start =
      v4l2_mmap(NULL, that->buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, that->fd, that->buf.m.offset);
    if (MAP_FAILED == that->buffers[that->n_buffers].start) {
      perror("mmap");
      exit(EXIT_FAILURE);
    }
  }

  for (i = 0; i < that->n_buffers; ++i) {
    COMMON_V4L2_CLEAR(that->buf);
    that->buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    that->buf.memory = V4L2_MEMORY_MMAP;
    that->buf.index = i;
    CommonV4l2_xioctl(that->fd, VIDIOC_QBUF, &that->buf);
  }

  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  CommonV4l2_xioctl(that->fd, VIDIOC_STREAMON, &type);
}

void CommonV4l2_updateImage(CommonV4l2 *that) {
  fd_set fds;
  int r;
  struct timeval tv;

  do {
    FD_ZERO(&fds);
    FD_SET(that->fd, &fds);

    /* Timeout. */
    tv.tv_sec = 2;
    tv.tv_usec = 0;

    r = select(that->fd + 1, &fds, NULL, NULL, &tv);
  } while ((r == -1 && (errno == EINTR)));  // System calls that are interrupted by signals can either abort and return EINTR
  if (r == -1) {
    perror("select");
    exit(EXIT_FAILURE);
  }

  COMMON_V4L2_CLEAR(that->buf);
  that->buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  that->buf.memory = V4L2_MEMORY_MMAP;
  CommonV4l2_xioctl(that->fd, VIDIOC_DQBUF, &that->buf);
  CommonV4l2_xioctl(that->fd, VIDIOC_QBUF, &that->buf);
}

/* TODO must be called after updateImage? Or is init enough? */
void* CommonV4l2_getImage(CommonV4l2 *that) {
  return that->buffers[that->buf.index].start;
}

/* TODO must be called after updateImage? Or is init enough? */
size_t CommonV4l2_getImageSize(CommonV4l2 *that) {
  return that->buffers[that->buf.index].length;
}

void CommonV4l2_deinit(CommonV4l2 *that) {
  unsigned int i;
  enum v4l2_buf_type type;

  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  CommonV4l2_xioctl(that->fd, VIDIOC_STREAMOFF, &type);
  for (i = 0; i < that->n_buffers; ++i)
    v4l2_munmap(that->buffers[i].start, that->buffers[i].length);
  v4l2_close(that->fd);
  free(that->buffers);
}

#endif