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

#include <sys/poll.h>
#include <sys/epoll.h>


#define COMMON_V4L2_CLEAR(x) memset(&(x), 0, sizeof(x))
#define COMMON_V4L2_DEVICE "/dev/video0"

typedef struct {
  void *start;
  size_t length;
} MyV4l2Buffer;

typedef struct {
  int fd;
  MyV4l2Buffer *buffers;
  //struct v4l2_buffer buf;
  int index;
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
void CommonV4l2_init(CommonV4l2 *that, char *dev_name, unsigned int x_res, unsigned int y_res, __u32 pix_format) {
  enum v4l2_buf_type type;
  struct v4l2_format fmt;
  struct v4l2_requestbuffers req;
  unsigned int i;

  // 1. open_device -> fd
  that->fd = v4l2_open(dev_name, O_RDWR | O_NONBLOCK, 0);
  printf("v4l2_open fd=%d\n", that->fd);
  if (that->fd < 0) {
    perror("Cannot open device");
    exit(EXIT_FAILURE);
  }

  // 2. set_format
  COMMON_V4L2_CLEAR(fmt);
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = x_res;
  fmt.fmt.pix.height = y_res;
  fmt.fmt.pix.pixelformat = pix_format;
  fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
  CommonV4l2_xioctl(that->fd, VIDIOC_S_FMT, &fmt);
  if ((fmt.fmt.pix.width != x_res) || (fmt.fmt.pix.height != y_res))
    printf("Warning: driver is sending image at %dx%d\n", fmt.fmt.pix.width, fmt.fmt.pix.height);
  printf("fmt.fmt.pix.pixelformat=%d V4L2_PIX_FMT_RGB24=%d\n", fmt.fmt.pix.pixelformat, V4L2_PIX_FMT_RGB24);

  // 3. prepare buffer
  COMMON_V4L2_CLEAR(req);
  req.count = 2;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  CommonV4l2_xioctl(that->fd, VIDIOC_REQBUFS, &req);

  that->buffers = (MyV4l2Buffer *) calloc(req.count, sizeof(*that->buffers));
  for (int i = 0; i < req.count; i++) {
    struct v4l2_buffer buf;
    //COMMON_V4L2_CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    CommonV4l2_xioctl(that->fd, VIDIOC_QUERYBUF, &buf);

    that->buffers[i].length = buf.length;
    that->buffers[i].start = v4l2_mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, that->fd, buf.m.offset);
    if (MAP_FAILED == that->buffers[i].start) {
      perror("mmap");
      exit(EXIT_FAILURE);
    }
  }
  that->n_buffers = req.count;

  for (i = 0; i < that->n_buffers; ++i) {
    struct v4l2_buffer buf;
    //COMMON_V4L2_CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    CommonV4l2_xioctl(that->fd, VIDIOC_QBUF, &buf);
  }

  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  CommonV4l2_xioctl(that->fd, VIDIOC_STREAMON, &type);
}

void waitBySelect(int fd) {
  fd_set fds;
  int r;
  struct timeval tv;

  do {
    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    /* Timeout. */
    tv.tv_sec = 2;
    tv.tv_usec = 0;

    r = select(fd + 1, &fds, NULL, NULL, &tv);
  } while ((r == -1 && (errno == EINTR)));  // System calls that are interrupted by signals can either abort and return EINTR
  if (r == -1) {
    perror("select");
    exit(EXIT_FAILURE);
  }
}

void waitBySelect(int fd1, int fd2) {
  fd_set fds;
  int r;
  struct timeval tv;

  do {
    FD_ZERO(&fds);
    FD_SET(fd1, &fds);
    FD_SET(fd2, &fds);

    /* Timeout. */
    tv.tv_sec = 2;
    tv.tv_usec = 0;

    r = select(fd2 + 1, &fds, NULL, NULL, &tv);
  } while ((r == -1 && (errno == EINTR)));  // System calls that are interrupted by signals can either abort and return EINTR
  if (r == -1) {
    perror("select");
    exit(EXIT_FAILURE);
  }
}

void waitByPoll(int fd) {
  int r;
  do {
    struct pollfd fds[1];
    fds[0].fd = fd;
    fds[0].events = POLLIN;
    r = poll(fds, 1, 2 * 1000);

    if (0 == r) {
      fprintf(stderr, "poll timeout\n");
      exit(EXIT_FAILURE);
    }
  } while ((r == -1 && (errno == EINTR)));  // System calls that are interrupted by signals can either abort and return EINTR
}

void waitByEpoll(int epfd) {
  struct epoll_event events[1];
  int nfds;
  do {
    nfds = epoll_wait(epfd, events, 1, 10000);
    if( nfds == 0)
      printf("nfds=0");
  } while (nfds == 0);
  printf("epfd = %d, event.data.fd = %d\n", epfd, events[0].data.fd);
}

void CommonV4l2_updateImage(CommonV4l2 *that) {
  struct v4l2_buffer buf;
  //COMMON_V4L2_CLEAR(buf);
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  CommonV4l2_xioctl(that->fd, VIDIOC_DQBUF, &buf);

  that->index = buf.index;
  CommonV4l2_xioctl(that->fd, VIDIOC_QBUF, &buf);
  //printf("dqbuf buf.index = %d, buf.length = %d %fms %fms\n", buf.index, buf.length, (tic2-tic1)*1000, (tic3-tic2)*1000);
}

/* TODO must be called after updateImage? Or is init enough? */
void* CommonV4l2_getImage(CommonV4l2 *that) {
  //printf("getImage buf.index = %d\n", that->index);
  return that->buffers[that->index].start;
}

/* TODO must be called after updateImage? Or is init enough? */
size_t CommonV4l2_getImageSize(CommonV4l2 *that) {
  return that->buffers[that->index].length;
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


int print_caps(int fd) {
  struct v4l2_capability caps = {};
  if (-1 == v4l2_ioctl(fd, VIDIOC_QUERYCAP, &caps)) {
    perror("Querying Capabilities");
    return 1;
  }

  // 1. Driver Caps
  printf("Driver Caps:\n"
         "  Driver: \"%s\"\n"
         "  Card: \"%s\"\n"
         "  Bus: \"%s\"\n"
         "  Version: %d.%d\n"
         "  Capabilities: %08x\n",
         caps.driver, caps.card, caps.bus_info, (caps.version >> 16) && 0xff,
         (caps.version >> 24) && 0xff, caps.capabilities);

  struct v4l2_cropcap cropcap = {0};
  cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == v4l2_ioctl(fd, VIDIOC_CROPCAP, &cropcap)) {
    perror("Querying Cropping Capabilities");
    return 1;
  }

  // 2. Camera Cropping
  printf("Camera Cropping:\n"
         "  Bounds: %dx%d+%d+%d\n"
         "  Default: %dx%d+%d+%d\n"
         "  Aspect: %d/%d\n",
         cropcap.bounds.width, cropcap.bounds.height, cropcap.bounds.left,
         cropcap.bounds.top, cropcap.defrect.width, cropcap.defrect.height,
         cropcap.defrect.left, cropcap.defrect.top,
         cropcap.pixelaspect.numerator, cropcap.pixelaspect.denominator);

  int support_grbg10 = 0;

  struct v4l2_fmtdesc fmtdesc = {0};
  fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  char fourcc[5] = {0};
  char c, e;
  printf("  FMT : CE Desc\n--------------------\n");
  while (0 == v4l2_ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc)) {
    strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
    if (fmtdesc.pixelformat == V4L2_PIX_FMT_SGRBG10)
      support_grbg10 = 1;
    c = fmtdesc.flags & 1 ? 'C' : ' ';  // compressed?
    e = fmtdesc.flags & 2 ? 'E' : ' ';  // emulated?
    printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
    fmtdesc.index++;
  }

  /*
  if (!support_grbg10) {
    printf("Doesn't support GRBG10.\n");
    return 1;
  }
  */

  /*
  struct v4l2_format fmt = {0};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = 752;
  fmt.fmt.pix.height = 480;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;

  if (-1 == v4l2_ioctl(fd, VIDIOC_S_FMT, &fmt)) {
    perror("Setting Pixel Format");
    return 1;
  }

  strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
  printf("Selected Camera Mode:\n"
         "  Width: %d\n"
         "  Height: %d\n"
         "  PixFmt: %s\n"
         "  Field: %d\n",
         fmt.fmt.pix.width, fmt.fmt.pix.height, fourcc, fmt.fmt.pix.field);
  */
  return 0;
}

#endif