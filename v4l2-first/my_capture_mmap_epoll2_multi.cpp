#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <sys/epoll.h>

#include <linux/videodev2.h>
#include <array>

#define CLEAR(x) memset(&(x), 0, sizeof(x))

#ifndef V4L2_PIX_FMT_H264
#define V4L2_PIX_FMT_H264     v4l2_fourcc('H', '2', '6', '4') /* H264 with start codes */
#endif


struct buffer {
  void   *start;
  size_t  length;
};

struct device_info {
  char            *dev_name;
  char            *out_name;
  int              fd = -1;
  unsigned int     n_buffers;
  buffer          *buffers;
};

static int              out_buf;
static int              force_format;
static int              frame_count = 200;
static int              frame_number = 0;

static void errno_exit(const char *s)
{
  fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
  exit(EXIT_FAILURE);
}

static int xioctl(int fh, int request, void *arg)
{
  int r;

  do {
    r = ioctl(fh, request, arg);
  } while (-1 == r && EINTR == errno);

  return r;
}

static void process_image(device_info* pDeviceInfo, const void *p, int size)
{
  frame_number++;
  char filename[30];

  sprintf(filename, "%s-%d.raw", pDeviceInfo->out_name, frame_number);
  printf("%s\n", filename);

  FILE *fp=fopen(filename,"wb");

  if (out_buf)
    fwrite(p, size, 1, fp);

  fflush(fp);
  fclose(fp);
}

static int read_frame(device_info* pDeviceInfo)
{
  struct v4l2_buffer buf;
  unsigned int i;

  CLEAR(buf);

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;

  if (-1 == xioctl(pDeviceInfo->fd, VIDIOC_DQBUF, &buf)) {
    switch (errno) {
    case EAGAIN:
      return 0;

    case EIO:
      /* Could ignore EIO, see spec. */

      /* fall through */

    default:
      errno_exit("VIDIOC_DQBUF");
    }
  }

  assert(buf.index < pDeviceInfo->n_buffers);

  process_image(pDeviceInfo, pDeviceInfo->buffers[buf.index].start, buf.bytesused);

  if (-1 == xioctl(pDeviceInfo->fd, VIDIOC_QBUF, &buf))
    errno_exit("VIDIOC_QBUF");

  return 1;
}

static void mainloop_epoll(device_info* pDevices, int epfd)
{
  unsigned int count;

  count = frame_count;
  struct epoll_event events[2];

  while (count-- > 0) {
    for (;;) {

      int nfds = epoll_wait(epfd, events, 2, 10000);
      if(nfds <= 0)
        continue;
      printf("nfds=%d fd=%d ", nfds, events[0].data.fd);

      for (int i = 0; i < nfds; i++) {
        if( events[i].data.fd == pDevices[0].fd)
          read_frame(&pDevices[0]);
        else if( events[i].data.fd == pDevices[1].fd)
          read_frame(&pDevices[1]);
      }
      break;

      //if (read_frame(pDevices))
      //  break;
      /* EAGAIN - continue select loop. */
    }
  }
}


static void stop_capturing(device_info* pDevices)
{
  enum v4l2_buf_type type;

  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == xioctl(pDevices->fd, VIDIOC_STREAMOFF, &type))
    errno_exit("VIDIOC_STREAMOFF");
}

static void start_capturing(device_info* pDevices)
{
  unsigned int i;
  enum v4l2_buf_type type;

  for (i = 0; i < pDevices->n_buffers; ++i) {
    struct v4l2_buffer buf;

    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    if (-1 == xioctl(pDevices->fd, VIDIOC_QBUF, &buf))
      errno_exit("VIDIOC_QBUF");
  }
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == xioctl(pDevices->fd, VIDIOC_STREAMON, &type))
    errno_exit("VIDIOC_STREAMON");
}

static void uninit_device(device_info* pDevices)
{
  unsigned int i;

  for (i = 0; i < pDevices->n_buffers; ++i)
    if (-1 == munmap(pDevices->buffers[i].start, pDevices->buffers[i].length))
      errno_exit("munmap");

  free(pDevices->buffers);
}

static void init_mmap(device_info* pDevices)
{
  struct v4l2_requestbuffers req;

  CLEAR(req);

  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (-1 == xioctl(pDevices->fd, VIDIOC_REQBUFS, &req)) {
    if (EINVAL == errno) {
      fprintf(stderr, "%s does not support "
                      "memory mapping\n", pDevices->dev_name);
      exit(EXIT_FAILURE);
    } else {
      errno_exit("VIDIOC_REQBUFS");
    }
  }

  if (req.count < 2) {
    fprintf(stderr, "Insufficient buffer memory on %s\n",
            pDevices->dev_name);
    exit(EXIT_FAILURE);
  }

  pDevices->buffers = static_cast<buffer *>(calloc(req.count, sizeof(*pDevices->buffers)));

  if (!pDevices->buffers) {
    fprintf(stderr, "Out of memory\n");
    exit(EXIT_FAILURE);
  }

  for (pDevices->n_buffers = 0; pDevices->n_buffers < req.count; ++pDevices->n_buffers) {
    struct v4l2_buffer buf;

    CLEAR(buf);

    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = pDevices->n_buffers;

    if (-1 == xioctl(pDevices->fd, VIDIOC_QUERYBUF, &buf))
      errno_exit("VIDIOC_QUERYBUF");

    pDevices->buffers[pDevices->n_buffers].length = buf.length;
    pDevices->buffers[pDevices->n_buffers].start =
        mmap(NULL /* start anywhere */,
             buf.length,
             PROT_READ | PROT_WRITE /* required */,
             MAP_SHARED /* recommended */,
             pDevices->fd, buf.m.offset);

    if (MAP_FAILED == pDevices->buffers[pDevices->n_buffers].start)
      errno_exit("mmap");
  }
}


static void init_device(device_info* pDevices)
{
  struct v4l2_capability cap;
  struct v4l2_cropcap cropcap;
  struct v4l2_crop crop;
  struct v4l2_format fmt;
  unsigned int min;

  if (-1 == xioctl(pDevices->fd, VIDIOC_QUERYCAP, &cap)) {
    if (EINVAL == errno) {
      fprintf(stderr, "%s is no V4L2 device\n",
              pDevices->dev_name);
      exit(EXIT_FAILURE);
    } else {
      errno_exit("VIDIOC_QUERYCAP");
    }
  }

  if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    fprintf(stderr, "%s is no video capture device\n",
            pDevices->dev_name);
    exit(EXIT_FAILURE);
  }

  if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
    fprintf(stderr, "%s does not support streaming i/o\n",
            pDevices->dev_name);
    exit(EXIT_FAILURE);
  }


  /* Select video input, video standard and tune here. */


  CLEAR(cropcap);

  cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if (0 == xioctl(pDevices->fd, VIDIOC_CROPCAP, &cropcap)) {
    crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    crop.c = cropcap.defrect; /* reset to default */

    if (-1 == xioctl(pDevices->fd, VIDIOC_S_CROP, &crop)) {
      switch (errno) {
      case EINVAL:
        /* Cropping not supported. */
        break;
      default:
        /* Errors ignored. */
        break;
      }
    }
  } else {
    /* Errors ignored. */
  }


  CLEAR(fmt);

  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (force_format) {
    fprintf(stderr, "Set H264\r\n");
    fmt.fmt.pix.width       = 640; //replace
    fmt.fmt.pix.height      = 480; //replace
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_H264; //replace
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;

    if (-1 == xioctl(pDevices->fd, VIDIOC_S_FMT, &fmt))
      errno_exit("VIDIOC_S_FMT");

    /* Note VIDIOC_S_FMT may change width and height. */
  } else {
    /* Preserve original settings as set by v4l2-ctl for example */
    if (-1 == xioctl(pDevices->fd, VIDIOC_G_FMT, &fmt))
      errno_exit("VIDIOC_G_FMT");
  }

  /* Buggy driver paranoia. */
  min = fmt.fmt.pix.width * 2;
  if (fmt.fmt.pix.bytesperline < min)
    fmt.fmt.pix.bytesperline = min;
  min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
  if (fmt.fmt.pix.sizeimage < min)
    fmt.fmt.pix.sizeimage = min;

  init_mmap(pDevices);
}

static void close_device(device_info* pDevices)
{
  if (-1 == close(pDevices->fd))
    errno_exit("close");

  pDevices->fd = -1;
}

static void open_device(device_info* pDevices)
{
  struct stat st;

  if (-1 == stat(pDevices->dev_name, &st)) {
    fprintf(stderr, "Cannot identify '%s': %d, %s\n",
            pDevices->dev_name, errno, strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (!S_ISCHR(st.st_mode)) {
    fprintf(stderr, "%s is no device\n", pDevices->dev_name);
    exit(EXIT_FAILURE);
  }

  pDevices->fd = open(pDevices->dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

  if (-1 == pDevices->fd) {
    fprintf(stderr, "Cannot open '%s': %d, %s\n",
            pDevices->dev_name, errno, strerror(errno));
    exit(EXIT_FAILURE);
  }
}

static void usage(FILE *fp, int argc, char **argv)
{
  fprintf(fp,
          "Usage: %s [options]\n\n"
          "Version 1.3\n"
          "Options:\n"
          "-d | --device name   Video device name\n"
          "-h | --help          Print this message\n"
          "-m | --mmap          Use memory mapped buffers [default]\n"
          "-r | --read          Use read() calls\n"
          "-u | --userp         Use application allocated buffers\n"
          "-o | --output        Outputs stream to stdout\n"
          "-f | --format        Force format to 640x480 YUYV\n"
          "-c | --count         Number of frames to grab [%i]\n"
          "",
          argv[0], frame_count);
}

static const char short_options[] = "d:hmruofc:";

static const struct option
    long_options[] = {
    { "device", required_argument, NULL, 'd' },
    { "help",   no_argument,       NULL, 'h' },
    { "output", no_argument,       NULL, 'o' },
    { "format", no_argument,       NULL, 'f' },
    { "count",  required_argument, NULL, 'c' },
    { 0, 0, 0, 0 }
};

int main(int argc, char **argv)
{

  for (;;) {
    int idx;
    int c;

    c = getopt_long(argc, argv,
                    short_options, long_options, &idx);

    if (-1 == c)
      break;

    switch (c) {
    case 0: /* getopt_long() flag */
      break;

    case 'h':
      usage(stdout, argc, argv);
      exit(EXIT_SUCCESS);

    case 'o':
      out_buf++;
      break;

    case 'f':
      force_format++;
      break;

    case 'c':
      errno = 0;
      frame_count = strtol(optarg, NULL, 0);
      if (errno)
        errno_exit(optarg);
      break;

    default:
      usage(stderr, argc, argv);
      exit(EXIT_FAILURE);
    }
  }

  std::array<device_info,2> deviceInfos;
  deviceInfos[0].dev_name = "/dev/video0";
  deviceInfos[0].out_name = "video0";
  deviceInfos[1].dev_name = "/dev/video2";
  deviceInfos[1].out_name = "video2";

  for(auto& deviceInfo : deviceInfos) {
    printf("init %s\n", deviceInfo.dev_name);
    open_device(&deviceInfo);
    init_device(&deviceInfo);
    start_capturing(&deviceInfo);
  }

  int epfd = epoll_create(1);

  struct epoll_event ev;
  ev.data.fd = deviceInfos[0].fd;
  ev.events = EPOLLIN;
  epoll_ctl(epfd, EPOLL_CTL_ADD, ev.data.fd, &ev);

  struct epoll_event ev1;
  ev1.data.fd = deviceInfos[1].fd;
  ev1.events = EPOLLIN;
  epoll_ctl(epfd, EPOLL_CTL_ADD, ev1.data.fd, &ev1);

  mainloop_epoll(deviceInfos.data(), epfd);

  epoll_ctl(epfd, EPOLL_CTL_DEL, ev.data.fd, &ev);
  epoll_ctl(epfd, EPOLL_CTL_DEL, ev1.data.fd, &ev);

  for(auto& deviceInfo : deviceInfos) {
    stop_capturing(&deviceInfo);
    uninit_device(&deviceInfo);
    close_device(&deviceInfo);
  }
  fprintf(stderr, "\n");
  return 0;
}
