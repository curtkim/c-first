#include <string>

#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>              /* low-level i/o */

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#define CLEAR(x) memset(&(x), 0, sizeof(x))

enum io_method {
  IO_METHOD_READ,
  IO_METHOD_MMAP,
  IO_METHOD_USERPTR,
};

struct buffer {
  void   *start;
  size_t  length;
};

struct CaptureSetting {
  std::string device_name;
  enum io_method io = IO_METHOD_MMAP;
  unsigned int n_buffers;
};

struct CaptureRuntimeInfo {
  int fd = -1;
  buffer* buffers;
};


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

static void init_mmap(CaptureRuntimeInfo info, const char* dev_name)
{
  struct v4l2_requestbuffers req;

  CLEAR(req);

  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (-1 == xioctl(info.fd, VIDIOC_REQBUFS, &req)) {
    if (EINVAL == errno) {
      fprintf(stderr, "%s does not support "
                      "memory mapping\n", dev_name);
      exit(EXIT_FAILURE);
    } else {
      errno_exit("VIDIOC_REQBUFS");
    }
  }

  if (req.count < 2) {
    fprintf(stderr, "Insufficient buffer memory on %s\n",
            dev_name);
    exit(EXIT_FAILURE);
  }

  info.buffers = static_cast<buffer *>(calloc(req.count, sizeof(*info.buffers)));

  if (!info.buffers) {
    fprintf(stderr, "Out of memory\n");
    exit(EXIT_FAILURE);
  }

  for (int n_buffers = 0; n_buffers < req.count; ++n_buffers) {
    struct v4l2_buffer buf;

    CLEAR(buf);

    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = n_buffers;

    if (-1 == xioctl(info.fd, VIDIOC_QUERYBUF, &buf))
      errno_exit("VIDIOC_QUERYBUF");

    info.buffers[n_buffers].length = buf.length;
    info.buffers[n_buffers].start =
      mmap(NULL /* start anywhere */,
           buf.length,
           PROT_READ | PROT_WRITE /* required */,
           MAP_SHARED /* recommended */,
           info.fd, buf.m.offset);

    if (MAP_FAILED == info.buffers[n_buffers].start)
      errno_exit("mmap");
  }
}


int open_device(const std::string& dev_name)
{
  struct stat st;

  if (-1 == stat(dev_name.c_str(), &st)) {
    fprintf(stderr, "Cannot identify '%s': %d, %s\n",
            dev_name.c_str(), errno, strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (!S_ISCHR(st.st_mode)) {
    fprintf(stderr, "%s is no device\n", dev_name.c_str());
    exit(EXIT_FAILURE);
  }

  int fd = open(dev_name.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);

  if (-1 == fd) {
    fprintf(stderr, "Cannot open '%s': %d, %s\n",
            dev_name.c_str(), errno, strerror(errno));
    exit(EXIT_FAILURE);
  }
  return fd;
}

