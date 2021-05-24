#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>

#include "capture_utils.hpp"


#define CLEAR(x) memset(&(x), 0, sizeof(x))

void errno_exit(const char *s)
{
    fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
    exit(EXIT_FAILURE);
}


int xioctl(int fh, int request, void *arg)
{
    int r;

    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);

    return r;
}

void open_device(DeviceContext& device_context)
{
    struct stat st;

    if (-1 == stat(device_context.dev_name, &st)) {
        fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                device_context.dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (!S_ISCHR(st.st_mode)) {
        fprintf(stderr, "%s is no device\n", device_context.dev_name);
        exit(EXIT_FAILURE);
    }

    device_context.fd = open(device_context.dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == device_context.fd) {
        fprintf(stderr, "Cannot open '%s': %d, %s\n",
                device_context.dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

void init_mmap(DeviceContext& device_context)
{
    struct v4l2_requestbuffers req;

    CLEAR(req);

    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(device_context.fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            fprintf(stderr, "%s does not support "
                            "memory mapping\n", device_context.dev_name);
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_REQBUFS");
        }
    }

    if (req.count < 2) {
        fprintf(stderr, "Insufficient buffer memory on %s\n",
                device_context.dev_name);
        exit(EXIT_FAILURE);
    }

    device_context.buffers = static_cast<buffer *>(calloc(req.count, sizeof(*device_context.buffers)));

    if (!device_context.buffers) {
        fprintf(stderr, "Out of memory\n");
        exit(EXIT_FAILURE);
    }

    for (device_context.n_buffers = 0; device_context.n_buffers < req.count; ++device_context.n_buffers) {
        struct v4l2_buffer buf;

        CLEAR(buf);

        buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory      = V4L2_MEMORY_MMAP;
        buf.index       = device_context.n_buffers;

        if (-1 == xioctl(device_context.fd, VIDIOC_QUERYBUF, &buf))
            errno_exit("VIDIOC_QUERYBUF");

        device_context.buffers[device_context.n_buffers].length = buf.length;
        device_context.buffers[device_context.n_buffers].start =
                mmap(NULL /* start anywhere */,
                     buf.length,
                     PROT_READ | PROT_WRITE /* required */,
                     MAP_SHARED /* recommended */,
                     device_context.fd, buf.m.offset);

        if (MAP_FAILED == device_context.buffers[device_context.n_buffers].start)
            errno_exit("mmap");
    }
}


void init_device(DeviceContext& device_context)
{
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;

    if (-1 == xioctl(device_context.fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            fprintf(stderr, "%s is no V4L2 device\n",
                    device_context.dev_name);
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_QUERYCAP");
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is no video capture device\n",
                device_context.dev_name);
        exit(EXIT_FAILURE);
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s does not support streaming i/o\n",
                device_context.dev_name);
        exit(EXIT_FAILURE);
    }


    /* Select video input, video standard and tune here. */


    CLEAR(cropcap);

    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (0 == xioctl(device_context.fd, VIDIOC_CROPCAP, &cropcap)) {
        crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        crop.c = cropcap.defrect; /* reset to default */

        if (-1 == xioctl(device_context.fd, VIDIOC_S_CROP, &crop)) {
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
    /*
    if (force_format) {
        fprintf(stderr, "Set H264\r\n");
        fmt.fmt.pix.width       = 640; //replace
        fmt.fmt.pix.height      = 480; //replace
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_H264; //replace
        fmt.fmt.pix.field       = V4L2_FIELD_ANY;

        if (-1 == xioctl(device_context.fd, VIDIOC_S_FMT, &fmt))
            errno_exit("VIDIOC_S_FMT");
    } else {
        if (-1 == xioctl(device_context.fd, VIDIOC_G_FMT, &fmt))
            errno_exit("VIDIOC_G_FMT");
    }
    */
    if (-1 == xioctl(device_context.fd, VIDIOC_G_FMT, &fmt))
        errno_exit("VIDIOC_G_FMT");


    /* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;

    init_mmap(device_context);
}

void start_capturing(DeviceContext& device_context)
{
    unsigned int i;
    enum v4l2_buf_type type;

    for (i = 0; i < device_context.n_buffers; ++i) {
        struct v4l2_buffer buf;

        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (-1 == xioctl(device_context.fd, VIDIOC_QBUF, &buf))
            errno_exit("VIDIOC_QBUF");
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(device_context.fd, VIDIOC_STREAMON, &type))
        errno_exit("VIDIOC_STREAMON");
}

void stop_capturing(DeviceContext& device_context)
{
    enum v4l2_buf_type type;

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(device_context.fd, VIDIOC_STREAMOFF, &type))
        errno_exit("VIDIOC_STREAMOFF");
}


void uninit_device(DeviceContext& device_context)
{
    unsigned int i;

    for (i = 0; i < device_context.n_buffers; ++i)
        if (-1 == munmap(device_context.buffers[i].start, device_context.buffers[i].length))
            errno_exit("munmap");
    free(device_context.buffers);
}

void close_device(DeviceContext& device_context)
{
    if (-1 == close(device_context.fd))
        errno_exit("close");

    device_context.fd = -1;
}
