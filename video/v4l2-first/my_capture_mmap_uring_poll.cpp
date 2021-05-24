#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

#include <poll.h>
#include <sys/ioctl.h>


#include <linux/videodev2.h>

#include "capture_utils.hpp"

#include <liburing.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgproc/types_c.h>


#define CLEAR(x) memset(&(x), 0, sizeof(x))

#ifndef V4L2_PIX_FMT_H264
#define V4L2_PIX_FMT_H264     v4l2_fourcc('H', '2', '6', '4') /* H264 with start codes */
#endif


static int MAX_FRAME_COUNT = 10;
static int WIDTH = 640;
static int HEIGHT = 480;

void process_image(DeviceContext &device_context, void *p, int size, int frame) {
    char filename[30];
    sprintf(filename, "frame-%d.jpeg", frame);
    // yvyu -> rgb -> jpeg
    cv::Mat A(HEIGHT, WIDTH, CV_8UC2, p);
    cv::Mat B;
    cvtColor(A, B, CV_YUV2RGB_YVYU);
    cv::imwrite(filename, B);
}

int read_frame(DeviceContext &device_context, int frame) {
    struct v4l2_buffer buf;
    unsigned int i;

    CLEAR(buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(device_context.fd, VIDIOC_DQBUF, &buf)) {
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

    assert(buf.index < device_context.n_buffers);

    process_image(device_context, device_context.buffers[buf.index].start, buf.bytesused, frame);

    if (-1 == xioctl(device_context.fd, VIDIOC_QBUF, &buf))
        errno_exit("VIDIOC_QBUF");

    return 1;
}

static void mainloop_poll(struct io_uring *pRing, DeviceContext &device_context) {
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;

    for (int i = 0; i < MAX_FRAME_COUNT; i++) {
        for (;;) {

            sqe = io_uring_get_sqe(pRing);
            io_uring_prep_poll_add(sqe, device_context.fd, POLLIN);
            io_uring_submit(pRing);

            int ret = io_uring_wait_cqe(pRing, &cqe);
            if (ret < 0) {
                fprintf(stderr, "Error waiting for completion: %s\n", strerror(-ret));
                return;
            }
            /* Now that we have the CQE, let's process it */
            if (cqe->res < 0) {
                fprintf(stderr, "Error in async operation: %s\n", strerror(-cqe->res));
            }
            io_uring_cqe_seen(pRing, cqe);

            if (read_frame(device_context, i))
                break;
        }
    }
}

int main(int argc, char **argv) {
    DeviceContext deviceInfo;
    deviceInfo.dev_name = "/dev/video0";

    open_device(deviceInfo);
    init_device(deviceInfo);
    start_capturing(deviceInfo);

    struct io_uring ring;
    int ret = io_uring_queue_init(8, &ring, 0);
    if (ret) {
        fprintf(stderr, "Unable to setup io_uring: %s\n", strerror(-ret));
        return 1;
    }

    mainloop_poll(&ring, deviceInfo);


    stop_capturing(deviceInfo);
    uninit_device(deviceInfo);
    close_device(deviceInfo);
    fprintf(stderr, "\n");
    return 0;
}
