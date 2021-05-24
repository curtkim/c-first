#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/epoll.h>

#include <linux/videodev2.h>

#include "capture_utils.hpp"

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

void mainloop_epoll(DeviceContext &device_context, int epfd) {
    struct epoll_event events[1];

    for(int i = 0; i < MAX_FRAME_COUNT; i++){
        for (;;) {

            int nfds = epoll_wait(epfd, events, 1, 10000);
            //std::cout << nfds << std::endl;
            if (nfds <= 0)
                continue;

            if (read_frame(device_context, i))
                break;
            /* EAGAIN - continue select loop. */
        }
    }
}


int main(int argc, char **argv) {
    DeviceContext deviceInfo;
    deviceInfo.dev_name = "/dev/video0";

    open_device(deviceInfo);
    init_device(deviceInfo);
    start_capturing(deviceInfo);

    int epfd = epoll_create(1);
    struct epoll_event ev;
    ev.data.fd = deviceInfo.fd;
    ev.events = EPOLLIN;
    epoll_ctl(epfd, EPOLL_CTL_ADD, ev.data.fd, &ev);

    mainloop_epoll(deviceInfo, epfd);

    epoll_ctl(epfd, EPOLL_CTL_DEL, ev.data.fd, &ev);

    stop_capturing(deviceInfo);
    uninit_device(deviceInfo);
    close_device(deviceInfo);
    fprintf(stderr, "\n");
    return 0;
}
