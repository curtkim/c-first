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

#include "./stopwatch.hpp"


#define CLEAR(x) memset(&(x), 0, sizeof(x))

#ifndef V4L2_PIX_FMT_H264
#define V4L2_PIX_FMT_H264     v4l2_fourcc('H', '2', '6', '4') /* H264 with start codes */
#endif


static int MAX_FRAME_COUNT = 100;
static int WIDTH = 640;
static int HEIGHT = 480;
static int PIXEL_FORMAT = V4L2_PIX_FMT_YUYV;


void process_image(DeviceContext &device_context, void *p, int size, int frame, int idx) {
    char filename[30];
    sprintf(filename, "frame-%d-%d.jpeg", idx, frame);
    // yvyu -> rgb -> jpeg
    cv::Mat A(HEIGHT, WIDTH, CV_8UC2, p);
    cv::Mat B;
    cvtColor(A, B, CV_YUV2RGB_YVYU);
    cv::imwrite(filename, B);
}

int read_frame(DeviceContext &device_context, int idx, int frame) {
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

    process_image(device_context, device_context.buffers[buf.index].start, buf.bytesused, frame, idx);

    if (-1 == xioctl(device_context.fd, VIDIOC_QBUF, &buf))
        errno_exit("VIDIOC_QBUF");

    return 1;
}

void mainloop_epoll(std::array<DeviceContext,2>& deviceInfos, int epfd) {
    struct epoll_event events[2];

    StopWatch watch;
    for(int f = 0; f < MAX_FRAME_COUNT; f++){
        for (;;) {

            watch.reset();
            int nfds = epoll_wait(epfd, events, 2, 10*1000);
            printf("frame=%d nfds=%d elapsed_time = %ld \n", f, nfds, watch.get_elapsed_time());
            if (nfds <= 0)
                continue;

            for (int i = 0; i < nfds; i++) {
                if( events[i].data.fd == deviceInfos[0].fd)
                    read_frame(deviceInfos[0], 0, f);
                else if( events[i].data.fd == deviceInfos[1].fd)
                    read_frame(deviceInfos[1], 1, f);
            }
            break;
            /* EAGAIN - continue select loop. */
        }
    }
}


int main(int argc, char **argv) {
    std::array<DeviceContext,2> deviceInfos;

    deviceInfos[0].dev_name = "/dev/video0";
    deviceInfos[1].dev_name = "/dev/video2";

    for(auto& deviceInfo : deviceInfos) {
        open_device(deviceInfo);
        init_device(deviceInfo, WIDTH, HEIGHT, PIXEL_FORMAT);
        start_capturing(deviceInfo);
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

    mainloop_epoll(deviceInfos, epfd);

    epoll_ctl(epfd, EPOLL_CTL_DEL, ev.data.fd, &ev);
    epoll_ctl(epfd, EPOLL_CTL_DEL, ev1.data.fd, &ev);

    for(auto& deviceInfo : deviceInfos) {
        stop_capturing(deviceInfo);
        uninit_device(deviceInfo);
        close_device(deviceInfo);
    }
    fprintf(stderr, "\n");
    return 0;
}
