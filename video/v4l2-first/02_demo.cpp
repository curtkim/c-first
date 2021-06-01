// from https://github.com/biotrump/v4l-capture/blob/master/demo.c

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/poll.h>

static int xioctl(int fd, int request, void *arg) {
    int r;

    do
        r = ioctl(fd, request, arg);
    while (-1 == r && EINTR == errno);

    return r;
}

void SetFPSParam(int fd, uint32_t fps) {
    struct v4l2_streamparm param;
    memset(&param, 0, sizeof(param));
    param.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    param.parm.capture.timeperframe.numerator = 1;
    param.parm.capture.timeperframe.denominator = fps;
    if (-1 == xioctl(fd, VIDIOC_S_PARM, &param)) {
        perror("unable to change device parameters");
        return;
    }

    if (param.parm.capture.timeperframe.numerator) {
        double fps_new = param.parm.capture.timeperframe.denominator
                         / param.parm.capture.timeperframe.numerator;
        if ((double) fps != fps_new) {
            printf("unsupported frame rate [%d,%f]\n", fps, fps_new);
            return;
        } else {
            printf("new fps:%u , %u/%u\n", fps, param.parm.capture.timeperframe.denominator,
                   param.parm.capture.timeperframe.numerator);
        }
    }
}

uint32_t GetFPSParam(int fd, double fps, struct v4l2_frmivalenum *pfrmival) {
    struct v4l2_frmivalenum frmival[10];
    float fpss[10];
    int i = 0;
/*
    memset(&frmival,0,sizeof(frmival));
    frmival.pixel_format = fmt;
    frmival.width = width;
    frmival.height = height;*/
    memset(fpss, 0, sizeof(fpss));

    while (pfrmival->index < 10) {
        if (-1 == xioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, pfrmival)) {
            perror("getting VIDIOC_ENUM_FRAMEINTERVALS");
            break;
        }
        frmival[pfrmival->index] = *pfrmival;
        if (pfrmival->type == V4L2_FRMIVAL_TYPE_DISCRETE) {
            double f;
            f = (double) pfrmival->discrete.denominator / pfrmival->discrete.numerator;
            printf("[%u/%u]\n", pfrmival->discrete.denominator, pfrmival->discrete.numerator);
            printf("[%dx%d] %f fps\n", pfrmival->width, pfrmival->height, f);

            fpss[pfrmival->index] = f;
            frmival[pfrmival->index] = *pfrmival;
        } else {
            double f1, f2;
            f1 = (double) pfrmival->stepwise.max.denominator / pfrmival->stepwise.max.numerator;
            f2 = (double) pfrmival->stepwise.min.denominator / pfrmival->stepwise.min.numerator;
            printf("[%dx%d] [%f,%f] fps\n", pfrmival->width, pfrmival->height, f1, f2);
        }
        printf("idx=%d\n", pfrmival->index);
        pfrmival->index++;
    }

    /* list is in increasing order */
    if (pfrmival->index) {
        i = pfrmival->index;
        while (--i >= 0) {
            if (fps <= fpss[i]) {
                break;
            }
        }
        *pfrmival = frmival[i];
        printf("found[%f,%f]\n", fps, fpss[i]);
    }
    return (uint32_t) fpss[i];
}


#define FORCED_WIDTH  640
#define FORCED_HEIGHT 480
#define FORCED_FORMAT V4L2_PIX_FMT_YUYV    //V4L2_PIX_FMT_MJPEG
#define FORCED_FIELD  V4L2_FIELD_ANY

/*http://forum.processing.org/one/topic/webcam-with-stable-framerate-25fps-on-high-resolution.html
The stable fps can only occur in the low frame rate. Higher fps>15 may not be stable.
*/
#define FORCED_FPS        (10)

int main() {

    struct v4l2_frmivalenum frmival;
    uint32_t fps;

    memset(&frmival, 0, sizeof(frmival));
    frmival.pixel_format = V4L2_PIX_FMT_YUYV;
    frmival.width = FORCED_WIDTH;
    frmival.height = FORCED_HEIGHT;


    int fd = open("/dev/video0", O_RDWR);

    fps = GetFPSParam(fd, (double) FORCED_FPS, &frmival);
    printf("fps=%d\n", fps);
    //SetFPSParam(fd, fps);
}