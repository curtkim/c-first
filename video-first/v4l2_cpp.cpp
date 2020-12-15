#include <string>
#include <thread>
#include <chrono>

#include <sys/epoll.h>

#include "common/common_v4l2.h"
#include "common/common_v4l2.hpp"

using namespace std::chrono;

int main(int argc, char *argv[]) {

  //int width = 1024;
  //int height = 768;
  //int width1 = 864, height1 = 480;
  int width1 = 800, height1 = 448;
  //int width1 = 1600, height1 = 896;


  // 5. webcam init
  CommonV4l2 cam1;
  CommonV4l2_init(&cam1, "/dev/video0", width1, height1, V4L2_PIX_FMT_YUYV);
  print_caps(cam1.fd);

  struct io_uring ring;
  io_uring_queue_init(16, &ring, 0);

  /*
  int epfd = epoll_create(1);
  struct epoll_event ev;
  ev.data.fd = common_v4l2.fd;
  ev.events = EPOLLIN;
  epoll_ctl(epfd, EPOLL_CTL_ADD, ev.data.fd, &ev);
  */

  for(int i = 0; i < 50; i++) {

    auto tic = high_resolution_clock::now();

    printf("===render\n");

    waitBySelect(cam1.fd);
    auto ticPoll = high_resolution_clock::now();
    //waitByPoll(common_v4l2.fd);
    //waitByEpoll(epfd);
    //waitByIOUring(ring, common_v4l2.fd);
    //sleep(1);

    // 6. get image
    CommonV4l2_updateImage(&cam1);
    auto ticImage10 = high_resolution_clock::now();
    void* image1 = CommonV4l2_getImage(&cam1);
    auto ticImage1 = high_resolution_clock::now();

    printf("poll=%d ns, image1_update=%d ns image1_get=%d ns\n",
           duration_cast<nanoseconds>(ticPoll - tic).count(),
           duration_cast<nanoseconds>(ticImage10 - ticPoll).count(),
           duration_cast<nanoseconds>(ticImage1 - ticImage10).count());
  }

  //epoll_ctl(epfd, EPOLL_CTL_DEL, ev.data.fd, &ev);
  CommonV4l2_deinit(&cam1);
}