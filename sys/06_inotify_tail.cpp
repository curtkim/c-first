// https://www.programmersought.com/article/28711046216/

#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/inotify.h>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#define BUFSIZE 81

int main(int argc, char* argv[]) {
  printf("LOG:argc = %d\n", argc);
  printf("LOG:argc[1] = %s\n", argv[1]);


  int fd = inotify_init();
  printf("LOG:fd = %d\n", fd);
  if (fd < 0) {
    printf("LOG:inotify_init error\n");
    return -1;
  }
  int wd = inotify_add_watch(fd, argv[1], IN_MODIFY);
  printf("LOG:wd = %d\n", wd);
  if (wd < 0) {
    printf("LOG:inotify_add_watch error");
    return -1;
  }

  int epfd = epoll_create(1);
  printf("LOG:epfd = %d\n", epfd);
  if (epfd < 0) {
    printf("LOG:epoll_create error\n");
    return -1;
  }

  struct epoll_event event;
  event.events = EPOLLIN | EPOLLET;
  event.data.fd = fd;

  int re = epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &event);
  printf("LOG:re = %d\n", re);
  if (re != 0) {
    printf("LOG:epoll_ctl error:%s\n", strerror(errno));
    return -1;
  }

  ssize_t n;
  ssize_t tmp;
  int file_fd = open(argv[1], O_RDONLY);
  printf("LOG:file_fd = %d\n", file_fd);
  if (file_fd == -1) {
    printf("LOG:open error\n");
    return -1;
  }
  char buf[BUFSIZE];

  int seek_re;

  while (1) {
    int num = epoll_wait(epfd, &event, 1, -1);
    printf("LOG:epoll_wait num = %d\n", num);
    if (num > 0) {
      // Here is to read the message without repeating the notification
      tmp = read(fd, buf, BUFSIZE);
      seek_re = lseek(file_fd, (off_t) -1, SEEK_CUR);
      n = read(file_fd, buf, BUFSIZE);
      if (n == -1) {
        printf("break\n");
        break;
      }
      if (n > 0) {
        printf("%s", buf);
      }
    }
    printf("LOG:new loop\n");
  }

  return 0;
}
 