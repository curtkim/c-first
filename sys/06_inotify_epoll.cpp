#include <stdio.h>
#include <string.h>
#include <sys/inotify.h>
#include <sys/epoll.h>
#include <errno.h>
#include <unistd.h>

#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (1024 * (EVENT_SIZE + 16))


void print_inotify_event(char* buffer, int length) {
  printf("==================================\n");

  int i = 0;
  while (i < length) {
    struct inotify_event *event = (struct inotify_event *) &buffer[i];
    printf("wd=%d, event->len = %d, length= %d, i=%d \n", event->wd, event->len, length, i);

    if (event->len) {
      if (event->mask & IN_CREATE) {
        printf("The file %s was created.\n", event->name);
      } else if (event->mask & IN_DELETE) {
        printf("The file %s was deleted.\n", event->name);
      } else if (event->mask & IN_MODIFY) {
        printf("The file %s was modified.\n", event->name);
      }
      else {
        printf("mask = %d", event->mask);
      }
    }

    i += EVENT_SIZE + event->len;
  }
  printf("----------------------------------\n");
}

int main(int argc, char **argv) {
  int fd = inotify_init();
  printf("after inotify_init df=%d\n", fd);
  if (fd < 0) {
    perror("inotify_init");
  }

  int wd = inotify_add_watch(fd, ".", IN_MODIFY | IN_CREATE | IN_DELETE);
  printf("after add watch wd = %d\n", wd);

  int epfd = epoll_create(1);
  printf("epfd = %d\n", epfd);
  if (epfd < 0) {
    printf("epoll_create error\n");
    return -1;
  }

  struct epoll_event event;
  event.events = EPOLLIN | EPOLLET;
  event.data.fd = fd;

  int re = epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &event);
  if (re != 0) {
    printf("epoll_ctl error:%s\n", strerror(errno));
    return -1;
  }

  // fd : inotify
  // wd : watch
  // epfd : epoll

  char buffer[BUF_LEN];
  while (1) {
    int num = epoll_wait(epfd, &event, 1, -1);
    printf("epoll_wait num = %d\n", num);
    printf("event.fd = %d\n", event.data.fd);

    int length = read(fd, buffer, BUF_LEN);
    printf("read fd length %d\n", length);

    if (length < 0) {
      perror("read");
    }

    print_inotify_event(buffer, length);
  }

  (void) inotify_rm_watch(fd, wd);
  (void) close(fd);

  return 0;
}


