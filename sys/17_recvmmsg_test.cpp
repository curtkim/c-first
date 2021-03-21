#define _GNU_SOURCE

#include <netinet/ip.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#define VLEN 10
#define BUFSIZE 200
#define TIMEOUT 1


int main(void) {
  int sockfd, retval;
  struct sockaddr_in addr;
  struct mmsghdr msgs[VLEN];
  struct iovec iovecs[VLEN];
  char bufs[VLEN][BUFSIZE + 1];
  struct timespec timeout;

  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd == -1) {
    perror("socket()");
    exit(EXIT_FAILURE);
  }

  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(1234);
  if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) == -1) {
    perror("bind()");
    exit(EXIT_FAILURE);
  }

  memset(msgs, 0, sizeof(msgs));
  for (int i = 0; i < VLEN; i++) {
    iovecs[i].iov_base = bufs[i];
    iovecs[i].iov_len = BUFSIZE;
    msgs[i].msg_hdr.msg_iov = &iovecs[i];
    msgs[i].msg_hdr.msg_iovlen = 1;
  }

  timeout.tv_sec = TIMEOUT;
  timeout.tv_nsec = 0;

  retval = recvmmsg(sockfd, msgs, VLEN, 0, &timeout);
  if (retval == -1) {
    perror("recvmmsg()");
    exit(EXIT_FAILURE);
  }

  printf("%d messages received\n", retval);
  for (int i = 0; i < retval; i++) {
    bufs[i][msgs[i].msg_len] = 0;
    printf("%d %s", i + 1, bufs[i]);
  }
  exit(EXIT_SUCCESS);
}