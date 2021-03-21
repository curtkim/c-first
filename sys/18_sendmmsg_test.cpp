#define _GNU_SOURCE

#include <netinet/ip.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>


int main(void) {
  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd == -1) {
    perror("socket()");
    exit(EXIT_FAILURE);
  }

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(1234);
  if (connect(sockfd, (struct sockaddr *) &addr, sizeof(addr)) == -1) {
    perror("connect()");
    exit(EXIT_FAILURE);
  }

  struct iovec msg1[2], msg2;

  memset(msg1, 0, sizeof(msg1));
  msg1[0].iov_base = (void *) "one";
  msg1[0].iov_len = 3;
  msg1[1].iov_base = (void *) "two";
  msg1[1].iov_len = 3;

  memset(&msg2, 0, sizeof(msg2));
  msg2.iov_base = (void *) "three";
  msg2.iov_len = 5;

  struct mmsghdr msg[2];
  memset(msg, 0, sizeof(msg));
  msg[0].msg_hdr.msg_iov = msg1;
  msg[0].msg_hdr.msg_iovlen = 2;

  msg[1].msg_hdr.msg_iov = &msg2;
  msg[1].msg_hdr.msg_iovlen = 1;

  int retval = sendmmsg(sockfd, msg, 2, 0);
  if (retval == -1)
    perror("sendmmsg()");
  else
    printf("%d messages sent\n", retval);

  exit(0);
}
