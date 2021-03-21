#include <stdio.h>
#include <sys/uio.h>


int main()
{
  struct iovec iov[2];
  char buf1[10] = {0};
  char buf2[10] = {0};
  int strLen = 0;

  iov[0].iov_base = buf1;
  iov[0].iov_len = sizeof(buf1) - 1;

  iov[1].iov_base = buf2;
  iov[1].iov_len = sizeof(buf2) - 1;

  strLen = readv(0,iov,2);
  printf("total : %d\n",strLen);
  printf("buf1 : %s\n",buf1);
  printf("buf2 : %s\n",buf2);

  return 0;
}
