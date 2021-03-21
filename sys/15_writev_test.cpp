#include <sys/uio.h>
#include <stdio.h>
#include <string.h>

int main()
{
  struct iovec iov[2] = {nullptr};
  char buf1[] = "Hello!!! ";
  char buf2[] = "I am Hacker!!\n";

  iov[0].iov_base = buf1;
  iov[0].iov_len = strlen(buf1);

  iov[1].iov_base = buf2;
  iov[1].iov_len = strlen(buf2);

  writev(1,iov,2);

  return 0;
}
