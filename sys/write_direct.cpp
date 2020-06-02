// from http://blog.naver.com/PostView.nhn?blogId=hseok74&logNo=120202296673
#define _GNU_SOURCE

#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#define __DD_ERR(x, args...)     do{ printf("%d:%s, ", x,errno,strerror(errno),##args); }while(0)
#define TMP_BUF_SIZE            (1024*1024)
#define ALIGN                   512

int main(int argc, char *argv[]) {
  unsigned int trv = 0;
  int fd, rv;
  printf("%s %s\n", __DATE__, __TIME__);

  #if 1
    unsigned char *tmp, *_tmp = (unsigned char *) malloc(TMP_BUF_SIZE + 512);
    //  tmp = _tmp;
    tmp = (unsigned char *) (((unsigned long) _tmp + (ALIGN - 1)) & ~(ALIGN - 1));
    printf("%p - %p\n", _tmp, tmp);
  #else
    void *tmp, *_tmp;
    posix_memalign(&_tmp, TMP_BUF_SIZE, TMP_BUF_SIZE);
    tmp = _tmp;
  #endif


  fd = open(argv[1], O_RDONLY | O_DIRECT);
  if (fd < 0) {
    __DD_ERR("open(%s)\n", argv[1]);
    return (-1);
  }

  while ((rv = read(fd, tmp, TMP_BUF_SIZE)) > 0) {
    trv += rv;
    printf("Total Read : %d KB\n", trv >> 10);
  }

  if (rv <= 0) {
    __DD_ERR("read(%s) = %d\n", argv[1], rv);
  }

  close(fd);
  free(_tmp);
  return 0;
}