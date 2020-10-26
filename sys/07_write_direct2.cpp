#define _GNU_SOURCE
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>


#define BLOCKSIZE 512

char image[] =
  {
    0, 1, 2, 3, 4,
  };

int main()
{
  printf("%d", sizeof(image));
  void *buffer;
  posix_memalign(&buffer, BLOCKSIZE, BLOCKSIZE);
  memcpy(buffer, image, sizeof(image));
  int f = open("temp.data", O_CREAT|O_TRUNC|O_WRONLY|O_DIRECT, S_IRWXU);
  write(f, buffer, BLOCKSIZE);
  close(f);
  free(buffer);
  return 0;
}