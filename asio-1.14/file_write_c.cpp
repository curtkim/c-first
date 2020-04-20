#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

int main(int argc, char **argv)
{
  int fd;
  char buf[20];
  size_t nbytes;
  ssize_t bytes_written;

  if (argc < 2) {
    printf("Usage : %s pathname\n", argv[0]);
    exit(1);
  }

  strcpy(buf, "This is a test\n");
  nbytes = strlen(buf);

  /* opening the file in write-only mode */
  if ((fd = open(argv[1], O_WRONLY | O_CREAT)) < 0) {
    perror("Problem in opening the file");
    exit(1);
  }

  /* writing to the file */
  if ((bytes_written = write(fd, buf, nbytes)) < 0) {
    perror("Problem in writing to file");
    exit(1);
  }
  printf("Successfully written to %s\n", argv[1]);

  close(fd);
}
