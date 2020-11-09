#include <stdio.h>
#include <unistd.h>
#include <sys/poll.h>
#include <iostream>
#include <string_view>

#define TIMEOUT 5

int main (void)
{
  struct pollfd fds[1];
  int ret;

  /* watch stdin for input */
  fds[0].fd = STDIN_FILENO;
  fds[0].events = POLLIN;

  ret = poll(fds, 1, TIMEOUT * 1000);

  if (ret == -1) {
    perror ("poll");
    return 1;
  }

  if (!ret) {
    printf ("%d seconds elapsed.\n", TIMEOUT);
    return 0;
  }

  if (fds[0].revents & POLLIN) {
    printf("stdin is readable\n");
    char buffer[10];
    int n = read(STDIN_FILENO, buffer, sizeof buffer);
    if( n > 0) {
      std::string_view str(buffer);
      std::cout << n << std::endl;
      std::cout << str.substr(0, n) << std::endl;
      std::cout << "===" << std::endl;
    }
  }

  return 0;

}