#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <vector>
#include <numeric>

#include "io_service.hpp"

#define USE_SPLICE 0
#define USE_LINK 0
#define USE_POLL 0

enum {
  BUF_SIZE = 512,
  MAX_CONN_SIZE = 512,
};

int runningCoroutines = 0;

task<> accept_connection(io_service& service, int serverfd) {
  while (int clientfd = co_await service.accept(serverfd, nullptr, nullptr)) {
    [=, &service](int clientfd) -> task<> {
      printf("sockfd %d is accepted; number of running coroutines: %d\n",clientfd, ++runningCoroutines);
      std::vector<char> buf(BUF_SIZE);

      while (true) {
        printf("service.recv\n");
        int r = co_await service.recv(clientfd, buf.data(), BUF_SIZE, MSG_NOSIGNAL);
        printf("service.recv after r=%d\n", r);
        if (r <= 0) break;
        co_await service.send(clientfd, buf.data(), r, MSG_NOSIGNAL);
        printf("service.send after\n");
      }

      shutdown(clientfd, SHUT_RDWR);
      printf("sockfd %d is closed; number of running coroutines: %d\n", clientfd, --runningCoroutines);
    }(clientfd);
  }
}

int main(int argc, char *argv[]) {
  uint16_t server_port = 0;
  if (argc == 2) {
    server_port = (uint16_t)std::strtoul(argv[1], nullptr, 10);
  }
  if (server_port == 0) {
    printf("Usage: %d <PORT>\n", argv[0]);
    return 1;
  }

  io_service service(MAX_CONN_SIZE);

  int sockfd = socket(AF_INET, SOCK_STREAM, 0) | panic_on_err("socket creation", true);
  on_scope_exit closesock([=]() { shutdown(sockfd, SHUT_RDWR); });

  if (sockaddr_in addr = {
      .sin_family = AF_INET,
      .sin_port = htons(server_port),
      .sin_addr = { INADDR_ANY },
      .sin_zero = {},
    }; bind(sockfd, reinterpret_cast<sockaddr *>(&addr), sizeof (sockaddr_in))) panic("socket binding", errno);

  if (listen(sockfd, MAX_CONN_SIZE * 2)) panic("listen", errno);
  printf("Listening: %d\n", server_port);

  service.run(accept_connection(service, sockfd));
}
