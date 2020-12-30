// https://unixism.net/2020/04/io-uring-by-example-part-3-a-web-server-with-io-uring/

#include <stdio.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <liburing.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "21_webserver.h"


#define DEFAULT_SERVER_PORT     8000
#define QUEUE_DEPTH             256
#define READ_SZ                 8192

#define EVENT_TYPE_ACCEPT       0
#define EVENT_TYPE_READ         1
#define EVENT_TYPE_WRITE        2

struct request {
  int event_type;
  int iovec_count;
  int client_socket;
  struct iovec iov[];
};

struct io_uring ring;

int add_accept_request(int server_socket, struct sockaddr_in *client_addr,
                       socklen_t *client_addr_len) {
  printf("add_accept_request port=%d\n", client_addr->sin_port);
  struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
  io_uring_prep_accept(sqe, server_socket, (struct sockaddr *) client_addr,
                       client_addr_len, 0);
  struct request *req = malloc(sizeof(*req));
  req->event_type = EVENT_TYPE_ACCEPT;
  io_uring_sqe_set_data(sqe, req);
  io_uring_submit(&ring);
  return 0;
}

int add_read_request(int client_socket) {
  struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);

  struct request *req = malloc(sizeof(*req) + sizeof(struct iovec));
  req->iov[0].iov_base = malloc(READ_SZ);
  req->iov[0].iov_len = READ_SZ;
  req->event_type = EVENT_TYPE_READ;
  req->client_socket = client_socket;
  memset(req->iov[0].iov_base, 0, READ_SZ);

  /* Linux kernel 5.5 has support for readv, but not for recv() or read() */
  io_uring_prep_readv(sqe, client_socket, &req->iov[0], 1, 0);
  io_uring_sqe_set_data(sqe, req);
  io_uring_submit(&ring);
  return 0;
}

int add_write_request(struct request *req) {
  struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
  req->event_type = EVENT_TYPE_WRITE;
  io_uring_prep_writev(sqe, req->client_socket, req->iov, req->iovec_count, 0);
  io_uring_sqe_set_data(sqe, req);
  io_uring_submit(&ring);
  return 0;
}

void _send_static_string_content(const char *str, int client_socket) {
  struct request *req = zh_malloc(sizeof(*req) + sizeof(struct iovec));
  unsigned long slen = strlen(str);
  req->iovec_count = 1;
  req->client_socket = client_socket;
  req->iov[0].iov_base = zh_malloc(slen);
  req->iov[0].iov_len = slen;
  memcpy(req->iov[0].iov_base, str, slen);
  add_write_request(req);
}

/*
 * When ZeroHTTPd encounters any other HTTP method other than GET or POST, this function
 * is used to inform the client.
 * */
void handle_unimplemented_method(int client_socket) {
  _send_static_string_content(unimplemented_content, client_socket);
}
/*
 * This function is used to send a "HTTP Not Found" code and message to the client in
 * case the file requested is not found.
 * */
void handle_http_404(int client_socket) {
  _send_static_string_content(http_404_content, client_socket);
}


void handle_get_method(char *path, int client_socket) {
  char final_path[1024];
  /*
   If a path ends in a trailing slash, the client probably wants the index
   file inside of that directory.
   */
  if (path[strlen(path) - 1] == '/') {
    strcpy(final_path, "public");
    strcat(final_path, path);
    strcat(final_path, "index.html");
  }
  else {
    strcpy(final_path, "public");
    strcat(final_path, path);
  }
  /* The stat() system call will give you information about the file
   * like type (regular file, directory, etc), size, etc. */
  struct stat path_stat;
  if (stat(final_path, &path_stat) == -1) {
    printf("404 Not Found: %s (%s)\n", final_path, path);
    handle_http_404(client_socket);
  }
  else {
    /* Check if this is a normal/regular file and not a directory or something else */
    if (S_ISREG(path_stat.st_mode)) {
      struct request *req = zh_malloc(sizeof(*req) + (sizeof(struct iovec) * 6));
      req->iovec_count = 6;
      req->client_socket = client_socket;
      send_headers(final_path, path_stat.st_size, req->iov);
      copy_file_contents(final_path, path_stat.st_size, &req->iov[5]);
      printf("200 %s %ld bytes\n", final_path, path_stat.st_size);
      add_write_request( req);
    }
    else {
      handle_http_404(client_socket);
      printf("404 Not Found: %s\n", final_path);
    }
  }
}
/*
 * This function looks at method used and calls the appropriate handler function.
 * Since we only implement GET and POST methods, it calls handle_unimplemented_method()
 * in case both these don't match. This sends an error to the client.
 * */
void handle_http_method(char *method_buffer, int client_socket) {
  char *method, *path, *saveptr;
  method = strtok_r(method_buffer, " ", &saveptr);
  strtolower(method);
  path = strtok_r(NULL, " ", &saveptr);
  if (strcmp(method, "get") == 0) {
    handle_get_method(path, client_socket);
  }
  else {
    handle_unimplemented_method(client_socket);
  }
}

int handle_client_request(struct request *req) {
  char http_request[1024];
  /* Get the first line, which will be the request */
  if(get_line(req->iov[0].iov_base, http_request, sizeof(http_request))) {
    fprintf(stderr, "Malformed request\n");
    exit(1);
  }
  handle_http_method(http_request, req->client_socket);
  return 0;
}

void server_loop(int server_socket) {
  struct io_uring_cqe *cqe;
  struct sockaddr_in client_addr;
  socklen_t client_addr_len = sizeof(client_addr);

  add_accept_request(server_socket, &client_addr, &client_addr_len);

  while (1) {
    int ret = io_uring_wait_cqe(&ring, &cqe);
    struct request *req = (struct request *) cqe->user_data;

    if (ret < 0)
      fatal_error("io_uring_wait_cqe");
    if (cqe->res < 0) {
      fprintf(stderr, "Async request failed: %s for event: %d\n",
              strerror(-cqe->res), req->event_type);
      exit(1);
    }

    switch (req->event_type) {
      case EVENT_TYPE_ACCEPT:
        add_accept_request(server_socket, &client_addr, &client_addr_len);
        add_read_request(cqe->res);
        free(req);
        break;
      case EVENT_TYPE_READ:
        if (!cqe->res) {
          fprintf(stderr, "Empty request!\n");
          break;
        }
        handle_client_request(req);
        free(req->iov[0].iov_base);
        free(req);
        break;
      case EVENT_TYPE_WRITE:
        for (int i = 0; i < req->iovec_count; i++) {
          free(req->iov[i].iov_base);
        }
        close(req->client_socket);
        free(req);
        break;
    }
    /* Mark this request as processed */
    io_uring_cqe_seen(&ring, cqe);
  }
}

void sigint_handler(int signo) {
  printf("^C pressed. Shutting down.\n");
  io_uring_queue_exit(&ring);
  exit(0);
}

int main() {
  int server_socket = setup_listening_socket(DEFAULT_SERVER_PORT);
  signal(SIGINT, sigint_handler);
  io_uring_queue_init(QUEUE_DEPTH, &ring, 0);
  server_loop(server_socket);
  return 0;
}