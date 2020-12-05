//
// Created by curt on 20. 12. 5..
//

#ifndef SYS_URING_21_WEBSERVER_H
#define SYS_URING_21_WEBSERVER_H

#define SERVER_STRING           "Server: zerohttpd/0.1\r\n"


const char *unimplemented_content = \
        "HTTP/1.0 400 Bad Request\r\n"
        "Content-type: text/html\r\n"
        "\r\n"
        "<html>"
        "<head>"
        "<title>ZeroHTTPd: Unimplemented</title>"
        "</head>"
        "<body>"
        "<h1>Bad Request (Unimplemented)</h1>"
        "<p>Your client sent a request ZeroHTTPd did not understand and it is probably not your fault.</p>"
        "</body>"
        "</html>";
const char *http_404_content = \
        "HTTP/1.0 404 Not Found\r\n"
        "Content-type: text/html\r\n"
        "\r\n"
        "<html>"
        "<head>"
        "<title>ZeroHTTPd: Not Found</title>"
        "</head>"
        "<body>"
        "<h1>Not Found (404)</h1>"
        "<p>Your client is asking for an object that was not found on this server.</p>"
        "</body>"
        "</html>";


/*
 * Utility function to convert a string to lower case.
 * */
void strtolower(char *str) {
  for (; *str; ++str)
    *str = (char)tolower(*str);
}
/*
 One function that prints the system call and the error details
 and then exits with error code 1. Non-zero meaning things didn't go well.
 */
void fatal_error(const char *syscall) {
  perror(syscall);
  exit(1);
}
/*
 * Helper function for cleaner looking code.
 * */
void *zh_malloc(size_t size) {
  void *buf = malloc(size);
  if (!buf) {
    fprintf(stderr, "Fatal error: unable to allocate memory.\n");
    exit(1);
  }
  return buf;
}

/*
 * Once a static file is identified to be served, this function is used to read the file
 * and write it over the client socket using Linux's sendfile() system call. This saves us
 * the hassle of transferring file buffers from kernel to user space and back.
 * */
void copy_file_contents(char *file_path, off_t file_size, struct iovec *iov) {
  int fd;
  char *buf = zh_malloc(file_size);
  fd = open(file_path, O_RDONLY);
  if (fd < 0)
    fatal_error("read");
  /* We should really check for short reads here */
  int ret = read(fd, buf, file_size);
  if (ret < file_size) {
    fprintf(stderr, "Encountered a short read.\n");
  }
  close(fd);
  iov->iov_base = buf;
  iov->iov_len = file_size;
}

/*
 * Simple function to get the file extension of the file that we are about to serve.
 * */
const char *get_filename_ext(const char *filename) {
  const char *dot = strrchr(filename, '.');
  if (!dot || dot == filename)
    return "";
  return dot + 1;
}

/*
 * This function is responsible for setting up the main listening socket used by the
 * web server.
 * */
int setup_listening_socket(int port) {
  int sock;
  struct sockaddr_in srv_addr;
  sock = socket(PF_INET, SOCK_STREAM, 0);
  if (sock == -1)
    fatal_error("socket()");
  int enable = 1;
  if (setsockopt(sock,
                 SOL_SOCKET, SO_REUSEADDR,
                 &enable, sizeof(int)) < 0)
    fatal_error("setsockopt(SO_REUSEADDR)");
  memset(&srv_addr, 0, sizeof(srv_addr));
  srv_addr.sin_family = AF_INET;
  srv_addr.sin_port = htons(port);
  srv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  /* We bind to a port and turn this socket into a listening
   * socket.
   * */
  if (bind(sock,
           (const struct sockaddr *)&srv_addr,
           sizeof(srv_addr)) < 0)
    fatal_error("bind()");
  if (listen(sock, 10) < 0)
    fatal_error("listen()");
  return (sock);
}

/*
 * Sends the HTTP 200 OK header, the server string, for a few types of files, it can also
 * send the content type based on the file extension. It also sends the content length
 * header. Finally it send a '\r\n' in a line by itself signalling the end of headers
 * and the beginning of any content.
 * */
void send_headers(const char *path, off_t len, struct iovec *iov) {
  char small_case_path[1024];
  char send_buffer[1024];
  strcpy(small_case_path, path);
  strtolower(small_case_path);
  char *str = "HTTP/1.0 200 OK\r\n";
  unsigned long slen = strlen(str);
  iov[0].iov_base = zh_malloc(slen);
  iov[0].iov_len = slen;
  memcpy(iov[0].iov_base, str, slen);
  slen = strlen(SERVER_STRING);
  iov[1].iov_base = zh_malloc(slen);
  iov[1].iov_len = slen;
  memcpy(iov[1].iov_base, SERVER_STRING, slen);
  /*
   * Check the file extension for certain common types of files
   * on web pages and send the appropriate content-type header.
   * Since extensions can be mixed case like JPG, jpg or Jpg,
   * we turn the extension into lower case before checking.
   * */
  const char *file_ext = get_filename_ext(small_case_path);
  if (strcmp("jpg", file_ext) == 0)
    strcpy(send_buffer, "Content-Type: image/jpeg\r\n");
  if (strcmp("jpeg", file_ext) == 0)
    strcpy(send_buffer, "Content-Type: image/jpeg\r\n");
  if (strcmp("png", file_ext) == 0)
    strcpy(send_buffer, "Content-Type: image/png\r\n");
  if (strcmp("gif", file_ext) == 0)
    strcpy(send_buffer, "Content-Type: image/gif\r\n");
  if (strcmp("htm", file_ext) == 0)
    strcpy(send_buffer, "Content-Type: text/html\r\n");
  if (strcmp("html", file_ext) == 0)
    strcpy(send_buffer, "Content-Type: text/html\r\n");
  if (strcmp("js", file_ext) == 0)
    strcpy(send_buffer, "Content-Type: application/javascript\r\n");
  if (strcmp("css", file_ext) == 0)
    strcpy(send_buffer, "Content-Type: text/css\r\n");
  if (strcmp("txt", file_ext) == 0)
    strcpy(send_buffer, "Content-Type: text/plain\r\n");
  slen = strlen(send_buffer);
  iov[2].iov_base = zh_malloc(slen);
  iov[2].iov_len = slen;
  memcpy(iov[2].iov_base, send_buffer, slen);
  /* Send the content-length header, which is the file size in this case. */
  sprintf(send_buffer, "content-length: %ld\r\n", len);
  slen = strlen(send_buffer);
  iov[3].iov_base = zh_malloc(slen);
  iov[3].iov_len = slen;
  memcpy(iov[3].iov_base, send_buffer, slen);
  /*
   * When the browser sees a '\r\n' sequence in a line on its own,
   * it understands there are no more headers. Content may follow.
   * */
  strcpy(send_buffer, "\r\n");
  slen = strlen(send_buffer);
  iov[4].iov_base = zh_malloc(slen);
  iov[4].iov_len = slen;
  memcpy(iov[4].iov_base, send_buffer, slen);
}

int get_line(const char *src, char *dest, int dest_sz) {
  for (int i = 0; i < dest_sz; i++) {
    dest[i] = src[i];
    if (src[i] == '\r' && src[i+1] == '\n') {
      dest[i] = '\0';
      return 0;
    }
  }
  return 1;
}

#endif //SYS_URING_21_WEBSERVER_H

