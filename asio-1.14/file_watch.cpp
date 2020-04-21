#include <array>
#include <iostream>
#include <asio.hpp>

#include <sys/inotify.h>

#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (1024 * (EVENT_SIZE + 16))


int main() {
    asio::io_context io_context;

    int fd = inotify_init();
    printf("after init\n");

    if (fd < 0) {
        perror("inotify_init");
    }
    int wd = inotify_add_watch(fd, ".", IN_MODIFY | IN_CREATE | IN_DELETE);

    std::array<char, BUF_LEN> recv_buf;

    asio::posix::stream_descriptor stream{io_context, fd};
    io_context.run();

    std::future<std::size_t> recv_length = asio::async_read(stream, asio::buffer(recv_buf), asio::use_future);

    // 작업중

    return 0;
}