#include <array>
#include <iostream>
#include <asio.hpp>

#include <sys/inotify.h>

#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (2 * (EVENT_SIZE + 16))

// 정상 동작하지 않음 ㅠㅠ
int main() {
    asio::io_context io_context;

    int fd = inotify_init();
    printf("after init\n");

    if (fd < 0) {
        perror("inotify_init");
    }
    int wd = inotify_add_watch(fd, ".", IN_MODIFY | IN_CREATE | IN_DELETE);
    std::cout << "EVENT_SIZE " << EVENT_SIZE << std::endl;
    std::cout << "fd " << fd << std::endl;
    std::cout << "wd " << wd << std::endl;

    std::array<char, BUF_LEN> recv_buf;

    asio::posix::stream_descriptor stream{io_context, fd};
    asio::async_read(stream, asio::buffer(recv_buf), [recv_buf](const std::error_code ec, std::size_t length) {
        std::cout << "error " << ec << std::endl;
        std::cout << "read " << length <<  std::endl;
        int i = 0;
        while (i < length) {
            struct inotify_event *event =(struct inotify_event *) &recv_buf[i];
            //const inotify_event *event = reinterpret_cast<const inotify_event*>(&recv_buf[i]);
            std::cout << "wd=" << event->wd << " mask=" << event->mask << " name=" << event->name << " len=" << event->len << std::endl;
            if (event->len) {
                if (event->mask & IN_CREATE) {
                    printf("The file %s was created.\n", event->name);
                } else if (event->mask & IN_DELETE) {
                    printf("The file %s was deleted.\n", event->name);
                } else if (event->mask & IN_MODIFY) {
                    printf("The file %s was modified.\n", event->name);
                }
            }

            i += EVENT_SIZE + event->len;
            printf("event->len = %d, length= %d, i=%d \n", event->len, length, i);
        }
    });

    io_context.run();

    inotify_rm_watch(fd, wd);
    close(fd);

    return 0;
}