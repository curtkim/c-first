#include <array>
#include <iostream>
#include <thread>
#include <asio.hpp>

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)


uint64_t value;

void read_from_stream(asio::posix::stream_descriptor& stream) {
  auto buffer = asio::buffer(&value, sizeof(value));
  async_read(stream, buffer, [&stream](const std::error_code ec, std::size_t) {
    //std::cout << std::this_thread::get_id() << " read " << value << std::endl;
    read_from_stream(stream);
  });
}

int main() {

  auto start = std::chrono::steady_clock::now();

  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  int efd;
  efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd == -1)
    handle_error("eventfd");

  asio::io_context io_context;
  asio::posix::stream_descriptor stream{io_context, efd};

  read_from_stream(stream);

  std::thread thread([&efd, &start](){
    for(uint64_t i = 1; i < 1000; i++){
      //std::cout << i << " write\n";
      int ret = write(efd, &i, sizeof(uint64_t));
      if (ret != 8)
        handle_error("[producer] failed to write eventfd");
      //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto diff = std::chrono::steady_clock::now() - start;
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << " \n";
    std::exit(0);
  });

  io_context.run();
  thread.join();

  return 0;
}