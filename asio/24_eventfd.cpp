#include <array>
#include <iostream>
#include <thread>
#include <asio.hpp>

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)


uint64_t value;

void read_from_stream(asio::posix::stream_descriptor& stream, asio::mutable_buffer& buffer) {
  async_read(stream, buffer, [&stream, &buffer](const std::error_code ec, std::size_t) {
    std::cout << std::this_thread::get_id() << " read " << value << std::endl;
    read_from_stream(stream, buffer);
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

  asio::mutable_buffer buffer = asio::buffer(&value, sizeof(value));
  read_from_stream(stream, buffer);

  std::thread thread([&efd, &start](){
    for(uint64_t i = 1; i < 10; i++){
      //std::cout << i << " write\n";
      int ret = write(efd, &i, sizeof(uint64_t));
      if (ret != 8)
        handle_error("[producer] failed to write eventfd");
      //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto diff = std::chrono::steady_clock::now() - start;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "elapsed time " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << "μs \n";
    std::exit(0);
  });

  //io_context.run();
  while(true){
    io_context.poll_one();
  }
  // https://topic.alibabacloud.com/a/boost-library-asio-io_service-and-run-run_one-poll-poll_one-differences_8_8_10260198.html
  thread.join();

  return 0;
}