#include <thread>
#include <rxcpp/rx.hpp>
#include <asio.hpp>


int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  asio::io_context io_context;
  asio::posix::stream_descriptor stream{io_context, STDIN_FILENO};

  auto work = asio::make_work_guard(io_context);
  std::thread thread([&io_context](){
    std::cout << std::this_thread::get_id() << " io_context.run()" << std::endl;
    io_context.run();
    std::cout << std::this_thread::get_id() << " io_context.run() end" << std::endl;
  });

  // create
  auto keyin$ = rxcpp::sources::create<std::string>(
      [&stream](rxcpp::subscriber<std::string> s) {
        asio::streambuf buffer;
        std::cout << std::this_thread::get_id() << " before while" << std::endl;
        while(true) {
          std::future<size_t> length_future = async_read_until(stream, buffer, "\n", asio::use_future);
          size_t length = length_future.get();
          std::cout << std::this_thread::get_id() << " length=" << length << std::endl;
          if( length > 1){
            std::istream is(&buffer);
            std::string result_line;
            std::getline(is, result_line);
            std::cout << std::this_thread::get_id() << " line=" << result_line << std::endl;
            s.on_next(result_line);
          }
          else {
            s.on_completed();
            break;
          }
        }
      });

  std::cout << std::this_thread::get_id() << " before subscribe" << std::endl;
  keyin$
      .observe_on(rxcpp::serialize_event_loop())
      .subscribe([](std::string str){
        std::cout << std::this_thread::get_id() << " next=" << str << std::endl;
      });

  thread.detach();
  std::cout << std::this_thread::get_id() << " io thread detached" << std::endl;

  io_context.stop();
  std::cout << std::this_thread::get_id() << " io_context.stopped" << std::endl;

  return 0;
}