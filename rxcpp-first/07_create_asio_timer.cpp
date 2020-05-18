#include <thread>
#include <rxcpp/rx.hpp>
#include <asio.hpp>

#include "rx-asio.hpp"

int main() {

  asio::io_context io_context;
  asio::system_timer timer(io_context,std::chrono::system_clock::now() + std::chrono::seconds(1));

  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  auto asio_coordinate = rxcpp::observe_on_asio(io_context);

  // create
  auto ints = rxcpp::sources::create<int>(
    [&timer](rxcpp::subscriber<int> s) {
        std::cout << std::this_thread::get_id() << " in create" << std::endl;
        s.on_next(1);

        timer.async_wait([&s](const asio::error_code & error) {
          std::cout << error << std::endl;
          std::cout << std::this_thread::get_id() << " in timer" << std::endl;
          s.on_next(2);
          std::cout << std::this_thread::get_id() << " on_next" << std::endl;
          s.on_completed();
          std::cout << std::this_thread::get_id() << " on_completed" << std::endl;
        });
    }).subscribe_on(asio_coordinate).publish();

  ints
    //.observe_on(asio_coordinate)
    .observe_on(rxcpp::synchronize_new_thread())
    //.observe_on(rxcpp::observe_on_new_thread())
    //.observe_on(rxcpp::observe_on_asio(io_context))
    .subscribe(
      [](int v) {
          std::cout << std::this_thread::get_id() << " onNext " << v << std::endl;
      },
      []() {
          std::cout << std::this_thread::get_id() << " onComplete " << std::endl;
      }
    );

  ints.connect();

  std::cout << std::this_thread::get_id() << " 9" << std::endl;

  std::thread thread([&io_context](){
    std::cout << std::this_thread::get_id() << " in thread" << std::endl;
    io_context.run();
  });

  thread.join();
  //io_context.stop();

  std::this_thread::sleep_for(std::chrono::seconds(5));
  std::cout << std::this_thread::get_id() << " 10" << std::endl;
  return 0;
}