#include <thread>
#include <rxcpp/rx.hpp>

int main() {

  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  // create
  auto ints = rxcpp::sources::create<int>(
    [](rxcpp::subscriber<int> s) {
        std::cout << std::this_thread::get_id() << " in create" << std::endl;
        std::thread myThread([&s](){
            std::cout << std::this_thread::get_id() << " in create thread" << std::endl;
            s.on_next(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            s.on_next(2);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            s.on_completed();
        });
        myThread.join();
    });

  ints
    .observe_on(rxcpp::synchronize_new_thread())
    .subscribe(
      [](int v) {
          std::cout << std::this_thread::get_id() << " onNext " << v << std::endl;
      },
      []() {
          std::cout << std::this_thread::get_id() << " onComplete " << std::endl;
      }
    );
}