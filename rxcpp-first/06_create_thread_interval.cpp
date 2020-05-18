#include <thread>
#include <rxcpp/rx.hpp>

int main() {

  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  // create
  auto ints = rxcpp::sources::create<int>(
    [](rxcpp::subscriber<int> s) {
        std::cout << std::this_thread::get_id() << " in create" << std::endl;
        std::thread myThread([&s](){
            for(int i = 0; i < 5; i++){
              s.on_next(i);
              std::cout << std::this_thread::get_id() << " on_next in create thread" << std::endl;
              std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
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