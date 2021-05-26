// http://www.vishalchovatiya.com/cpp20-coroutine-under-the-hood/
#include <coroutine>
#include <iostream>

struct HelloWorldCoro {
  struct promise_type { // compiler looks for `promise_type`
    HelloWorldCoro get_return_object() { return this; }

    std::suspend_always initial_suspend() {
      std::cout << "initial_suspend\n";
      return {};
    }
    std::suspend_always final_suspend() noexcept {
      std::cout << "final_suspend\n";
      return {};
    }

    void unhandled_exception() {
      std::exit(1);
    }
  };

  HelloWorldCoro(promise_type* p) : m_handle(std::coroutine_handle<promise_type>::from_promise(*p)) {}
  ~HelloWorldCoro() { m_handle.destroy(); }

  std::coroutine_handle<promise_type>      m_handle;
};

HelloWorldCoro print_hello_world() {
  std::cout << "Hello\n";
  co_await std::suspend_always{};
  std::cout << "World!\n";
}

int main() {
  HelloWorldCoro mycoro = print_hello_world();
  std::cout << "1\n";
  mycoro.m_handle.resume();
  std::cout << "2\n";
  mycoro.m_handle(); // Equal to mycoro.m_handle.resume();
  std::cout << "3\n";

  // initial_suspend
  // 1
  // Hello
  // 2
  // World
  // final_suspend
  // 3

  // final_suspend의 suspend_always는 누가 resume하는 것인가?

  return EXIT_SUCCESS;
}
