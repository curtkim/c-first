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
  std::cout << "Hello ";
  co_await std::suspend_always{};
  std::cout << "World!" << std::endl;
}

int main() {
  HelloWorldCoro mycoro = print_hello_world();
  std::cout << "\n\t1\n";
  mycoro.m_handle.resume();
  std::cout << "\n\t2\n";
  mycoro.m_handle.resume(); // Equal to mycoro.m_handle.resume();
  std::cout << "\n\t3\n";
  return EXIT_SUCCESS;
}
