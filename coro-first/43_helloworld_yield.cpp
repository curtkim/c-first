#include <coroutine>
#include <iostream>
#include <cassert>
struct HelloWorldCoro {
  struct promise_type {
    int m_val;

    HelloWorldCoro get_return_object() { return this; }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(int val) {
        m_val = val;
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
    co_yield 1;
    std::cout << "World!" << std::endl;
}
int main() {
    HelloWorldCoro mycoro = print_hello_world();

    mycoro.m_handle.resume();
    assert(mycoro.m_handle.promise().m_val == 1);
    mycoro.m_handle.resume();
    return EXIT_SUCCESS;
}
