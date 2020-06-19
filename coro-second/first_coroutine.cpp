#include <coroutine>
#include <iostream>
#include <assert.h>


class resumable {
public:
  struct promise_type;
  using coro_handle = std::coroutine_handle<promise_type>;

  resumable(coro_handle handle) : handle_(handle) { assert(handle); }
  resumable(resumable&) = delete;
  resumable(resumable&&) = delete;

  bool resume() {
    if (not handle_.done())
      handle_.resume();
    return not handle_.done();
  }
  ~resumable() { handle_.destroy(); }

private:
  coro_handle handle_;
};

struct resumable::promise_type {
  using coro_handle = std::coroutine_handle<promise_type>;
  auto get_return_object() {
    return coro_handle::from_promise(*this);
  }
  auto initial_suspend() { return std::suspend_always(); }
  auto final_suspend() { return std::suspend_always(); }
  void return_void() {}
  void unhandled_exception() {
    std::terminate();
  }
};

resumable foo(){
  std::cout << "Hello" << std::endl;
  co_await std::suspend_always();
  std::cout << "Coroutine" << std::endl;
}

int main(){
  resumable res = foo();
  while (res.resume());
}