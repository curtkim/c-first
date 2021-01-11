#include <coroutine>
#include <iostream>
#include <assert.h>

class resumable {
public:
  struct promise_type;

  using coro_handle = std::coroutine_handle<promise_type>;

  resumable(coro_handle handle) : handle_(handle) {
    assert(handle);
  }
  //resumable(resumable&) = delete;
  //resumable(resumable&&) = delete;

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
    std::cout << "typeid(*this) " << typeid(*this).name() << std::endl; // N9resumable12promise_typeE
    return coro_handle::from_promise(*this);
    // Note that the type of the return-object doesn’t need to be the same type as the return-type of the coroutine function.
    // An implicit conversion from the return-object to the return-type of the coroutine is performed if necessary.
    // coro_handle을 반환하지만 implcit conversion으로 resumable로 변환하는 것 같다.
    // promise -> coro_handle -> coro_handle holder(return-type of coroutine)
  }
  auto initial_suspend() {
    return std::suspend_always();
  }
  auto final_suspend() {
    return std::suspend_always();
  }
  void return_void() {}
  void unhandled_exception() {
    std::terminate();
  }
  /*
  void* operator new(std::size_t) noexcept {
    return nullptr;
  }
  static resumable get_return_object_on_allocation_failure(){
    throw std::bad_alloc();
  }
   */
};

resumable foo(){
  std::cout << "Hello" << std::endl;
  co_await std::suspend_always();
  std::cout << "Coroutine" << std::endl;
  co_return;
}

int main(){
  resumable res = foo();
  while (res.resume()){
    std::cout << "in while" << std::endl;
  };
}
