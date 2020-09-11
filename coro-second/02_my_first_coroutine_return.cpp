#include <coroutine>
#include <iostream>
#include <assert.h>


class resumable {
public:
  struct promise_type {
    const char* string_;

    using coro_handle = std::coroutine_handle<promise_type>;
    auto get_return_object() {
      return coro_handle::from_promise(*this);
    }
    auto initial_suspend() {
      return std::suspend_always();
    }
    auto final_suspend() {
      return std::suspend_always();
    }
    //void return_void() {}
    void return_value(const char* string){
      string_ = string;
    }

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

  using coro_handle = std::coroutine_handle<promise_type>;

  resumable(coro_handle handle) : handle_(handle) { assert(handle); }
  //resumable(resumable&) = delete;
  //resumable(resumable&&) = delete;

  bool resume() {
    if (not handle_.done())
      handle_.resume();
    return not handle_.done();
  }
  ~resumable() { handle_.destroy(); }

  const char* return_val(){
    return handle_.promise().string_;
  }

private:
  coro_handle handle_;
};

resumable foo(){
  std::cout << "Hello" << std::endl;
  co_await std::suspend_always();
  std::cout << "Coroutine" << std::endl;
  co_return "Coroutine end";
}

int main(){
  resumable res = foo();
  while (res.resume()){
    std::cout << "in while" << std::endl;
  };
  std::cout << res.return_val() << std::endl;
}