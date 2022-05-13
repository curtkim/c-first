// https://nmilo.ca/blog/coroutines.html
#include <iostream>
#include <coroutine>


struct Task{
    struct promise_type {
        int value;

        Task get_return_object() {
          return {std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_never initial_suspend() {return {};}
        std::suspend_never final_suspend() noexcept {return {};}
        void unhandled_exception(){}

        std::suspend_always yield_value(int val) { value=val; return{};}
        void return_value(int val){ value=val;}
    };

    std::coroutine_handle<promise_type> handle;

    int value(){
      return handle.promise().value;
    }
    void resume(){
      handle.resume();
    }
};

struct Awaiter2 {
    bool await_ready(){ return true;}
    void await_suspend(std::coroutine_handle<>){}
    int await_resume(){return 3;}
};

Awaiter2 my_async_function(){
  return {};
}

Task my_coroutine(){
  int result = co_await my_async_function();
  co_yield result*2;
  co_return result+5;
}

int main(){
  Task task = my_coroutine();
  std::cout << task.value() << "\n";
  task.resume();
  std::cout << task.value() << "\n";
}