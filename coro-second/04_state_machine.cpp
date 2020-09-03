#include <optional>
#include <coroutine>
#include <iostream>
#include <assert.h>


enum class button_press {LEFT_MOUSE, RIGHT_MOUSE};


template <typename Signal, typename Result>
class StateMachine {
public:
  struct promise_type {
    std::optional<Signal> recent_signal;
    std::optional<Result> returned_value;
    StateMachine get_return_object() {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_always final_suspend() { return {}; }
    void unhandled_exception() {
      auto exceptionPtr = std::current_exception();
      if(exceptionPtr)
        std::rethrow_exception(exceptionPtr);
    }
    void return_value(Result value) { returned_value.emplace(value); };

    struct SignalAwaiter {
      std::optional<Signal> &recent_signal;
      SignalAwaiter(std::optional<Signal> &signal) : recent_signal(signal) {}

      bool await_ready() { return recent_signal.has_value(); }
      void await_suspend(std::coroutine_handle<promise_type> coro_handle) {}
      Signal await_resume() {
        assert(recent_signal.has_value());
        Signal tmp = *recent_signal;
        recent_signal.reset();
        return tmp;
      }
    };

    SignalAwaiter await_transform(Signal) {
      return SignalAwaiter(recent_signal);
    }
  };


  //struct signal {};

  using coro_handle = std::coroutine_handle<promise_type>;
  StateMachine(coro_handle coro_handle) : coroutine_handle(coro_handle) {}
  StateMachine(StateMachine &&) = default;
  StateMachine(const StateMachine &) = delete;
  ~StateMachine() { coroutine_handle.destroy(); }

  void send_signal(Signal signal) {
    coroutine_handle.promise().recent_signal = signal;
    if (not coroutine_handle.done())
      coroutine_handle.resume();
  }
  std::optional<Result> get_result() {
    return coroutine_handle.promise().returned_value;
  }

private:
  coro_handle coroutine_handle;
};


StateMachine<button_press, std::FILE *> open_file(const char *file_name) {
  using this_coroutine = StateMachine<button_press, std::FILE *>;

  button_press first_button = co_await this_coroutine::signal{};
  while (true) {
    button_press second_button = co_await this_coroutine::signal{};
    if (first_button == button_press::LEFT_MOUSE and second_button == button_press::LEFT_MOUSE)
      co_return std::fopen(file_name, "r");

    first_button = second_button;
  }
}


int main() {
  auto machine = open_file("test");
  machine.send_signal(button_press::LEFT_MOUSE);
  machine.send_signal(button_press::RIGHT_MOUSE);
  machine.send_signal(button_press::LEFT_MOUSE);
  machine.send_signal(button_press::LEFT_MOUSE);

  auto result = machine.get_result();
  /*
  milli::raii close_guard ([&result] {
    if (result.has_value())
      std::fclose(*result);
  });
   */

  std::cout << result.value() << std::endl;
  return 0;
}