// https://luncliff.github.io/coroutine/articles/russian-roulette-kor/
#include <array>
#include <random>
#include <iostream>

#include <coroutine>

#include <gsl/gsl>

using namespace std;

auto select_chamber() -> int {
  std::random_device device{};
  std::mt19937_64 gen{device()};
  return static_cast<int>(gen());
}



class promise_manual_control {
public:
  auto initial_suspend() noexcept {
    return suspend_always{};
  }
  auto final_suspend() noexcept {
    return suspend_always{};
  }
  void unhandled_exception() noexcept {
  }
};

// 1. Coroutine Interface Type
class user_behavior_t : public coroutine_handle<void> {
public:

  // 2. Promise
  class promise_type final : public promise_manual_control {
  public:
    void return_void() noexcept {}
    auto get_return_object() noexcept -> user_behavior_t {
      return {this};
    }
  };

private:
  user_behavior_t(promise_type* p) noexcept : coroutine_handle<void>{} {
    coroutine_handle<void>& self = *this;
    self = coroutine_handle<promise_type>::from_promise(*p);
  }

public:
  user_behavior_t() noexcept = default;
};


// 3. Awaiter
// -------
// trigger fires the bullet
// all players will 'wait' for it
class trigger_t {
protected:
  const int& loaded;
  int current;

public:
  trigger_t(const int& _loaded, int _current) : loaded{_loaded}, current{_current} {
    printf("constructor: current=%d loaded=%d\n", current, loaded);
  }

private:
  bool pull() { // pull the trigger. is it the bad case?
    printf("current=%d loaded=%d\n", current, loaded);
    return --current == loaded;
  }

public:
  bool await_ready() {
    return false;
  }
  void await_suspend(coroutine_handle<void>) {
  }
  bool await_resume() {
    return pull();
  }
};

// revolver knows which is the loaded chamber
class revolver_t : public trigger_t {
  const int loaded;

public:
  revolver_t(int _loaded, int num_player) : loaded{_loaded}, trigger_t{loaded, num_player} {}
};


// 어떻게 return type이 user_behavior_t가 될 수 있을까?
//  this player will ...
//  1. be bypassed
//     (fired = false; then return)
//  2. receive the bullet
//     (fired = true; then return)
//  3. be skipped because of the other player became a victim
//     (destroyed when it is suspended - no output)
auto make_player(gsl::index id, bool& fired, trigger_t& trigger) -> user_behavior_t {
  // bang !
  fired = co_await trigger;
  fired ? printf("player %zu dead  :( \n", id)
        : printf("player %zu alive :) \n", id);
}

// russian_roulette은 coroutine은 아니다.
// user_behavior_t가 coroutine이자 coroutine_handle이다..
// trigger가 awaiter이다.
// 여러개의 coroutine이 하나의 awaiter를 사용한다.

// the game will go on until the revolver fires its bullet
void russian_roulette(revolver_t& revolver, gsl::span<user_behavior_t> users) {
  bool fired = false;

  // spawn player coroutines with their id
  gsl::index id{};
  for (auto& user : users)
    user = make_player(++id, fired, revolver);

  // cleanup the game on return
  auto on_finish = gsl::finally([users] {
    for (coroutine_handle<void>& frame : users) {
      printf("destroy user\n");
      frame.destroy();
    }
  });

  // until there is a victim ...
  for (int i = 0; fired == false; i = (i + 1) % users.size()) {
    // continue the users' behavior in round-robin manner
    coroutine_handle<void>& task = users[i];
    if (task.done() == false)
      task.resume();
  }
}


int main(int, char*[]) {
  // select some chamber with the users
  array<user_behavior_t, 6> users{};

  const int loaded = select_chamber() % users.size();
  std::cout << "loaded=" << loaded << std::endl;
  revolver_t revolver{loaded, users.max_size()};

  russian_roulette(revolver, users);
  return EXIT_SUCCESS;
}
