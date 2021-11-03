#include <unifex/scheduler_concepts.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/timed_single_thread_context.hpp>
#include <unifex/transform.hpp>
#include <unifex/when_all.hpp>
#include <unifex/just.hpp>

#include <chrono>
#include <iostream>

using namespace unifex;
using namespace std::chrono;
using namespace std::chrono_literals;

int main() {

  int count = 0;

  // no return, parameter 1개
  just()
  | transform([&]{ ++count; })
  | sync_wait();
  assert(count == 1);

  sync_wait(
    just() |
    transform([&](){ ++count; })
  );
  assert(count == 2);


  // parameter 2개
  transform(
    transform(
      just(1),
      [](int i){ return i+1;}
    ),
    [](int a){
      assert(a == 2);
    }
  ) | sync_wait();

}