#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>
#include <unifex/via.hpp>
#include <unifex/just.hpp>

#include <cstdio>

using namespace unifex;

int main() {
  sync_wait(
    via(
      transform(just(1), [](int a) { // Successor
        std::printf("%d\n", a);
      }),
      transform(just(2), [](int b) { // Predecessor
        std::printf("%d\n", b);
      })
    )
  );
  return 0;
}