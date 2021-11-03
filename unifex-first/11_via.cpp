#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>
#include <unifex/via.hpp>
#include <unifex/just.hpp>

#include <cstdio>

using namespace unifex;

int main() {
  int context = 1;

  sync_wait(
    via(
      transform(
        just(1),
        [&](int a) {
          std::printf("%d %d\n", a, context); }
      ),  // Successor
      transform(
        just(2),
        [&](int b) {
          context = b;
          std::printf("%d\n", b);
        }
      ) // Predecessor
    )
  );
  return 0;
}