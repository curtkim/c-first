#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>
#include <unifex/via.hpp>
#include <unifex/just.hpp>

#include <cstdio>

using namespace unifex;

int main() {
  sync_wait(
    via(
      transform(
        just(1),
        [](int a) { std::printf("%d\n", a); }
      ),  // Successor
      transform(
        just(2),
        [](int b) { std::printf("%d\n", b);}
      ) // Predecessor
    )
  );
  return 0;
}