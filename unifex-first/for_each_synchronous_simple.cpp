#include <unifex/sync_wait.hpp>
#include <unifex/transform_stream.hpp>
#include <unifex/for_each.hpp>
#include <unifex/transform.hpp>
#include <unifex/range_stream.hpp>

#include <cstdio>

using namespace unifex;

int main() {
  sync_wait(
    for_each(
      range_stream{0, 5},
      [](int value) { std::printf("got %i\n", value); }
    )
  );
  std::printf("done\n");
  
  sync_wait(
    transform(
      for_each(
        range_stream{0, 5},
        [](int value) { std::printf("got %i\n", value); }
      ),
      []() { std::printf("done\n"); }
    )
  );

  return 0;
}