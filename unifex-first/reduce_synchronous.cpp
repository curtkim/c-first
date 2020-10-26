#include <unifex/sync_wait.hpp>
#include <unifex/transform_stream.hpp>
#include <unifex/reduce_stream.hpp>
#include <unifex/transform.hpp>
#include <unifex/range_stream.hpp>

#include <cstdio>

using namespace unifex;

int main() {

  std::optional<int> result = sync_wait(
    reduce_stream(
      range_stream{0, 10},
      0,
      [](int state, int value) {
        return state + value;
      }
    )
  );

  std::printf("result = %i\n", result.value());

  return 0;
}