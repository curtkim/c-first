#include <unifex/when_all.hpp>
#include <unifex/materialize.hpp>
#include <unifex/dematerialize.hpp>
#include <unifex/transform.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/single_thread_context.hpp>
#include <unifex/scheduler_concepts.hpp>

#include <cassert>
#include <optional>

using namespace unifex;

int main() {
  single_thread_context ctx;

  std::optional<int> result = sync_wait(
    dematerialize(
      materialize(
        transform(
          schedule(ctx.get_scheduler()),
          []() { return 42; }))));


  assert(result.value() == 42);

  return 0;
}