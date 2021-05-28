#include <array>

#include <20_coroutine/linux.h>
#include <20_coroutine/return.h>

using namespace std;
using namespace coro;

auto wait_for_multiple_times(epoll_owner& ep, event& event, const char* name, uint32_t counter) -> frame_t {
  int i = 0;
  while (i < counter) {
    printf("[%s] i=%d\n", name, i);
    co_await wait_in(ep, event);
    i++;
  }
}

void resume_signaled_tasks(epoll_owner& ep) {
  array<epoll_event, 10> events{};
  auto count = ep.wait(1000, events); // wait for 1 sec
  printf("epoll_event count = %d\n", count);

  if (count == 0)
    return;

  for_each(events.begin(), events.begin() + count, [](epoll_event& e) {
    printf("e.data.ptr = %p\n", e.data.ptr);
    auto coro = coroutine_handle<void>::from_address(e.data.ptr);
    coro.resume();
  });
}

int main(int, char*[]) {
  epoll_owner ep{};
  event e1{};
  //event e2{};
  printf("e1.fd = %d\n", e1.fd());
  //printf("e2.fd = %d\n", e2.fd());

  wait_for_multiple_times(ep, e1, "e1", 6); // 6 await
  //wait_for_multiple_times(ep, e2, "e2", 6); // 왜 e2는 wait에 안 나오는 것인가?

  auto repeat = 8u;
  while (repeat--) {
    e1.set();
    //e2.set();
    // resume if there is available event-waiting coroutines
    resume_signaled_tasks(ep);
  };
  return 0;
}