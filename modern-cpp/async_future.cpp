#include <future>

std::future<int> async_algo() {
  std::promise<int> p;
  auto f = p.get_future();
  std::thread t {[p = std::move(p)]() mutable {
    int answer = 1;
    p.set_value(answer);
  }};
  t.detach();
  return f;
}

int main() {
  auto f= async_algo();
  auto f2 = f.
}
