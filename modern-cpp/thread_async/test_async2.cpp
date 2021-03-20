#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <future>
#include <string>
#include <mutex>
#include <thread>

std::mutex m;

struct X {
  void foo(int i, const std::string& str) {
    std::thread::id this_id = std::this_thread::get_id();
    std::lock_guard<std::mutex> lk(m);
    std::cout << this_id << " " << str << ' ' << i << '\n';
  }
  void bar(const std::string& str) {
    std::thread::id this_id = std::this_thread::get_id();
    std::lock_guard<std::mutex> lk(m);
    std::cout << this_id << " "<< str << '\n';
  }
  int operator()(int i) {
    std::thread::id this_id = std::this_thread::get_id();
    std::lock_guard<std::mutex> lk(m);
    std::cout << this_id << " "<< i << '\n';
    return i + 10;
  }
};

int main()
{
  std::thread::id this_id = std::this_thread::get_id();
  std::cout << this_id << " main\n";

  X x;
  // Calls (&x)->foo(42, "Hello") with default policy:
  // may print "Hello 42" concurrently or defer execution
  auto a1 = std::async(&X::foo, &x, 42, "Hello");

  // Calls x.bar("world!") with deferred policy
  // prints "world!" when a2.get() or a2.wait() is called
  auto a2 = std::async(std::launch::deferred, &X::bar, x, "world!");

  // Calls X()(43); with async policy
  // prints "43" concurrently
  auto a3 = std::async(std::launch::async, X(), 43);

  a2.wait();                     // prints "world!"
  std::cout << a3.get() << '\n'; // prints "53"
} // if a1 is not done at this point, destructor of a1 prints "Hello 42" here
