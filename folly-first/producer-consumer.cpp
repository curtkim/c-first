#include <iostream>
#include <thread>
#include <folly/ProducerConsumerQueue.h>
#include <folly/FBString.h>

int i = 0;
static folly::fbstring source() {
  return std::to_string(i++);
}

int main() {
  const int size = 10;
  folly::ProducerConsumerQueue<folly::fbstring> queue{size};

  std::thread reader([&queue] {
    for (;;) {
      folly::fbstring str;
      while (!queue.read(str)) {
        //spin until we get a value
        continue;
      }

      std::cout << std::this_thread::get_id() << " " << str << std::endl;
    }
  });

  // producer thread:
  for (;;) {
    folly::fbstring str = source();
    while (!queue.write(str)) {
      //spin until the queue has room
      continue;
    }
  }
}