#include <iostream>
#include <memory>
#include <tbb/tbb.h>

const int A_VERY_LARGE_NUMBER = 1 << 12;

void spinWaitForAtLeast(double sec) {
  tbb::tick_count t0 = tbb::tick_count::now();
  while ((tbb::tick_count::now() - t0).seconds() < sec);
}

void warmupTBB() {
  tbb::parallel_for(
    0, tbb::task_scheduler_init::default_num_threads(),
    [](int) {
      spinWaitForAtLeast(0.001);
    });
}

tbb::atomic<int> bigObjectCount;
int maxCount = 0;

class BigObject {
  const int id;
  /* And a big amount of other data */

public:
  BigObject() : id(-1) {}

  BigObject(int i) : id(i) {
    int cnt = bigObjectCount.fetch_and_increment() + 1;
    if (cnt > maxCount)
      maxCount = cnt;
  }

  BigObject(const BigObject &b) : id(b.id) {}

  virtual ~BigObject() {
    bigObjectCount.fetch_and_decrement();
  }

  int get_id() const { return id; }
};

using BigObjectPtr = std::shared_ptr<BigObject>;


void fig_17_13() {
  tbb::flow::graph g;

  tbb::flow::function_node<BigObjectPtr, BigObjectPtr> serial_node{
    g, tbb::flow::serial,
    [](BigObjectPtr m) {
      spinWaitForAtLeast(0.0001);
      return m;
    }
  };

  tbb::flow::function_node<BigObjectPtr, BigObjectPtr> unlimited_node{
    g, tbb::flow::unlimited,
    [](BigObjectPtr m) {
      spinWaitForAtLeast(0.0001);
      return m;
    }
  };

  bigObjectCount = 0;
  for (int i = 0; i < A_VERY_LARGE_NUMBER; ++i) {
    serial_node.try_put(std::make_shared<BigObject>(i));
    unlimited_node.try_put(std::make_shared<BigObject>(i));
  }

  g.wait_for_all();
  std::cout << "maxCount == " << maxCount << std::endl;
}

int main(int argc, char *argv[]) {
  warmupTBB();

  tbb::tick_count t0 = tbb::tick_count::now();
  fig_17_13();
  tbb::tick_count t1 = tbb::tick_count::now();
  std::cout << "Total time == " << (t1 - t0).seconds() << std::endl;
  return 0;
}