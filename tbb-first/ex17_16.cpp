#include <tbb/tbb.h>
#include <iostream>
#include <memory>
#include <thread>

const int A_VERY_LARGE_NUMBER = 1 << 12;

void spinWaitForAtLeast(double sec) {
  tbb::tick_count t0 = tbb::tick_count::now();
  while ((tbb::tick_count::now() - t0).seconds() < sec);
}

void warmupTBB() {
  tbb::parallel_for(0, tbb::task_scheduler_init::default_num_threads(),
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


void fig_17_16() {
  tbb::flow::graph g;

  int src_count = 0;
  tbb::flow::source_node<BigObjectPtr> source{
    g,
    [&](BigObjectPtr &m) -> bool {
      if (src_count < A_VERY_LARGE_NUMBER) {
        m = std::make_shared<BigObject>(src_count++);
        std::cout << std::this_thread::get_id() << " src id: " << m->get_id() << std::endl;
        return true;
      }
      return false;
    }, false
  };

  // rejecting policy
  // policy( rejection, reserving, queueing, lightweight)
  tbb::flow::function_node<BigObjectPtr, BigObjectPtr, tbb::flow::rejecting> limited_to_3_node{
    g,
    3, // <== concurrency
    [](BigObjectPtr m) {
      std::cout << std::this_thread::get_id() << " id: " << m->get_id() << std::endl;
      spinWaitForAtLeast(0.0001);
      return m;
    }
  };
  tbb::flow::make_edge(source, limited_to_3_node);

  bigObjectCount = 0;
  maxCount = 0;
  source.activate();
  g.wait_for_all();
  std::cout << "maxCount == " << maxCount << std::endl;
}

int main(int argc, char *argv[]) {
  warmupTBB();
  tbb::tick_count t0 = tbb::tick_count::now();
  fig_17_16();
  tbb::tick_count t1 = tbb::tick_count::now();
  std::cout << "Total time == " << (t1 - t0).seconds() << std::endl;
  return 0;
}