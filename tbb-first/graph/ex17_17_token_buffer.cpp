#include <iostream>
#include <memory>
#include <tbb/tbb.h>
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

void fig_17_17() {
  using token_t = int;
  tbb::flow::graph g;
  int src_count = 0;

  tbb::flow::source_node<BigObjectPtr> source{
    g,
    [&](BigObjectPtr &m) -> bool {
      if (src_count < A_VERY_LARGE_NUMBER) {
        m = std::make_shared<BigObject>(src_count++);
        return true;
      }
      return false;
    }, false};

  tbb::flow::buffer_node<token_t> token_buffer{g};
  tbb::flow::join_node<std::tuple<BigObjectPtr, token_t>, tbb::flow::reserving> join{g};

  tbb::flow::function_node<std::tuple<BigObjectPtr, token_t>, token_t> unlimited_node{
    g, tbb::flow::unlimited,
    [](const std::tuple<BigObjectPtr, token_t> &m) {
      spinWaitForAtLeast(0.0001);
      std::cout << std::this_thread::get_id() << " " << std::get<0>(m)->get_id() << " " << std::get<1>(m) << std::endl;
      return std::get<1>(m);
    }};

  tbb::flow::make_edge(source, tbb::flow::input_port<0>(join));
  tbb::flow::make_edge(token_buffer, tbb::flow::input_port<1>(join));
  tbb::flow::make_edge(join, unlimited_node);
  tbb::flow::make_edge(unlimited_node, token_buffer);

  bigObjectCount = 0;
  maxCount = 0;
  for (token_t i = 0; i < 3; ++i)
    token_buffer.try_put(i);

  source.activate();
  g.wait_for_all();
  std::cout << "maxCount == " << maxCount << std::endl;
}

int main(int argc, char *argv[]) {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;
  warmupTBB();
  tbb::tick_count t0 = tbb::tick_count::now();
  fig_17_17();
  tbb::tick_count t1 = tbb::tick_count::now();
  std::cout << "Total time == " << (t1 - t0).seconds() << std::endl;
  return 0;
}