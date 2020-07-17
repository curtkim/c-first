#include <tbb/tbb.h>
#include <string>
#include <iostream>
#include <memory>

static inline void spinWaitForAtLeast(double sec) {
  tbb::tick_count t0 = tbb::tick_count::now();
  while ((tbb::tick_count::now() - t0).seconds() < sec);
}

void warmupTBB() {
  tbb::parallel_for(0, tbb::task_scheduler_init::default_num_threads(),
                    [](int) {
                      spinWaitForAtLeast(0.001);
                    });
}

struct Message {
  size_t my_seq_no;
  std::string my_string;

  Message(int i) : my_seq_no(i), my_string(std::to_string(i)) {}
};

using MessagePtr = std::shared_ptr<Message>;

void fig_17_21() {
  const int N = 10;
  tbb::flow::graph g;
  tbb::flow::function_node<MessagePtr, MessagePtr> first_node{
    g, tbb::flow::unlimited,
    [](MessagePtr m) {
      m->my_string += " with sequencer";
      return m;
    }};

  tbb::flow::sequencer_node<MessagePtr> sequencer(g, [](MessagePtr m) {
    return m->my_seq_no;
  });

  tbb::flow::function_node<MessagePtr, int, tbb::flow::rejecting> last_node{
    g, tbb::flow::serial,
    [](MessagePtr m) {
      std::cout << m->my_string << std::endl;
      return 0;
    }};
  tbb::flow::make_edge(first_node, sequencer);
  tbb::flow::make_edge(sequencer, last_node);

  for (int i = 0; i < N; ++i)
    first_node.try_put(std::make_shared<Message>(i));
  g.wait_for_all();
}

void fig_17_21_no_sequencer() {
  const int N = 10;
  tbb::flow::graph g;
  tbb::flow::function_node<MessagePtr, MessagePtr> first_node{
    g, tbb::flow::unlimited,
    [](MessagePtr m) {
      m->my_string += " no sequencer";
      return m;
    }};
  tbb::flow::function_node<MessagePtr, int> last_node{
    g, tbb::flow::serial,
    [](MessagePtr m) {
      std::cout << m->my_string << std::endl;
      return 0;
    }};
  tbb::flow::make_edge(first_node, last_node);

  for (int i = 0; i < N; ++i)
    first_node.try_put(std::make_shared<Message>(i));
  g.wait_for_all();
}


int main() {
  warmupTBB();
  tbb::tick_count t0 = tbb::tick_count::now();
  fig_17_21_no_sequencer();
  double time_wo_sequencer = (tbb::tick_count::now() - t0).seconds();
  std::cout << std::endl;

  warmupTBB();
  t0 = tbb::tick_count::now();
  fig_17_21();
  double time_with_sequencer = (tbb::tick_count::now() - t0).seconds();
  std::cout << std::endl;

  std::cout << "Time with no sequencer == " << time_wo_sequencer << std::endl
            << "Time with sequencer == " << time_with_sequencer << std::endl;
  return 0;
}
