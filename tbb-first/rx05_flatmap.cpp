#include <iostream>
#include <thread>
#include <tbb/tbb.h>


struct Message {
  int id;
  int data;
};

static inline void spinWaitForAtLeast(double sec) {
  tbb::tick_count t0 = tbb::tick_count::now();
  while ((tbb::tick_count::now() - t0).seconds() < sec);
}

int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  tbb::task_scheduler_init init{4};

  tbb::flow::graph g;

  // Due to parallelism the node can push messages to its successors in any order
  tbb::flow::function_node<int, Message> prefare(
    g, tbb::flow::unlimited,
    [](int i) -> Message {
      //printf("process id: %d\n", msg.id);
      return {i, i};
    });

  tbb::flow::function_node<Message, Message> process(
    g, tbb::flow::unlimited,
    [](const Message &msg) -> Message {
      spinWaitForAtLeast((10-msg.data) * 0.1);
      std::cout << std::this_thread::get_id() << " done : " << msg.data << " id=" << msg.id << std::endl;
      return msg;
    });

  tbb::flow::sequencer_node<Message> ordering(
    g,
    [](const Message &msg) -> int {
      return msg.id;
    });

  tbb::flow::function_node<Message> writer(
    g, tbb::flow::serial,
    [](const Message &msg) {
      std::cout << std::this_thread::get_id() << " Message recieved with : " << msg.data << " id=" << msg.id << std::endl;
    });

  tbb::flow::make_edge(prefare, process);
  tbb::flow::make_edge(process, ordering);
  tbb::flow::make_edge(ordering, writer);

  for (int i = 0; i < 10; ++i) {
    prefare.try_put(i);
  }

  g.wait_for_all();
}