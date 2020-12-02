//#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1
#include "tbb/flow_graph.h"
#include <cstdio>
#include <cassert>

class MyMessage {
  int my_key;
  float my_value;
public:
  MyMessage(int k = 0, float v = 0) : my_key(k), my_value(v) {}

  int key() const {
    return my_key;
  }

  float value() const {
    return my_value;
  }
};

int main() {
  using namespace tbb::flow;

  graph g;
  function_node<int, MyMessage> f1(
    g, unlimited,
    [](int i) { return MyMessage(i, (float) i); });
  function_node<int, MyMessage> f2(
    g, unlimited,
    [](int i) { return MyMessage(i, (float) 2 * i); });

  function_node<tuple<MyMessage, MyMessage>> f3(
    g, unlimited,
    [](const tuple<MyMessage, MyMessage> &t) {
      assert(get<0>(t).key() == get<1>(t).key());
      std::printf("The result is %f for key %d\n", get<0>(t).value() + get<1>(t).value(), get<0>(t).key());
    });

  join_node<tuple<MyMessage, MyMessage>, key_matching<int> > jn(
    g,
    [](MyMessage a) { return a.key(); },
    [](MyMessage b) { return b.key(); });

  make_edge(f1, input_port<0>(jn));
  make_edge(f2, input_port<1>(jn));
  make_edge(jn, f3);

  f1.try_put(1);
  f1.try_put(2);
  f2.try_put(2);
  f2.try_put(1);

  g.wait_for_all();
}