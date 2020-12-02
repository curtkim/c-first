#include "tbb/flow_graph.h"
#include <chrono>
#include <thread>
#include <iostream>

using namespace tbb::flow;

tbb::atomic<size_t> g_cnt;

struct fn_body1 {
  tbb::atomic<size_t> &body_cnt;
  fn_body1(tbb::atomic<size_t> &b_cnt) : body_cnt(b_cnt) {}
  
  continue_msg operator()( continue_msg /*dont_care*/) {
    ++g_cnt;
    ++body_cnt;
    return continue_msg();
  }
};

void run_example1() {  // example for Flow_Graph_Single_Vs_Broadcast.xml
  graph g;

  tbb::atomic<size_t> b1;  // local counts
  tbb::atomic<size_t> b2;  // for each function _node body
  tbb::atomic<size_t> b3;  //

  function_node<continue_msg> f1(g,serial, fn_body1(b1));
  function_node<continue_msg> f2(g,serial, fn_body1(b2));
  function_node<continue_msg> f3(g,serial, fn_body1(b3));
  buffer_node<continue_msg> buf1(g);

  // single-push policy
  g_cnt = b1 = b2 = b3 = 0;
  make_edge(buf1,f1);
  make_edge(buf1,f2);
  make_edge(buf1,f3);
  buf1.try_put(continue_msg());
  buf1.try_put(continue_msg());
  buf1.try_put(continue_msg());
  g.wait_for_all();
  printf( "after single-push test, g_cnt == %d, b1==%d, b2==%d, b3==%d\n", (int)g_cnt, (int)b1, (int)b2, (int)b3);
  remove_edge(buf1,f1);
  remove_edge(buf1,f2);
  remove_edge(buf1,f3);


  // broadcast-push policy
  broadcast_node<continue_msg> bn(g);
  g_cnt = b1 = b2 = b3 = 0;
  make_edge(bn,f1);
  make_edge(bn,f2);
  make_edge(bn,f3);
  bn.try_put(continue_msg());
  bn.try_put(continue_msg());
  bn.try_put(continue_msg());
  g.wait_for_all();
  printf( "after broadcast-push test, g_cnt == %d, b1==%d, b2==%d, b3==%d\n", (int)g_cnt, (int)b1, (int)b2, (int)b3);
}

int main() {

  run_example1();
  return 0;
}