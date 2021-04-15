#pragma once

#include <chrono>
#include <tuple>
#include "track.hpp"
#include "ring_span.hpp"

// fields(first field가 key column임)
// idx, name, type, size, minimum_size


const int QUEUE_SIZE = 20;


struct Frame {
  ring_span<std::tuple<Header, long>> lidar1;
  ring_span<std::tuple<Header, int>> camera1;
  ring_span<std::tuple<Header, int>> camera2;
  ring_span<std::tuple<Header, int>> gps1;
};


struct Timeline {
  Track<long> lidar1 = {QUEUE_SIZE};
  Track<int> camera1 = {QUEUE_SIZE};
  Track<int> camera2 = {QUEUE_SIZE};
  Track<int> gps1 = {QUEUE_SIZE};

  /*
  auto get_tracks() {
    return std::forward_as_tuple(lidar1, camera1, camera2);
  }
  */

  Frame frame(){
    return Frame{
      lidar1.span(),
      camera1.span(),
      camera2.span(),
      gps1.span()
    };
  }

  void release(Frame& frame) {
    for(auto& item : frame.lidar1)
      lidar1.dequeue();
    for(auto& item : frame.camera1)
      camera1.dequeue();
    for(auto& item : frame.camera2)
      camera2.dequeue();
    for(auto& item : frame.gps1)
      gps1.dequeue();
  }
};
