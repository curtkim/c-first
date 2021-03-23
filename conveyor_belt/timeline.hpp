#pragma once

#include <chrono>
#include <tuple>
#include "track.hpp"
#include "nonstd/ring_span.hpp"

// fields(first field가 key column임)
// idx, name, type, size, minimum_size


const int QUEUE_SIZE = 10;




struct Frame {
  nonstd::ring_span<std::tuple<Header, long>> lidar1;
  nonstd::ring_span<std::tuple<Header, int>> camera1;
  nonstd::ring_span<std::tuple<Header, int>> camera2;
  nonstd::ring_span<std::tuple<Header, int>> gps1;

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
      nonstd::ring_span<std::tuple<Header, long>>(
        lidar1.data.data(), lidar1.data.data()+lidar1.data.size(), lidar1.data.data()+lidar1.front, lidar1.rear
      ),
      nonstd::ring_span<std::tuple<Header, int>>(
        camera1.data.data(), camera1.data.data()+camera1.data.size(), camera1.data.data()+camera1.front, camera1.rear
      ),
      nonstd::ring_span<std::tuple<Header, int>>(
        camera2.data.data(), camera2.data.data()+camera2.data.size(), camera2.data.data()+camera2.front, camera1.rear
      ),
      nonstd::ring_span<std::tuple<Header, int>>(
        gps1.data.data(), gps1.data.data()+gps1.data.size(), gps1.data.data()+gps1.front, gps1.rear
      )
    };
  }
};
