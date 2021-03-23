#pragma once

#include <chrono>
#include <tuple>
#include "track.hpp"
#include "nonstd/ring_span.hpp"

// fields(first field가 key column임)
// idx, name, type, size, minimum_size


const int QUEUE_SIZE = 10;




struct TimeSpan {
  nonstd::ring_span<long> lidar;
  nonstd::ring_span<int> camera1;
  nonstd::ring_span<int> camera2;

};

/*
struct Timeline {
  Track<long> lidar1 = {QUEUE_SIZE};
  Track<int> camera1 = {QUEUE_SIZE};
  Track<int> camera2 = {QUEUE_SIZE};

  auto get_tracks() {
    return std::forward_as_tuple(lidar1, camera1, camera2);
  }

  TimeSpan frame(){

  }
};
*/