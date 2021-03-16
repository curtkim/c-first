#pragma once

#include "circular_queue.hpp"
#include "nonstd/ring_span.hpp"

constexpr int QUEUE_SIZE = 10;

struct TimeSpan {
  nonstd::ring_span<long> lidar;
  nonstd::ring_span<int> camera1;
  nonstd::ring_span<int> camera2;

};

struct Timeline {
  CircularQueue<long> lidar1 = {QUEUE_SIZE};
  CircularQueue<int> camera1 = {QUEUE_SIZE};
  CircularQueue<int> camera2 = {QUEUE_SIZE};

  TimeSpan frame(){

  }

};