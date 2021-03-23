#pragma once

#include <tuple>
#include <vector>
#include <chrono>

struct Header {
  long seq;
  std::chrono::system_clock::time_point time;
};

template <typename T>
class Track {
public:

  std::vector<std::tuple<Header,T>> data;
  int max_size;
  int front = 0;
  int rear = 0;
  long seq = 0;

public:

  Track(int max_size) : max_size(max_size), data(max_size)
  {
  }

  Track()=default;
  Track(const Track&) = delete;            // disable copying
  Track& operator=(const Track&) = delete; // disable assignment

  inline bool is_empty() {
    return front == rear;
  }

  inline bool is_full() {
    return front == (rear+1)%max_size;
  }

  void enqueue(const T item) {
    if (is_full()) {
      throw "queue is full";
    }
    else {
      std::get<0>(data[rear]) = Header{seq++, std::chrono::system_clock::now()};
      std::get<1>(data[rear]) = item;
      rear = ++rear%max_size;
    }
  }

  const std::tuple<Header,T>& dequeue() {
    if (is_empty()) {
      throw "queue is empty";
    }
    else {
      return data[front++%max_size];
    }
  }
};
