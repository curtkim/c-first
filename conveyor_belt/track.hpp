#pragma once

#include <tuple>
#include <vector>
#include <chrono>
#include "ring_span.hpp"

struct Header {
  long seq;
  std::chrono::system_clock::time_point time;
};

template <typename T>
class Track {
public:

  std::tuple<Header,T>* data;
  int max_size;
  int front = -1;
  int rear = -1;

  long seq = 0;

public:

  Track(int max_size) : max_size(max_size)
  {
    data = new std::tuple<Header,T>[max_size];
  }

  Track()=default;
  Track(const Track&) = delete;            // disable copying
  Track& operator=(const Track&) = delete; // disable assignment

  ~Track(){
    delete [] data;
  }

  inline bool is_empty() {
    //return front == rear;
    return front == -1;
  }

  inline bool is_full() {
    //return front == (rear+1)%max_size;
    return (front == 0 && rear == max_size-1) || (rear == (front-1)%(max_size-1));
  }

  void enqueue(const T item) {
    if (is_full()) {
      throw "queue is full";
    }
    else if (front == -1) /* Insert First Element */
    {
      front = rear = 0;
    }
    else if (rear == max_size-1 && front != 0)
    {
      rear = 0;
    }
    else
    {
      rear++;
    }

    std::get<0>(data[rear]) = Header{++seq, std::chrono::system_clock::now()};
    std::get<1>(data[rear]) = item;
  }

  const std::tuple<Header,T>& dequeue() {
    if (is_empty()) {
      throw "queue is empty";
    }
    else {
      //front = (front + 1) % max_size;
      //return data[front];

      int temp = front;
      if (front == rear)
      {
        front = -1;
        rear = -1;
      }
      else if (front == max_size-1)
        front = 0;
      else
        front++;
      return data[temp];
    }
  }

  ring_span<std::tuple<Header, T>> span() {
    return ring_span<std::tuple<Header, T>>(data, data+max_size, data+front, size());
  }
  ring_span<std::tuple<Header, T>> span(const ring_span<std::tuple<Header, T>>& prev_span) {
    return ring_span<std::tuple<Header, T>>(data, data+max_size, data+front+prev_span.size(), size() - prev_span.size());
  }


  const int size() {
    if (front == -1)
      return 0;

    //fmt::print("front={} rear={} rear - front= {}\n", front, rear, rear - front);
    int temp = rear - front + 1;
    return temp >=0 ? temp : max_size + temp;
  }


};
