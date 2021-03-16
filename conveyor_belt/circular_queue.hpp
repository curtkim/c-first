#pragma once

#include <mutex>
#include <condition_variable>

template <typename T>
class CircularQueue {
private:

  T* data;
  int max_size;
  int front = 0;
  int rear = 0;

  std::mutex mutex_;
  std::condition_variable cond_;

public:

  CircularQueue(int max_size) : max_size(max_size)
  {
    data = new T[max_size];
  }

  CircularQueue()=default;
  CircularQueue(const CircularQueue&) = delete;            // disable copying
  CircularQueue& operator=(const CircularQueue&) = delete; // disable assignment


  virtual ~CircularQueue() {
    delete[] data;
  }

  bool is_empty() {
    return front == rear;
  }

  bool is_full() {
    return front == (rear+1)%max_size;
  }

  void enqueue(T item) {
    if (is_full()) {
      throw "queue is full";
    }
    else {
      rear = ++rear%max_size;
      data[rear] = item;
    }
  }

  T dequeue() {
    if (is_empty()) {
      throw "queue is empty";
    }
    else {
      return data[front%max_size];
    }
  }

};
