#include <iostream>
#include <thread>

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ConcurrentQueue {
public:
    T pop() {
      std::unique_lock<std::mutex> mlock(mutex_);
      while (queue_.empty()) {
        cond_.wait(mlock);
      }
      auto val = queue_.front();
      queue_.pop();
      mlock.unlock();
      cond_.notify_one();
      return val;
    }

    void pop(T& item) {
      std::unique_lock<std::mutex> mlock(mutex_);
      while (queue_.empty()) {
        cond_.wait(mlock);
      }
      item = queue_.front();
      queue_.pop();
      mlock.unlock();
      cond_.notify_one();
    }

    void push(const T& item) {
      std::unique_lock<std::mutex> mlock(mutex_);
      while (queue_.size() >= BUFFER_SIZE) {
        cond_.wait(mlock);
      }
      queue_.push(item);
      mlock.unlock();
      cond_.notify_one();
    }
    ConcurrentQueue()=default;
    ConcurrentQueue(const ConcurrentQueue&) = delete;            // disable copying
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete; // disable assignment

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    const static unsigned int BUFFER_SIZE = 10;
};

const int SIZE = 10000;

void produce(ConcurrentQueue<int>& q) {
  for (int i = 0; i< SIZE; ++i) {
    std::cout << "Pushing " << i << "\n";
    q.push(i);
  }
}

void consume(ConcurrentQueue<int>& q, unsigned int id) {
  for (int i = 0; i< SIZE/4; ++i) {
    auto item = q.pop();
    std::cout << "Consumer " << id << " popped " << item << "\n";
  }
}

int main() {
  ConcurrentQueue<int> q;

  // producer thread
  std::thread prod1([&q](){produce(q);});

  // consumer threads
  std::thread consumer1([&q](){consume(q, 1);});
  std::thread consumer2([&q](){consume(q, 2);});
  std::thread consumer3([&q](){consume(q, 3);});
  std::thread consumer4([&q](){consume(q, 4);});

  prod1.join();
  consumer1.join();
  consumer2.join();
  consumer3.join();
  consumer4.join();
  return 0;
}