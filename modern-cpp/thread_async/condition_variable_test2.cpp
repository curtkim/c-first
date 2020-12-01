
#include <iostream>           // std::cout
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

std::mutex mtx;
std::condition_variable cv;
bool first = false;
bool second = false;

void print_id (int id) {
  int i = 0;
  while(1) {
    {
      std::unique_lock<std::mutex> lck(mtx);
      cv.wait(lck, []() { return first; });
      std::cout << "thread first " << id << " " << i << '\n';
    }

    {
      std::unique_lock<std::mutex> lck(mtx);
      cv.wait(lck, []() { return second; });
      std::cout << "thread second " << id << " " << i << '\n';
    }
    i++;
  }
}

int main ()
{
  std::thread threads[3];
  // spawn 10 threads:
  for (int i=0; i<3; ++i)
    threads[i] = std::thread(print_id,i);
  std::cout << "3 threads ready to race...\n";

  char str[100];

  for(int i = 0;i < 10; i++){
    {
      std::unique_lock<std::mutex> lck(mtx);
      first = true;
      second = false;
      cv.notify_all();
    }

    std::cin.getline(str, 5);
    std::this_thread::sleep_for(std::chrono::microseconds(10));

    {
      std::unique_lock<std::mutex> lck(mtx);
      second = true;
      first = false;
      cv.notify_all();
    }
  }


  for (auto& th : threads)
    th.join();

  return 0;
}