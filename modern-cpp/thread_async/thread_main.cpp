#include <thread>
#include <iostream>

// https://redforce01.tistory.com/85?category=695802
int main() {

  // without parameter
  std::thread myThread(
    []() {
      for (int i = 0; i < 5; i++) {
        std::cout << std::this_thread::get_id() << " Hello~ I am myThread" << std::endl;
      }
    });

  // with parameter
  std::thread myThread2 = std::thread(
    [](int nParam) {
      for (int i = 0; i < 3; i++) {
        std::cout << std::this_thread::get_id() << " thread called : " << nParam << std::endl;
      }
    }, 4);


  if (myThread.joinable() == true)
    myThread.join();

  if (myThread2.joinable() == true)
    myThread2.join();

  return 0;
}