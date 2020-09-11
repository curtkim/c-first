// from https://stackoverflow.com/questions/11833070/when-is-the-use-of-stdref-necessary
#include <thread>
#include <iostream>

void update(int& data)  //expects a reference to int
{
  std::cout << &data << std::endl;
  data = 15;
}

int main()
{
  int data = 10;
  std::cout << &data << std::endl;

  // This doesn't compile as the data value is copied when its reference is expected.
  //std::thread t0(update, data);

  std::thread t1(update, std::ref(data));  // works
  t1.join();

  std::cout << data << std::endl;
  return 0;
}