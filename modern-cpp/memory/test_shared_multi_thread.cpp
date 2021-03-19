#include <memory>
#include <iostream>
#include <thread>

using namespace std;

shared_ptr<int> g_ptr;


void thread1() {
  while(true){
    shared_ptr<int> temp;
    temp = g_ptr;
    if( 1 != *temp)
      cout << "error\n";
  }
}

int main() {
  g_ptr = make_shared<int>(1);
  thread th{thread1};

  while(true){
    auto t_ptr = make_shared<int>(1);
    g_ptr = t_ptr;
    auto ptr2 = make_shared<int>(2);
  }
}