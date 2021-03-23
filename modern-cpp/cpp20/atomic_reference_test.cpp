// https://mariusbancila.ro/blog/2020/04/21/cpp20-atomic_ref/
#include <atomic>
#include <thread>
#include <iostream>

struct Data {
  long a;
  long b;
  long c;
  long d;
  long e;
  long f;
  long g;
  long h;
  long i;
  long j;
};

const long TERMINATE = 99;


void doit(std::atomic_ref<Data> data){
  while(1){
    const Data& d = data.load();

    if( d.a == TERMINATE) break;
    
    std::cout << d.a << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
}


int main() {
  Data data1;
  Data data2;
  const Data terminal {TERMINATE,TERMINATE,TERMINATE,TERMINATE,TERMINATE,TERMINATE,TERMINATE,TERMINATE,TERMINATE,TERMINATE};

  bool isData1 = true;
  std::atomic_ref<Data> ref(data1);

  std::thread t(doit, ref);

  for(long i = 1;i < 10; i++){
    if( isData1){
      data2.a = i;
      ref.store(data2);
    }
    else{
      data1.a = i;
      ref.store(data1);
    }

    isData1 = !isData1; // toggle
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ref.store(terminal);
  t.join();
  
  return 0;
}