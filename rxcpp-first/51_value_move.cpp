#include <memory>
#include <rxcpp/rx.hpp>


namespace Rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
}
using namespace Rx;

using namespace std;
#include <chrono>

class Value {
private:
    long frame;

public:
    Value(long f): frame(f) {
      std::cout << f << " constructor" << std::endl;
    }

    ~Value(){
      std::cout << frame << " deconstructor" << std::endl;
    }

    Value(Value& v): frame(v.frame) {
      std::cout << v.frame << " copy constructor" << std::endl;
    }
    Value(Value&& v): frame(v.frame) {
      std::cout << v.frame << " move constructor" << std::endl;
    }
    
    long get() {
      return frame;
    }
};

int main() {
  auto lidar_period = std::chrono::milliseconds(1000);
  auto lidar1$ = rxcpp::sources::interval(lidar_period).map([](long i){ return std::make_shared<Value>(i);});

  lidar1$
    .take(10)
    .subscribe(
      []( std::shared_ptr<Value> v) { std:cout << v->get() << std::endl;},
      []() { printf("\nOnCompleted\n"); });

  return 0;
}
