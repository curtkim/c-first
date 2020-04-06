#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

struct my_type
{
  int i;

  template<typename OStream>
  friend OStream &operator<<(OStream &os, const my_type &c)
  {
    return os << "[my_type i=" << c.i << "]";
  }
};

void user_defined_example()
{
  auto my_value = my_type{14};
  spdlog::info("Some info message with arg: {}", my_value);
}

int main(){
  user_defined_example();
}