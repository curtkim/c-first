// from https://github.com/bfgroup/duck_invoke

#include <iostream>
#include "tag_invoke.h"

namespace compute {

  BFG_TAG_INVOKE_DEF(formula);

  template <typename Compute>
  float do_compute(const Compute & c, float a, float b)
  {
    return compute::formula(c, a, b);
  }

} // namespace compute


namespace myapp {
  struct custom_compute {
  private:
    friend float tag_invoke(compute::formula_t, const custom_compute &, float a, float b) {
      return a * b;
    }
  };
}

int main()
{
  std::cout << compute::do_compute(myapp::custom_compute{}, 2, 3) << std::endl;
  return 0;
}