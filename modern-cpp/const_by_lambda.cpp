#include <vector>
#include <assert.h>

int main() {
  const auto data = [](){
    std::vector<int> result;
    result.emplace_back(1);
    result.emplace_back(2);
    return result;
  }();

  assert(1 == data[0]);
  assert(2 == data[1]);
}