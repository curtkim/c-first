#include <assert.h>
#include <functional>
#include <vector>

// lambda, function을 파라미터로 받는다.
template<typename Func>
auto bind_3(Func func)
{
  return [func](const int value){ return func(value, 3); };
}

// TODO Func의 형식을 T를 받아서 bool을 반환하는 함수로 제한하고 싶다.
template<typename T, typename Func>
int count(std::vector<T>& vec, Func predicate) {
  int result = 0;
  for(const auto i : vec){
    if( predicate(i) )
      result++;
  }
  return result;
}


int main()
{
  assert(4 == bind_3(std::plus<>{})(1));

  std::vector<int> list = {1,2,3,4,5};
  assert(2 == count(list, [](int i){ return i % 2 == 0;}));
}