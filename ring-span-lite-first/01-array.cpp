#include "common.h"
#include <assert.h>
#include <algorithm>
#include <stdio.h>

static_assert(sizeof(nonstd::ring_span<double>) == 40);
static_assert(sizeof(nonstd::ring_span<int>) == 40);


// 생성자(begin, end, first, size)
int main()
{
  // array
  double arr[] = { 2.0 , 3.0, 5.0, };

  const nonstd::ring_span<double> buffer( arr, arr + dim(arr), arr, dim(arr) );

  assert(buffer.size() == 3);
  assert(buffer.capacity() == 3);
  assert(buffer[0] == 2.0);  // by index
  assert(buffer[1] == 3.0);
  assert(buffer[2] == 5.0);

  auto it = std::find_if(buffer.begin(), buffer.end(), [](auto i){ return i > 4.0;});
  assert(*it == 5.0);

  // for loop
  for(const auto& item : buffer){
    std::cout << item << ", ";
  }
  std::cout << "\n";


  // 2. end를 넘어가는 경우
  nonstd::ring_span<double> buffer2( arr, arr + dim(arr), arr+1, dim(arr) );
  std::cout << buffer2 << "\n";

  // 3.
  nonstd::ring_span<double> buffer3( arr, arr + dim(arr), arr+2, 2 );
  std::cout << buffer3 << "\n";
  assert(buffer3.size() == 2);
  assert(buffer3.capacity() == 3);

  printf("%lu %lu", buffer3.begin(), arr);

}