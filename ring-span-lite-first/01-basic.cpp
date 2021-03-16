#include "common.h"

// 생성자(begin, end, first, size)
int main()
{
  // array
  double arr[] = { 2.0 , 3.0, 5.0, };

  nonstd::ring_span<double> buffer( arr, arr + dim(arr), arr, dim(arr) );
  std::cout << buffer << "\n";
  std::cout << "sizeof(nonstd::ring_span<double>) = " << sizeof(buffer) << "\n";

  std::cout << "by index" << "\n";
  std::cout << buffer[0] << " " << buffer[1] << " " << buffer[2] << "\n";

  // end를 넘어가는 경우
  nonstd::ring_span<double> buffer2( arr, arr + dim(arr), arr+1, dim(arr) );
  std::cout << buffer2 << "\n";


  // vector

  std::vector<double> vec = {2.0, 3.0, 5.0};

  nonstd::ring_span<double> buffer3( vec.data(), vec.data() + vec.size(), vec.data(), vec.size() );
  std::cout << buffer3 << "\n";

  // end를 넘어, 일부분만
  nonstd::ring_span<double> buffer4( vec.data(), vec.data() + vec.size(), vec.data()+2, vec.size()-1 );
  std::cout << buffer4 << "\n";
}