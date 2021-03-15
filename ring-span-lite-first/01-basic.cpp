#include "common.h"

int main()
{
  double arr[] = { 2.0 , 3.0, 5.0, };

  nonstd::ring_span<double> buffer( arr, arr + dim(arr), arr, dim(arr) );
  std::cout << buffer << "\n";
  std::cout << "sizeof(nonstd::ring_span<double>) = " << sizeof(buffer) << "\n";

  std::cout << "by index" << "\n";
  std::cout << buffer[0] << " " << buffer[1] << " " << buffer[2] << "\n";

  nonstd::ring_span<double> buffer2( arr, arr + dim(arr), arr+1, dim(arr) );
  std::cout << buffer2 << "\n";

}