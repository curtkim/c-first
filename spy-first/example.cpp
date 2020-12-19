#include <spy/spy.hpp>
#include <iostream>

void f()
{
  if constexpr( spy::operating_system == spy::windows_ )
  {
    std::cout << "This code is Windows only.\n";
  }

  if constexpr( spy::compiler == spy::gcc_ )
  {
    std::cout << "This code has been compiled with g++.\n";
  }

  if constexpr( spy::compiler == spy::amd64_ )
  {
    std::cout << "This code has been compiled on AMD64 architecture.\n";
  }

  if constexpr( spy::stdlib == spy::libcpp_ )
  {
    std::cout << "This code uses libcpp as its standard library implementation.\n";
  }
}
void simd() {
  if constexpr( spy::simd_instruction_set == spy::avx_ )
  {
    std::cout << "This code has been compiled with AVX instructions set.\n";
  }

  if constexpr( spy::simd_instruction_set >= spy::sse41_ )
  {
    std::cout << "This code has been compiled with at least support for SSE 4.1\n";
  }

  if constexpr( spy::simd_instruction_set <= spy::sse2_ )
  {
    std::cout << "This code has been compiled with support for SSE2 at most.\n";
  }

  if constexpr (spy::simd_instruction_set <= spy::avx512_){
    std::cout << "avx512_\n";
  }
}

int main() {
  simd();
  f();
}