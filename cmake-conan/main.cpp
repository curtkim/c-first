#include <fmt/format.h>

#include <iostream>


int main(int argc, char** argv)
{
  // format
  std::cout << fmt::format("The answer is {}.", 42) << std::endl;
  
}

