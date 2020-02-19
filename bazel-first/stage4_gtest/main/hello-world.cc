#include "lib/hello-time.h"
#include "lib/hello-math.h"
#include "main/hello-greet.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string who = "world";
  if (argc > 1) {
    who = argv[1];
  }
  std::cout << get_greet(who) << std::endl;
  print_localtime();

  std::cout << add(1,2) << std::endl;
  
  return 0;
}
