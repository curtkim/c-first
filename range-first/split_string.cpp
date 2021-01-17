#include <string>
#include <iostream>
#include <range/v3/all.hpp> // get everything

int main() {
  std::string text = "Let me split this into words";
  auto splitText = text | ranges::view::split(' ') | ranges::to<std::vector<std::string>>();

  for(const auto & s : splitText){
    std::cout << s << std::endl;
  }
}