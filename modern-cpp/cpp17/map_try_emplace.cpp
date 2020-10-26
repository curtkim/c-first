#include <map>
#include <string>
#include <iostream>

int main() {
  std::map<char, std::string> m;
  m['H'] = "World";

  std::string s = "C++";
  m.emplace(std::make_pair('H', std::move(s)));
  // 이미 존재해서 실패함. 하지만 s는 move됨.

  // what happens with the string 's'?
  std::cout << s << '\n';
  std::cout << m['H'] << '\n';

  s = "C++";
  m.try_emplace('H', std::move(s));
  std::cout << s << '\n';
  std::cout << m['H'] << '\n';
}