#include <string>

namespace mystring {
std::string substring(std::string a, int pos, int len){
  return a.substr(pos, len);
}
}
