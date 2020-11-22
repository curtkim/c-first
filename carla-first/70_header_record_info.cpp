#include <iostream>
#include <vector>
#include "70_header.hpp"

using namespace std;

int main() {
  cout << "sizeof(Header) " << sizeof(Header) << endl;
  cout << "sizeof(Record) " << sizeof(Record) << endl;
  cout << "sizeof(string) " << sizeof(std::string) << endl;
  cout << "sizeof(std::vector<char>) " << sizeof(std::vector<char>) << endl;

}