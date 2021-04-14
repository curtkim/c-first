// https://www.linkedin.com/pulse/c-return-value-optimization-dipanjan-das-roy
// https://www.youtube.com/watch?v=80TXwV_sdCY
#include <iostream>
#include <vector>
#include <assert.h>

using namespace std;

class BigObject{
public:
  string name;

  ~BigObject(){
    cout << "Destroyed " << name << "\n";
  }
};

BigObject* addr = nullptr;

BigObject getBigObject() {
  BigObject obj{"hello"};
  addr = &obj;
  return obj;
}


vector<int>* addr2 = nullptr;

vector<int> getVector() {
  vector<int> v{1,2,3,4,5,6,7,8,9,0};
  addr2 = &v;
  return v;
}

int main() {
  // Copy elision
  auto obj = getBigObject();
  assert(addr == &obj);

  // Implicit move
  auto vector = getVector();
  assert(addr2 == &vector);

  return 0;
}