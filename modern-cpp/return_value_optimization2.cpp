#include <iostream>
#include <vector>

using namespace std;


struct Snitch {   // Note: All methods have side effects
  Snitch() { cout << "c'tor" << endl; }
  ~Snitch() { cout << "d'tor" << endl; }

  Snitch(const Snitch&) { cout << "copy c'tor" << endl; }
  Snitch(Snitch&&) { cout << "move c'tor" << endl; }

  Snitch& operator=(const Snitch&) {
    cout << "copy assignment" << endl;
    return *this;
  }

  Snitch& operator=(Snitch&&) {
    cout << "move assignment" << endl;
    return *this;
  }
};


Snitch ExampleRVO() {
  return Snitch();
}

Snitch Named_ExampleNRVO() {
  Snitch snitch; //  object with a name
  return snitch;
}

vector<Snitch> makeVector() {
  vector<Snitch> list;
  list.emplace_back();
  list.emplace_back();
  list.emplace_back();
  std::cout << "makeVector end\n";
  return list;
}

int main() {
  //Snitch snitch = ExampleRVO();
  //Snitch snitch = Named_ExampleNRVO();

  auto vector = makeVector();
  std::cout << vector.size() << "\n";

  std::cout << "in main\n";
}