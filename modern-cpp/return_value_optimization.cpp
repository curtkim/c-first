#include <iostream>

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

int main() {
  //Snitch snitch = ExampleRVO();

  Snitch snitch = Named_ExampleNRVO();
}