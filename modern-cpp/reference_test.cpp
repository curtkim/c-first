#include <assert.h>

struct Data {
  long a;
  long b;
  long c;
  long d;
};

int main() {
  Data data1 = {1,1,1,1};
  Data data2 = {2,2,2,2};

  Data& ref = data1;
  assert(1 == ref.a);

  ref = data2;
  assert(2 == ref.a);
}