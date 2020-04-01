#include <iostream>

using namespace std;

class BigOne{

public:
  BigOne(int t1): a(t1){
    cout << "BigOne constructor" << endl;
  }
  ~BigOne(){
    cout << "BigOne deconstructor" << endl;
  }

  string to_string() {
    return "a="+std::to_string(a);
  }

private:
  int a;
};

BigOne make(int a){
  BigOne bigOne(a);
  return bigOne;
}

int main() {
  BigOne x = make(1);
  // BigOne이 삭제되고 다시 생성되는 건가?

  cout << x.to_string() << endl;
}
