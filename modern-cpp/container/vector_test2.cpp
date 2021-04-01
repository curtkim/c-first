#include <vector>
#include <tuple>

struct Header{
  int seq;
};
struct Data {
  int value;
};

int main(){
  std::vector<std::tuple<Header, Data>> list(10);

  list.em
}