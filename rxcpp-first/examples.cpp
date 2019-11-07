#include "rx.hpp"

namespace Rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
}

using namespace Rx;

using namespace std;


#include <fstream>

void read_file(){

  // create
  auto lines = rxcpp::observable<>::create<std::string>(
    [](rxcpp::subscriber<std::string> s){
      std::ifstream file("Makefile");
      std::string line;
      while (getline(file, line)) {
        s.on_next(line);
      }
      s.on_completed();
      file.close();
    });



  lines.
  subscribe(
    [](std::string str){printf("OnNext: %s\n", str.c_str());},
    [](){printf("OnCompleted\n");}
    );
}

int main()
{
  read_file();
  return 0;
}
