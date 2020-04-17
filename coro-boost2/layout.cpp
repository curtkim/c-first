#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include <boost/coroutine2/all.hpp>

using namespace boost::coroutines2;


struct FinalEOL{
  ~FinalEOL(){
    std::cout << " destructors" << std::endl;
  }
};


int main(int argc,char* argv[]){
  using std::begin;
  using std::end;
  std::vector<std::string> words{
      "peas", "porridge", "hot", "peas",
      "porridge", "cold", "peas", "porridge",
      "in", "the", "pot", "nine",
      "days", "old" };

  int num=5, width=15;

  coroutine<std::string>::push_type writer(
      [&](coroutine<std::string>::pull_type& in){

        // finish the last line when we leave by whatever means
        FinalEOL eol;

        // pull values from upstream, lay them out 'num' to a line
        for (;;){
          for(int i=0;i<num;++i){
            // when we exhaust the input, stop
            if(!in) return;
            std::cout << std::setw(width) << in.get();
            // now that we've handled this item, advance to next
            in();
          }
          // after 'num' items, line break
          std::cout << std::endl;
        }
      });

  std::copy(begin(words), end(words), begin(writer));
  std::cout << "\nDone";

  return EXIT_SUCCESS;
}