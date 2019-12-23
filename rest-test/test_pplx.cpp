#include "unistd.h"
#include "bits/stdc++.h"
#include "cpprest/asyncrt_utils.h"

using namespace std;
using namespace utility;

int main() {
    pplx::task_options(a);

    // This is one more way of submitting tasks to thread pool
    pplx::task_from_result()
        .then([]() {
            std::cout << "Entry3 with TID : " << std::this_thread::get_id() << std::endl;
            return pplx::task_from_result<std::string>("Hello"); // return as task
        })
        .then([](string x) // capture the return value from prevTask (as simple string)
              {
                  std::cout << "Entry4 with TID : " << std::this_thread::get_id() << std::endl;
                  std::vector<string> ret = {x, "World !"};
                  return ret;
              })
        .then([](pplx::task<std::vector<string>> prevTask) // capturing as task (works)
              {
                  std::vector<string> v_strings;
                  try {
                      v_strings = prevTask.get();
                  }
                  catch (const std::exception &e) {
                      std::cout << e.what() << std::endl;
                  }
                  for (auto &str:v_strings)
                      std::cout << str << " ";
                  std::cout << std::endl;
              }).wait(); // please wait main thread, for these chain to complete
}