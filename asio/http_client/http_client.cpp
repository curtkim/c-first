#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "AsyncHttpClient.hpp"

using namespace std;

void OnFinish(std::shared_ptr<AsyncHttpClient> client) {
  std::cout << "call back in finish, err: " << client->Err() << "\n";
}

int main() {
  std::map<string, std::shared_ptr<AsyncHttpClient>> urls = {
      {"http://www.qq.com", nullptr},
      {"http://en.cppreference.com/w/cpp/regex/match_results", nullptr},
  };

  asio::io_context io_context;
  for (auto &&it : urls) {
    it.second = std::make_shared<AsyncHttpClient>(io_context, it.first);
    it.second->Start(OnFinish);
  }

  io_context.run();

  return 0;
}
