#include "rpc/server.h"
#include <string>
using std::string;

int main() {
  rpc::server srv(8080);

  srv.bind("echo", [](string const& s) {
    return string("> ") + s;
  });

  srv.run();
  return 0;
}