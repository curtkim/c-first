#include <iostream>

#include "rpc/server.h"
#include "rpc/this_server.h"
#include "rpc/client.h"

int main() {
  rpc::server server(18080);
  server.bind("hello", [](){
    std::cout << "Hello from RPC!\n";
    rpc::this_server().stop();
  });

  server.async_run();

  rpc::client client("localhost", 18080);
  client.call("hello");

  return 0;
}
