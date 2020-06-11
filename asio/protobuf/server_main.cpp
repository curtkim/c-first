#include "db_server.hpp"
#include <asio.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, const char *argv[]) {
  unsigned port = 4050;
  if (argc > 1)
    port = atoi(argv[1]);
  cout << "Serving on port " << port << endl;

  try {
    asio::io_context io_context;
    DbServer server(io_context, 4050);
    io_context.run();
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}