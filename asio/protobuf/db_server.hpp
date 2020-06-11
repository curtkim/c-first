#pragma once

#include <asio.hpp>
#include <memory>
#include <string>
#include <map>

typedef std::map<std::string, std::string> StringDatabase;

using asio::ip::tcp;

class DbServer {
public:
  DbServer(asio::io_context &io_context, unsigned port);
  ~DbServer() {};

private:
  void start_accept();

  tcp::acceptor acceptor;
  StringDatabase db;
};