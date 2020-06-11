#pragma once

#include <asio.hpp>
#include <memory>

// Database server. The constructor starts it listening on the given
// port with the given io_service.
class DbServer {
public:
  DbServer(asio::io_context &io_context, unsigned port);
  ~DbServer();

private:
  DbServer();

  struct DbServerImpl;
  std::shared_ptr<DbServerImpl> d;
};