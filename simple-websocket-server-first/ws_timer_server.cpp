#include <simple-websocket-server/server_ws.hpp>
#include "timer.hpp"

using namespace std;

using WsServer = SimpleWeb::SocketServer<SimpleWeb::WS>;


int main() {

  std::shared_ptr<asio::io_context> io_context{new asio::io_context{1}};

  asio::signal_set signals(*io_context, SIGINT, SIGTERM);
  signals.async_wait([&](const std::error_code&, int signal){
    cout << "end signal=" << signal << endl;
    io_context->stop();
  });

  WsServer server;
  server.config.port = 8080;
  server.io_service = io_context;


  auto &echo = server.endpoint["^/echo/?$"];

  echo.on_message = [](shared_ptr<WsServer::Connection> connection, shared_ptr<WsServer::InMessage> in_message) {
    auto out_message = in_message->string();

    cout << "Server: Message received: \"" << out_message << "\" from " << connection.get() << endl;

    cout << "Server: Sending message \"" << out_message << "\" to " << connection.get() << endl;

    // connection->send is an asynchronous function
    connection->send(out_message, [](const SimpleWeb::error_code &ec) {
      if(ec) {
        cout << "Server: Error sending message. " <<
             // See http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html, Error Codes for error code meanings
             "Error: " << ec << ", error message: " << ec.message() << endl;
      }
    });

    // Alternatively use streams:
    // auto out_message = make_shared<WsServer::OutMessage>();
    // *out_message << in_message->string();
    // connection->send(out_message);
  };

  echo.on_open = [](shared_ptr<WsServer::Connection> connection) {
    cout << "Server: Opened connection " << connection.get() << endl;
  };

  // See RFC 6455 7.4.1. for status codes
  echo.on_close = [](shared_ptr<WsServer::Connection> connection, int status, const string & /*reason*/) {
    cout << "Server: Closed connection " << connection.get() << " with status code " << status << endl;
  };

  long cnt = 0;
  server.start([&io_context, &server, &cnt](unsigned short port){
    cout << "Server listening on port " << port << endl << endl;
    TimerContext* t = setInterval(*io_context, [&cnt, &server](){
      cout << cnt++ << endl;
      for(auto &a_connection : server.get_connections()) {
        a_connection->send(std::to_string(cnt));
      }
    }, 1000);
  });

  io_context->run();
  return 0;
}
