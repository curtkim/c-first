#include "db_server.hpp"
#include "packedmessage.hpp"
#include "stringdb.pb.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <asio.hpp>
#include <memory>
#include <functional>

using namespace std;

#define DEBUG true


// Database connection - handles a connection with a single client.
// Create only through the DbConnection::create factory.
//
class DbConnection : public std::enable_shared_from_this<DbConnection>
{
public:
  //typedef std::shared_ptr<DbConnection> Pointer;
  typedef std::shared_ptr<stringdb::Request> RequestPointer;
  typedef std::shared_ptr<stringdb::Response> ResponsePointer;

  DbConnection(tcp::socket socket, StringDatabase& db)
    : m_socket(std::move(socket)),
      m_db_ref(db),
      m_packed_request(std::shared_ptr<stringdb::Request>(new stringdb::Request()))
  {
  }

  void start()
  {
    start_read_header();
  }

private:
  tcp::socket m_socket;
  StringDatabase& m_db_ref;
  vector<uint8_t> m_readbuf;
  PackedMessage<stringdb::Request> m_packed_request;

  void handle_read_header(const std::error_code& error)
  {
    DEBUG && (cerr << "handle read " << error.message() << '\n');
    if (!error) {
      DEBUG && (cerr << "Got header!\n");
      DEBUG && (cerr << show_hex(m_readbuf) << endl);
      unsigned msg_len = m_packed_request.decode_header(m_readbuf);
      DEBUG && (cerr << msg_len << " bytes\n");
      start_read_body(msg_len);
    }
  }

  void handle_read_body(const std::error_code& error)
  {
    DEBUG && (cerr << "handle body " << error << '\n');
    if (!error) {
      DEBUG && (cerr << "Got body!\n");
      DEBUG && (cerr << show_hex(m_readbuf) << endl);
      handle_request();
      start_read_header();
    }
  }

  // Called when enough data was read into m_readbuf for a complete request
  // message.
  // Parse the request, execute it and send back a response.
  //
  void handle_request()
  {
    if (m_packed_request.unpack(m_readbuf)) {
      RequestPointer req = m_packed_request.get_msg();
      ResponsePointer resp = prepare_response(req);

      vector<uint8_t> writebuf;
      PackedMessage<stringdb::Response> resp_msg(resp);
      resp_msg.pack(writebuf);
      asio::write(m_socket, asio::buffer(writebuf));
    }
  }

  void start_read_header()
  {
    m_readbuf.resize(HEADER_SIZE);
    asio::async_read(m_socket, asio::buffer(m_readbuf),
                     std::bind(&DbConnection::handle_read_header, shared_from_this(),
                               std::placeholders::_1));
  }

  void start_read_body(unsigned msg_len)
  {
    // m_readbuf already contains the header in its first HEADER_SIZE
    // bytes. Expand it to fit in the body as well, and start async
    // read into the body.
    //
    m_readbuf.resize(HEADER_SIZE + msg_len);
    asio::mutable_buffers_1 buf = asio::buffer(&m_readbuf[HEADER_SIZE], msg_len);
    asio::async_read(m_socket, buf,
                     std::bind(&DbConnection::handle_read_body, shared_from_this(),
                               std::placeholders::_1));
  }

  ResponsePointer prepare_response(RequestPointer req)
  {
    string value;
    switch (req->type())
    {
      case stringdb::Request::GET_VALUE:
      {
        StringDatabase::iterator i = m_db_ref.find(req->request_get_value().key());
        value = i == m_db_ref.end() ? "" : i->second;
        break;
      }
      case stringdb::Request::SET_VALUE:
        value = req->request_set_value().value();
        m_db_ref[req->request_set_value().key()] = value;
        break;
      case stringdb::Request::COUNT_VALUES:
      {
        stringstream sstr;
        sstr << m_db_ref.size();
        value = sstr.str();
        break;
      }
      default:
        assert(0 && "Whoops, bad request!");
        break;
    }
    ResponsePointer resp(new stringdb::Response);
    resp->set_value(value);
    return resp;
  }
};


DbServer::DbServer(asio::io_context& io_context, unsigned port)
  : acceptor(io_context, tcp::endpoint(tcp::v4(), port))
{
  start_accept();
}

void DbServer::start_accept() {
  acceptor.async_accept([this](std::error_code ec, tcp::socket socket) {
    if (!ec) {
      std::make_shared<DbConnection>(std::move(socket), db)->start();
    }
    start_accept();
  });
}
