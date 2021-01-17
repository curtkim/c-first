#pragma once

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/variant2/variant.hpp>

namespace net      = boost::asio;
using error_code   = boost::system::error_code;
using system_error = boost::system::system_error;
namespace asio     = boost::asio;
namespace ssl      = asio::ssl;
namespace beast    = boost::beast;

using boost::variant2::monostate;
using boost::variant2::variant;
