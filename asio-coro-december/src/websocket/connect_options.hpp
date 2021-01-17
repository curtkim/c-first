//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef DECEMBER_2020_CONNECT_OPTIONS_HPP
#define DECEMBER_2020_CONNECT_OPTIONS_HPP

#include "async/stop_source.hpp"
#include "config.hpp"

namespace websocket
{
ssl::context &
default_ssl_context();

struct connect_options
{
    beast::http::fields       headers;
    std::chrono::milliseconds pingpong_timeout = std::chrono::seconds(30);

    ssl::context &       sslctx = default_ssl_context();
    async::stop_token    stop;
};

}   // namespace websocket

#endif   // DECEMBER_2020_CONNECT_OPTIONS_HPP
