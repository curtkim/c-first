//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include "connect_options.hpp"

namespace websocket
{
ssl::context &
default_ssl_context()
{
    struct X
    {
        ssl::context ctx;

        X()
        : ctx(ssl::context::tls_client)
        {
        }

    };
    static X impl;
    return impl.ctx;
}
}   // namespace websocket