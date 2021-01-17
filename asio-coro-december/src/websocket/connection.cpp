//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "connection.hpp"

#include "websocket/connection_impl.hpp"

namespace websocket
{
net::awaitable< connection >
connect(std::string url, websocket::connect_options options)
{
    auto impl =
        std::make_shared< connection_impl >(co_await net::this_coro::executor);
    co_await impl->connect(std::move(url), std::move(options));

    co_return connection(connection_lifetime(impl));
}

void
connection::send(std::string_view msg, bool is_text)
{
    auto impl = life_.get_impl();

    net::co_spawn(
        impl->get_executor(),
        [impl, is_text, msg = std::string(msg.begin(), msg.end())]() mutable {
            return impl->send(std::move(msg), is_text);
        },
        net::detached);
}

beast::websocket::close_reason
connection::reason() const
{
    return life_.get_impl()->reason();
}

net::awaitable< event >
connection::consume()
{
    auto impl = life_.get_impl();
    assert(impl);
    if (co_await net::this_coro::executor == impl->get_executor())
        co_return co_await impl->consume();
    else
        co_return co_await net::co_spawn(
            impl->get_executor(),
            [impl] { return impl->consume(); },
            net::use_awaitable);
}

}   // namespace websocket