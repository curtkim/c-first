//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef DECEMBER_2020_CONNECTION_HPP
#define DECEMBER_2020_CONNECTION_HPP

#include "websocket/connect_options.hpp"
#include "websocket/connection_impl.hpp"
#include "websocket/event.hpp"

namespace websocket
{
struct connection_lifetime
{
    connection_lifetime(std::shared_ptr< connection_impl > impl)
    : lifetime_(construct_lifetime(impl))
    , impl_(std::move(impl))
    {
    }

    std::shared_ptr< connection_impl > const &
    get_impl() const
    {
        return impl_;
    }

  private:
    static std::shared_ptr< void >
    construct_lifetime(std::shared_ptr< connection_impl > const &impl)
    {
        static int useful_address;

        auto deleter = [impl](void *) {
            net::co_spawn(
                impl->get_executor(),
                [impl]() -> net::awaitable< void > {
                    co_await impl->shutdown();
                },
                net::detached);
        };

        return std::shared_ptr< void >(&useful_address, deleter);
    }

    std::shared_ptr< void >            lifetime_;
    std::shared_ptr< connection_impl > impl_;
};

struct connection
{
    connection(connection_lifetime life)
    : life_(std::move(life))
    {
    }

    void
    send(std::string_view msg, bool is_text = true);

    net::awaitable< event >
    consume();

    beast::websocket::close_reason
    reason() const;

  private:
    connection_lifetime life_;
};

net::awaitable< connection >
connect(std::string url, connect_options options = {});
}   // namespace websocket

#endif   // DECEMBER_2020_CONNECTION_HPP
