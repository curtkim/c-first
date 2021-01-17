//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef DECEMBER_2020_CONNECTION_IMPL_HPP
#define DECEMBER_2020_CONNECTION_IMPL_HPP

#include "async/condition_variable_impl.hpp"
#include "websocket/connect_options.hpp"
#include "websocket/event.hpp"
#include "websocket/variant_websocket.hpp"

namespace websocket
{
struct connection_impl
{
    connection_impl(net::any_io_executor exec)
    : exec_(std::move(exec))
    {
    }

    net::awaitable< void >
    connect(std::string url, connect_options opts);

    net::awaitable< event >
    consume();

    net::awaitable< void >
    send(std::string frame, bool is_text);

    net::any_io_executor const &
    get_executor() const
    {
        return exec_;
    }

    /// Cause the websocket to drop and wait for the websocket's comms  to be
    /// shutdown
    net::awaitable< void >
    shutdown();

    beast::websocket::close_reason
    reason() const
    {
        assert(state_ == state_stopped);
        return ws_.reason();
    }

  private:
    enum state_type
    {
        state_initial,
        state_running,
        state_stopped,
    };

    void
    transition(state_type newstate);

  private:
    net::any_io_executor           exec_;
    variant_websocket              ws_;
    async::condition_variable_impl state_condition_ { get_executor() };
    state_type                     state_ = state_initial;
    async::condition_variable_impl send_condition_ { get_executor() };
    bool                           sending_ = false;
};

}   // namespace websocket

#endif   // DECEMBER_2020_CONNECTION_IMPL_HPP
