//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include "connection_impl.hpp"
#include <iostream>

namespace websocket
{
net::awaitable< void >
connection_impl::connect(std::string url, connect_options opts)
try
{
    assert(state_ == state_initial);
    assert(get_executor() == co_await net::this_coro::executor);
    std::cout << "url=" << url << std::endl;
    co_await ws_.connect(get_executor(), std::move(url), std::move(opts));
    transition(state_running);
}
catch (...)
{
    std::cout << errno << std::endl;
    transition(state_stopped);
}

net::awaitable< void >
connection_impl::shutdown()
{
    assert(get_executor() == co_await net::this_coro::executor);
    switch (state_)
    {
    case state_running:
        co_await ws_.drop();
        co_await state_condition_.wait(
            [this] { return state_ == state_stopped; });

        [[fallthrough]];

    case state_stopped:
    case state_initial:
        break;
    }
}

net::awaitable< void >
connection_impl::send(std::string frame, bool is_text)
{
    // This function implements a send queue, using the order in which
    // coroutines arrive at the send_condition_ condition variable as the queue
    // ordering mechanism.
    assert(get_executor() == co_await net::this_coro::executor);
    switch (state_)
    {
    case state_running:
        co_await send_condition_.wait(
            [this] { return state_ != state_running || !sending_; });
        if (state_ == state_running)
        {
            assert(!sending_);
            sending_ = true;
            if (is_text)
                ws_.text();
            else
                ws_.binary();

            error_code ec;
            co_await ws_.send(net::buffer(frame),
                              net::redirect_error(net::use_awaitable, ec));
            sending_ = false;
            send_condition_.notify_one();
        }

        [[fallthrough]];

    case state_stopped:
    case state_initial:
        break;
    }
}

void
connection_impl::transition(state_type newstate)
{
    std::cout << "connection_impl::transition " << newstate << std::endl;
    state_ = newstate;
    state_condition_.notify_all();
    send_condition_.notify_all();
}

net::awaitable< event >
connection_impl::consume()
{
    auto dynbuf    = std::make_shared< beast::flat_buffer >();
    auto [ec, len] = co_await ws_.read(*dynbuf);
    if (ec)
    {
        transition(state_stopped);
        co_return event(ec);
    }
    else
        co_return event(message(dynbuf, ws_.is_binary()));
}

}   // namespace websocket