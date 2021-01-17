//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include "config.hpp"
#include "websocket/connection.hpp"

#include <cstdio>
#include <iostream>

net::awaitable< void > do_console(async::stop_token stop, websocket::connection ws)
try
{
    auto console = asio::posix::stream_descriptor(co_await net::this_coro::executor, ::dup(STDIN_FILENO));
    auto stopconn = stop.connect([&] { console.cancel(); });

    std::string console_chars;
    while (!stop.stopped())
    {
        auto line_len =
            co_await net::async_read_until(console,
                                           net::dynamic_buffer(console_chars),
                                           '\n',
                                           net::use_awaitable);
        auto line = console_chars.substr(0, line_len - 1);
        console_chars.erase(0, line_len);
        std::cout << "you typed this: " << line << std::endl;
        ws.send(line);
    }
}
catch (system_error &se)
{
    if (se.code() != net::error::operation_aborted)
    {
        std::cerr << "console error: " << se.what() << std::endl;
        std::exit(1);
    }
}
catch (std::exception &e)
{
    std::cerr << "console error: " << e.what() << std::endl;
    std::exit(1);
}

net::awaitable< void > chat() {
    auto ws = co_await websocket::connect("ws://localhost:8080/");

    auto stop_children = async::stop_source();
    net::co_spawn(
        co_await net::this_coro::executor,
        [stop = async::stop_token(stop_children), ws]() mutable {
            return do_console(std::move(stop), std::move(ws));
        },
        net::detached);

    for (;;)
    {
        auto event = co_await ws.consume();
        if (event.is_error())
        {
            if (event.error() == beast::websocket::error::closed)
                std::cerr << "peer closed connection: " << ws.reason()
                          << std::endl;
            else {
              if( event.error() == asio::error::operation_aborted)
                std::cout << "asio::error::operation_aborted \n";
              std::cerr << "connection error: " << event.error() << std::endl;
            }
            break;
        }
        else
        {
            std::cout << "message received: " << event.message() << std::endl;
        }
    }
}

int main() {
    net::io_context ioctx;
    net::co_spawn(ioctx.get_executor(), [] { return chat(); }, net::detached);
    ioctx.run();
}
