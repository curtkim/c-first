//
// Created by rhodges on 09/11/2020.
//

#include "message_service.hpp"

message_service_impl::message_service_impl(
    const boost::asio::io_context::executor_type &exec)
    : exec_(net::make_strand(exec))
{}

net::awaitable<void>
message_service_impl::run()
{
    using namespace std::literals;

    auto timer = net::high_resolution_timer(co_await net::this_coro::executor);

    auto done = false;

    listen_for_stop([&] {
      done = true;
      timer.cancel();
    });

    while (!done)
    {
        for (int i = 0; i < 10 && !done; ++i)
        {
            timer.expires_after(1s);
            auto ec = boost::system::error_code();
            co_await timer.async_wait( net::redirect_error(net::use_awaitable, ec));
            if (ec)
                break;
            message_dist_.notify_value(std::to_string(i + 1) + " seconds");
        }

        for (int i = 10; i-- && !done;)
        {
            timer.expires_after(250ms);
            auto ec = boost::system::error_code();
            co_await timer.async_wait(net::redirect_error(net::use_awaitable, ec));
            if (ec)
                break;
            message_dist_.notify_value(std::to_string(i));
        }
    }
}

auto
message_service_impl::connect(net::any_io_executor client_exec)
    -> net::awaitable<basic_connection<std::string>>
{
    assert(co_await net::this_coro::executor == get_executor());
    co_return message_dist_.connect(client_exec, fifo());
}

void
message_service_impl::stop()
{
    if (!stop_reason_)
        stop_reason_ = net::error::operation_aborted;
    auto copy = std::exchange(stop_signals_, {});
    for (auto &slot : copy) slot();
}

void
message_service_impl::listen_for_stop(std::function<void()> slot)
{
    if (stop_reason_)
        slot();
    else
        stop_signals_.push_back(std::move(slot));
}

message_service::message_service(const message_service::executor_type &exec)
    : exec_(exec)
    , impl_(std::make_shared<message_service_impl>(exec_))
{
    net::co_spawn(
        impl_->get_executor(),
        [impl = impl_]() -> net::awaitable<void> { co_await impl->run(); },
        net::detached);
}

void message_service::reset() noexcept
{
    if (impl_)
        net::dispatch(impl_->get_executor(),
                      [impl = impl_]() { impl->stop(); });
}

message_service::~message_service() { reset(); }

auto message_service::connect() -> net::awaitable<basic_connection<std::string>>
{
    assert(impl_);

    // capture a copy of the impl pointer in the current coroutine
    auto impl = impl_;

    auto client_exec = co_await net::this_coro::executor;
    auto our_exec = impl->get_executor();

    if (client_exec != our_exec)
    {
        co_return co_await net::co_spawn(
            our_exec, [&] { return impl->connect(client_exec); },
            net::use_awaitable);
    }
    else
    {
        co_return co_await impl->connect(client_exec);
    }
}
