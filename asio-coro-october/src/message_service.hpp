//
// Created by rhodges on 09/11/2020.
//

#pragma once
#include "async_condition_variable.hpp"
#include "basic_distributor.hpp"
#include "config.hpp"
#include <memory>

struct message_service_impl
{
    using executor_type = net::strand<net::io_context::executor_type>;

    message_service_impl(net::io_context::executor_type const &exec);

    net::awaitable<void> run();

    auto connect(net::any_io_executor client_exec)
        -> net::awaitable<basic_connection<std::string>>;

    void stop();

    auto get_executor() const -> executor_type const &
    {
        return exec_;
    }

private:
    void listen_for_stop(std::function<void()> slot);

    executor_type exec_;
    error_code stop_reason_;
    std::vector<std::function<void()>> stop_signals_;
    async_condition_variable stop_condition_{get_executor()};

    basic_distributor_impl<std::string> message_dist_;
};

struct message_service
{
    using executor_type = net::io_context::executor_type;

    message_service(executor_type const &exec);

    message_service(message_service &&) noexcept = default;

    message_service & operator=(message_service &&) noexcept = default;

    message_service(message_service const &) = delete;

    message_service & operator=(message_service const &) = delete;

    void reset() noexcept;

    ~message_service();

    auto connect() -> net::awaitable<basic_connection<std::string>>;

    auto get_executor() const -> executor_type const &
    {
        return exec_;
    }

private:
    executor_type exec_;
    std::shared_ptr<message_service_impl> impl_;
};
