#pragma once

#pragma once
#include "async_condition_variable.hpp"
#include "config.hpp"
#include <boost/noncopyable.hpp>
#include <queue>

template<class ValueType>
struct shared_state_base
    : std::enable_shared_from_this<shared_state_base<ValueType>>
    , boost::noncopyable
{
    // distributor interface
    virtual auto
    stop() -> void
        = 0;

    virtual auto
    get_executor() -> net::any_io_executor = 0;

    virtual void
    notify_value(ValueType value)
        = 0;

    // consumer interface

    auto
    consume() -> net::awaitable<ValueType>
    {
        assert(get_executor() == co_await net::this_coro::executor);
        co_return co_await on_consume();
    }

    virtual ~shared_state_base() = default;

private:
    virtual auto
    on_consume() -> net::awaitable<ValueType> = 0;
};

template<class ValueType, class ExecutorType = net::any_io_executor>
struct basic_shared_state : shared_state_base<ValueType>
{
    using executor_type = ExecutorType;

    explicit basic_shared_state(executor_type exec)
        : exec_(std::move(exec))
        , condition_(exec_)
    {}

private:
    // distribution interface
    auto
    get_executor() -> net::any_io_executor override
    {
        return exec_;
    }

    void
    notify_value(ValueType value) override
    {
        // assert(get_executor() == co_await net::this_coro::executor);
        add_value(std::move(value));
        condition_.notify_one();
    }

    void
    stop() override
    {
        //assert(get_executor() == co_await net::this_coro::executor);
        condition_.stop();
    }

    // consumer interface
    auto
    on_consume() -> net::awaitable<ValueType> override
    {
        // keep alive during the consume operation
        auto self = this->shared_from_this();

        auto op = [this]() -> net::awaitable<ValueType> {
            co_await condition_.wait([this] { return data_available(); });
            co_return consume_data();
        };

        auto value = co_await net::this_coro::executor == get_executor()
                       ? co_await op()
                       : co_await net::co_spawn(get_executor(), op(),
                                                net::use_awaitable);

        self.reset();

        co_return value;
    }

private:
    virtual void
    add_value(ValueType snap)
        = 0;

    virtual bool
    data_available() const = 0;

    virtual ValueType
    consume_data()
        = 0;

private:
    executor_type exec_;
    async_condition_variable condition_{get_executor()};
};

template<class ValueType, class Executor = net::any_io_executor>
struct fifo_model : basic_shared_state<ValueType, Executor>
{
    fifo_model(Executor const &exec, std::size_t limit = 1024)
        : basic_shared_state<ValueType, Executor>(exec)
        , limit_(limit)
    {}

private:
    std::queue<ValueType, std::deque<ValueType>> queue_;
    std::size_t limit_;

private:
    void
    add_value(ValueType value) override
    {
        while (queue_.size() >= limit_) queue_.pop();
        queue_.push(std::move(value));
    }

    bool
    data_available() const override
    {
        return not queue_.empty();
    }

    ValueType
    consume_data() override
    {
        assert(not queue_.empty());
        auto result = std::move(queue_.front());
        queue_.pop();
        return result;
    }
};

template<class ValueType>
struct basic_distributor_impl;

/// Build a connection which returns the latest only snapshot

struct fifo
{
    fifo(std::size_t limit = 1024)
        : limit_(limit)
    {}

    template<class ValueType, class ExecutorType = net::any_io_executor>
    auto
    operator()(basic_distributor_impl<ValueType> const &,
               ExecutorType const &exec) const
    {
        // select the type of shared state
        using state_model = fifo_model<ValueType, ExecutorType>;
        return std::make_shared<state_model>(exec, limit_);
    }

private:
    std::size_t limit_;
};
