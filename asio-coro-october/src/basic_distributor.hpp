#pragma once

#include "basic_connection.hpp"
#include "basic_shared_state.hpp"
#include "config.hpp"

template<class ValueType>
struct basic_distributor_handle
{
    using impl_class = shared_state_base<ValueType>;
    using implementation_type = std::shared_ptr<impl_class>;

    basic_distributor_handle(implementation_type impl)
        : impl_(std::move(impl))
    {}

    bool
    notify_value(ValueType const &v)
    {
        if (!impl_)
            return false;

        net::dispatch(impl_->get_executor(), [impl = impl_, v]() mutable {
            impl->notify_value(std::move(v));
        });
        return true;
    }

    bool
    stop()
    {
        if (!impl_)
            return false;

        impl_->stop();
        return true;
    }

    implementation_type
    release()
    {
        return std::move(impl_);
    }

private:
    implementation_type impl_;
};

template<class ValueType>
struct basic_distributor_impl
{
    using value_type = ValueType;

    using connection_type = basic_connection<ValueType>;

    basic_distributor_impl()
        : last_value_()
    {}

    basic_distributor_impl(value_type initial)
        : last_value_(std::move(initial))
    {}

    /// Connect to the distributor, return a connection which the client may
    /// wait on
    /// @not NOT THREAD SAFE
    /// @tparam Executor The executor type on which the client will want
    /// notifications to complete
    /// @tparam Builder A function which builds the implementation type,
    /// e.g. fifo, latest_only
    /// @param exec The executor on which the client wants to have
    /// notifications delivered
    /// @param builder The builder function to use. This will determine the
    /// handling of values waiting to be consumed
    /// @return the connection on which the client can wait for values
    template<class Executor, class Builder>
    auto
    connect(Executor &&exec, Builder &&builder) -> connection_type
    {
        // this will happen on the "current" executor. It is not thread-safe
        auto state = builder(*this, std::forward<Executor>(exec));
        auto handle = basic_distributor_handle<ValueType>(state);

        if (stop_error_)
            handle.stop();
        else
        {
            if (last_value_.has_value())
                handle.notify_value(*last_value_);
            connections_.push_back(handle.release());
        }

        return connection_type(std::move(state));
    }

    std::size_t
    count() const
    {
        auto total = std::size_t(0);
        for (auto &&weak : connections_)
            if (auto lock = weak.lock())
                ++total;
        return total;
    }

    auto
    last_value() const -> std::optional<ValueType> const &
    {
        return last_value_;
    }

    void
    stop()
    {
#ifndef NDEBUG
        assert(!distributing_);
#endif
        stop_error_ = net::error::operation_aborted;
        distribute();
    }

    // NOT REENTRANT
    void
    notify_value(ValueType value)
    {
#ifndef NDEBUG
        assert(!distributing_);
#endif
        last_value_ = std::move(value);
        distribute();
    }

private:
    // NOT REENTRANT
    void
    distribute()
    {
#ifndef NDEBUG
        assert(!distributing_);
        distributing_ = true;
#endif
        if (stop_error_)
            distribute_error();
        else
            distribute_value();

#ifndef NDEBUG
        assert(distributing_);
        distributing_ = false;
#endif
    }

    void
    distribute_value()
    {
#ifndef NDEBUG
        assert(distributing_);
#endif
        assert(last_value_.has_value());
        swap_buffer_.clear();
        for (auto &weak : connections_)
            if (auto handle = basic_distributor_handle(weak.lock());
                handle.notify_value(*last_value_))
                swap_buffer_.push_back(handle.release());
        connections_.swap(swap_buffer_);
    }

    void
    distribute_error()
    {
#ifndef NDEBUG
        assert(distributing_);
#endif
        for (auto &weak : connections_)
        {
            auto handle = basic_distributor_handle(weak.lock());
            handle.stop();
        }
        connections_.clear();
    }

    std::optional<value_type> last_value_{};

    error_code stop_error_{};

    std::vector<std::weak_ptr<shared_state_base<ValueType>>> connections_ = {},
                                                             swap_buffer_ = {};

#ifndef NDEBUG
    bool distributing_ = false;
#endif
};

template<class ValueType, class Executor = net::any_io_executor>
struct basic_distributor
{
    using executor_type = Executor;
    using impl_class = basic_distributor_impl<ValueType>;
    using connection_type = typename impl_class::connection_type;

    basic_distributor(executor_type const &e)
        : exec_(e)
        , impl_(new impl_class())
    {}

    basic_distributor(executor_type const &e, ValueType value)
        : exec_(e)
        , impl_(new impl_class(std::move(value)))
    {}

    basic_distributor(basic_distributor const &other) = delete;

    basic_distributor(basic_distributor &&other) noexcept
        : exec_(std::move(other.exec_))
        , impl_(std::exchange(other.impl_, nullptr))
    {}

    basic_distributor &
    operator=(basic_distributor const &other)
        = delete;

    basic_distributor &
    operator=(basic_distributor &&other) noexcept
    {
        auto tmp = std::move(other);
        std::swap(this->exec_, tmp.exec_);
        std::swap(this->impl_, tmp.impl_);
        return *this;
    }

    ~basic_distributor() { stop(); }

    auto
    get_executor() const -> executor_type const &
    {
        return exec_;
    }

    void
    notify_value(ValueType val)
    {
        assert(impl_);
        net::dispatch(get_executor(),
                      [impl = impl_, val = std::move(val)]() mutable {
                          impl->notify_value(std::move(val));
                      });
    }

    template<class Executor2, class Builder>
    auto
    connect(Executor2 &&exec, Builder &&builder)
        -> net::awaitable<connection_type>
    {
        assert(impl_);
        co_return co_await net::co_spawn(
            get_executor(),
            [impl = impl_, exec = std::forward<Executor2>(exec),
             builder = std::forward<Builder>(
                 builder)]() mutable -> net::awaitable<connection_type> {
                auto c = impl->connect(std::move(exec), std::move(builder));
                co_return c;
            },
            net::use_awaitable);
    }

    auto
    stop() -> void
    {
        if (impl_)
        {
            net::co_spawn(
                get_executor(),
                [impl
                 = std::exchange(impl_, nullptr)]() -> net::awaitable<void> {
                    assert(impl);
                    impl->stop();
                    delete impl;
                    co_return;
                },
                net::detached);
        }
    }

private:
    executor_type exec_;
    impl_class *impl_;
};
