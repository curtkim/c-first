#pragma once

#include "config.hpp"
#include <boost/core/ignore_unused.hpp>
#include <condition_variable>
#include <set>

struct async_condition_variable
{
private:
    using timer_type = net::high_resolution_timer;

public:
    using clock_type = timer_type::clock_type;
    using duration = timer_type::duration;
    using time_point = timer_type::time_point;
    using executor_type = timer_type::executor_type;

    /// Constructor
    /// @param exec is the executor to associate with the internal timer.
    explicit inline async_condition_variable(net::any_io_executor exec);

    template<class Pred>
    [[nodiscard]]
    auto
    wait(Pred pred) -> net::awaitable<void>;

    template<class Pred>
    [[nodiscard]]
    auto
    wait_until(Pred pred, time_point limit) -> net::awaitable<std::cv_status>;

    template<class Pred>
    [[nodiscard]]
    auto
    wait_for(Pred pred, duration d) -> net::awaitable<std::cv_status>;

    auto
    get_executor() noexcept -> executor_type
    {
        return timer_.get_executor();
    }

    inline void
    notify_one();

    inline void
    notify_all();

    /// Put the condition into a stop state so that all future awaits fail.
    inline void
    stop();

    auto
    error() const -> error_code const &
    {
        return error_;
    }

    void
    reset()
    {
        error_ = {};
    }

private:
    timer_type timer_;
    error_code error_;
    std::multiset<timer_type::time_point> wait_times_;
};

template<class Pred>
auto
async_condition_variable::wait_until(Pred pred, time_point limit)
    -> net::awaitable<std::cv_status>
{
    assert(co_await net::this_coro::executor == timer_.get_executor());

    while (not error_ and not pred())
    {
        if (auto now = clock_type::now(); now >= limit)
            co_return std::cv_status::timeout;

        // insert our expiry time into the set and remember where it is
        auto where = wait_times_.insert(limit);

        // find the nearest expiry time and set the timeout for that one
        auto when = *wait_times_.begin();
        if (timer_.expiry() != when)
            timer_.expires_at(when);

        // wait for timeout or cancellation
        error_code ec;
        co_await timer_.async_wait(net::redirect_error(net::use_awaitable, ec));

        // remove our expiry time from the set
        wait_times_.erase(where);

        // any error other than operation_aborted is unexpected
        if (ec and ec != net::error::operation_aborted)
            if (not error_)
                error_ = ec;
    }

    if (error_)
        throw system_error(error_);

    co_return std::cv_status::no_timeout;
}

template<class Pred>
auto
async_condition_variable::wait(Pred pred) -> net::awaitable<void>
{
    auto stat = co_await wait_until(std::move(pred), time_point::max());
    boost::ignore_unused(stat);
    co_return;
}

template<class Pred>
auto
async_condition_variable::wait_for(Pred pred, duration d)
    -> net::awaitable<std::cv_status>
{
    return wait_until(std::move(pred), clock_type::now() + d);
}

async_condition_variable::async_condition_variable(net::any_io_executor exec)
    : timer_(std::move(exec))
    , error_()
{}

void
async_condition_variable::notify_one()
{
    timer_.cancel_one();
}

void
async_condition_variable::notify_all()
{
    timer_.cancel();
}

void
async_condition_variable::stop()
{
    error_ = net::error::operation_aborted;
    notify_all();
}
