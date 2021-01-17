#pragma once
#include "config.hpp"
#include "qt_execution_context.hpp"

struct qt_executor
{
    qt_executor(qt_execution_context &context = qt_execution_context::singleton()) noexcept
        : context_(std::addressof(context))
    {
    }

    qt_execution_context &query(net::execution::context_t) const noexcept
    {
        return *context_;
    }

    static constexpr net::execution::blocking_t
    query(net::execution::blocking_t) noexcept
    {
        return net::execution::blocking.never;
    }

    static constexpr net::execution::relationship_t
    query(net::execution::relationship_t) noexcept
    {
        return net::execution::relationship.fork;
    }

    static constexpr net::execution::outstanding_work_t
    query(net::execution::outstanding_work_t) noexcept
    {
        return net::execution::outstanding_work.tracked;
    }

    template < typename OtherAllocator >
    static constexpr auto query(
    net::execution::allocator_t< OtherAllocator >) noexcept
    {
        return std::allocator<void>();
    }

    static constexpr auto query(net::execution::allocator_t< void >) noexcept
    {
        return std::allocator<void>();
    }

    template<class F>
    void execute(F f) const
    {
        context_->post(std::move(f));
    }

    bool operator==(qt_executor const &other) const noexcept
    {
        return context_ == other.context_;
    }

    bool operator!=(qt_executor const &other) const noexcept
    {
        return !(*this == other);
    }

private:
    qt_execution_context *context_;
};


static_assert(net::execution::is_executor_v<qt_executor>);
