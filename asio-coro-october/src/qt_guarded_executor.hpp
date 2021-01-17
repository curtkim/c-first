#pragma once
#include "config.hpp"
#include "qt_execution_context.hpp"

struct qt_guarded_executor
{
    qt_guarded_executor(std::weak_ptr<void> guard,
                        qt_execution_context &context
                        = qt_execution_context::singleton()) noexcept
        : context_(std::addressof(context))
        , guard_(std::move(guard))
    {}

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

    template<typename OtherAllocator>
    static constexpr auto
    query(net::execution::allocator_t<OtherAllocator>) noexcept
    {
        return std::allocator<void>();
    }

    static constexpr auto query(net::execution::allocator_t<void>) noexcept
    {
        return std::allocator<void>();
    }

    template<class F>
    void
    execute(F f) const
    {
        if (auto lock1 = guard_.lock())
        {
            context_->post([guard = guard_, f = std::move(f)]() mutable {
                if (auto lock2 = guard.lock())
                    f();
            });
        }
    }

    bool
    operator==(qt_guarded_executor const &other) const noexcept
    {
        return context_ == other.context_ && !guard_.owner_before(other.guard_)
            && !other.guard_.owner_before(guard_);
    }

    bool
    operator!=(qt_guarded_executor const &other) const noexcept
    {
        return !(*this == other);
    }

private:
    qt_execution_context *context_;
    std::weak_ptr<void> guard_;
};