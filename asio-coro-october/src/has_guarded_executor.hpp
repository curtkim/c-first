#pragma once
#include "qt_execution_context.hpp"
#include "qt_guarded_executor.hpp"

struct has_guarded_executor
{
    using executor_type = qt_guarded_executor;

    has_guarded_executor(qt_execution_context &ctx
                         = qt_execution_context::singleton())
        : context_(std::addressof(ctx))
    {
        new_guard();
    }

    void
    new_guard()
    {
        static int x = 0;
        guard_ = std::shared_ptr<int>(std::addressof(x),
                                      // no-op deleter
                                      [](auto *) {});
    }

    void
    reset_guard()
    {
        guard_.reset();
    }

    executor_type
    get_executor() const
    {
        return qt_guarded_executor(guard_, *context_);
    }

private:
    qt_execution_context *context_;
    std::shared_ptr<void> guard_;
};