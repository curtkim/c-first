#pragma once
#include "basic_shared_state.hpp"

template<class ValueType>
struct basic_connection
{
    basic_connection(std::shared_ptr<shared_state_base<ValueType>> impl
                     = nullptr)
        : impl_(std::move(impl))
    {}

    basic_connection(basic_connection &&) noexcept = default;
    basic_connection(basic_connection const &) = delete;
    basic_connection &
    operator=(basic_connection &&) noexcept = default;
    basic_connection &
    operator=(basic_connection const &)
        = delete;

    ~basic_connection() noexcept { disconnect(); }

    bool
    connected() const
    {
        return bool(impl_);
    }

    auto
    consume() -> net::awaitable<ValueType>
    {
        assert(connected());
        return impl_->consume();
    }

    void
    disconnect() noexcept
    {
        if (impl_)
        {
            impl_->stop();
            impl_.reset();
        }
    }

private:
    std::shared_ptr<shared_state_base<ValueType>> impl_;
};
