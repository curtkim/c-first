//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "message.hpp"

#include <iomanip>
#include <ostream>

namespace websocket
{
message::message(std::shared_ptr< beast::flat_buffer > data, bool isbin)
: data_(std::move(data))
, is_binary_(isbin)
{
}

bool
message::is_text() const
{
    return !is_binary_;
}

bool
message::is_binary() const
{
    return is_binary_;
}

std::string_view
message::text() const
{
    assert(is_text());
    auto d = data_->data();
    return std::string_view(reinterpret_cast< const char * >(d.data()),
                            d.size());
}

std::span< std::byte >
message::binary() const
{
    assert(is_binary());
    auto d = data_->data();
    return std::span< std::byte >(reinterpret_cast< std::byte * >(d.data()),
                                  d.size());
}

std::ostream &
operator<<(std::ostream &os, const message &m)
{
    if (m.data_)
    {
        if (m.is_binary())
            return os << "binary message: " << m.binary().size() << " bytes";
        else
            return os << std::quoted(m.text());
    }
    else
    {
        return os << "empty message";
    }
}
}   // namespace websocket