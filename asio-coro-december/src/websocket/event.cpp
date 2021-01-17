//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "event.hpp"

namespace websocket
{
bool
event::is_error() const
{
    return holds_alternative< error_code >(var_);
}

bool
event::is_message() const
{
    return holds_alternative< struct message >(var_);
}

error_code const &
event::error() const
{
    assert(is_error());
    return get< error_code >(var_);
}

struct message const &
event::message() const
{
    assert(is_message());
    return get< struct message >(var_);
}

struct message &
event::message()
{
    assert(is_message());
    return get< struct message >(var_);
}

event::event(error_code ec)
: var_(ec)
{
}

event::event(struct message msg)
: var_(std::move(msg))
{
}

}   // namespace websocket