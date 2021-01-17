//
// Copyright (c) 2020 Richard Hodges (hodges.r@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include "variant_websocket.hpp"

#include <boost/algorithm/string.hpp>

#include <regex>

using namespace std::literals;

namespace websocket
{
namespace
{
    // from https://en.cppreference.com/w/cpp/utility/variant/visit
    template < class... Ts >
    struct overloaded : Ts...
    {
        using Ts::operator()...;
    };
    template < class... Ts >
    overloaded(Ts...) -> overloaded< Ts... >;

    std::string
    deduce_port(std::string const &scheme, std::string port)
    {
        using boost::algorithm::iequals;

        if (port.empty())
        {
            if (iequals(scheme, "ws") || iequals(scheme, "http"))
                port = "http";
            else if (iequals(scheme, "wss") || iequals(scheme, "https"))
                port = "https";
            else
                throw system_error(net::error::invalid_argument,
                                   "can't deduce port");
        }

        return port;
    }

    transport_type
    deduce_transport(std::string const &scheme, std::string const &port)
    {
        using boost::algorithm::iequals;
        if (scheme.empty())
        {
            if (port.empty())
                return transport_tcp;

            if (iequals(port, "http") || iequals(port, "ws") or
                iequals(port, "80"))
                return transport_tcp;

            if (iequals(port, "https") || iequals(port, "wss") or
                iequals(port, "443"))
                return transport_tls;

            throw system_error(net::error::invalid_argument,
                               "cannot deduce transport");
        }
        else
        {
            if (iequals(scheme, "http") || iequals(scheme, "ws"))
                return transport_tcp;

            if (iequals(scheme, "https") || iequals(scheme, "wss"))
                return transport_tls;

            throw system_error(net::error::invalid_argument, "invalid scheme");
        }
    }

    auto
    set_sni(tls_layer &stream, std::string const &host) -> void
    {
        if (not SSL_set_tlsext_host_name(stream.native_handle(), host.c_str()))
            throw system_error(error_code(static_cast< int >(::ERR_get_error()),
                                          net::error::get_ssl_category()));
    }

    std::string
    build_target(std::string const &path,
                 std::string const &query,
                 std::string const &fragment)
    {
        std::string result;

        if (path.empty())
            result = "/";
        else
            result = path;

        if (!query.empty())
            result += "?" + query;

        if (!fragment.empty())
            result += "#" + fragment;

        return result;
    }

    net::awaitable< net::ip::tcp::resolver::results_type >
    resolve(std::string const &host,
            std::string const &port,
            async::stop_token  stop)
    {
        auto resolver =
            net::ip::tcp::resolver(co_await net::this_coro::executor);
        auto stopconn = stop.connect([&] { resolver.cancel(); });
        co_return co_await resolver.async_resolve(
            host, port, net::use_awaitable);
    }

    net::awaitable< void >
    connect_tcp(beast::tcp_stream &                  stream,
                net::ip::tcp::resolver::results_type results,
                async::stop_token                    stop)
    {
        stream.expires_after(30s);
        auto stopconn = stop.connect([&] { stream.cancel(); });
        auto ep = co_await stream.async_connect(results, net::use_awaitable);
        boost::ignore_unused(ep);
    }

    net::awaitable< void >
    connect_tls(beast::ssl_stream< beast::tcp_stream > &stream,
                std::string const &                     host,
                async::stop_token                       stop)
    {
        set_sni(stream, host);
        stream.next_layer().expires_after(30s);
        auto stopconn = stop.connect([&] { stream.next_layer().cancel(); });
        co_await stream.async_handshake(net::ssl::stream_base::client,
                                        net::use_awaitable);
    }
}   // namespace

auto
variant_websocket::connect(net::any_io_executor   exec,
                           std::string const &    url,
                           connect_options const &opts)
    -> net::awaitable< void >
{
    assert(holds_alternative< monostate >(var_));

    static auto url_regex = std::regex(
        R"regex((ws|wss|http|https)://([^/ :]+):?([^/ ]*)(/?[^ #?]*)\x3f?([^ #]*)#?([^ ]*))regex",
        std::regex_constants::icase);
    auto match = std::smatch();
    if (not std::regex_match(url, match, url_regex))
        throw system_error(net::error::invalid_argument, "invalid url");

    auto &protocol = match[1];
    auto &host     = match[2];
    auto &port_ind = match[3];
    auto &path     = match[4];
    auto &query    = match[5];
    auto &fragment = match[6];

    auto transport = deduce_transport(protocol, port_ind);
    auto port      = deduce_port(protocol, port_ind);

    auto to = beast::websocket::stream_base::timeout {
        .handshake_timeout = std::chrono::seconds(30),
        .idle_timeout      = opts.pingpong_timeout.count()
                                 ? opts.pingpong_timeout / 2
                                 : beast::websocket::stream_base::none(),
        .keep_alive_pings  = opts.pingpong_timeout.count() ? true : false
    };

    switch (transport)
    {
    case transport_tcp:
        emplace_tcp(exec, to);
        break;
    case transport_tls:
        emplace_tls(exec, opts.sslctx, to);
        break;
    }
    auto &tcp_layer = get_tcp();

    // connect tcp
    co_await connect_tcp(
        tcp_layer, co_await resolve(host.str(), port, opts.stop), opts.stop);

    // tls handshake
    if (auto tls = query_tls())
        co_await connect_tls(*tls, host.str(), opts.stop);

    // websocket handshake

    set_headers(opts.headers);
    beast::websocket::response_type response;
    co_await client_handshake(
        response, host.str(), build_target(path, query, fragment));

    tcp_layer.expires_never();
}

tcp_layer &
variant_websocket::get_tcp()
{
    return visit(overloaded {
        [](ws_layer &ws) -> decltype(auto) { return ws.next_layer(); },
        [](wss_layer &wss) -> decltype(auto) {
            return wss.next_layer().next_layer();
        } });
}

void
variant_websocket::set_headers(const beast::http::fields &headers)
{
    visit([&](auto &ws) {
        ws.set_option(beast::websocket::stream_base::decorator(
            [headers](beast::websocket::request_type &req) {
                for (auto &&field : headers)
                    req.insert(field.name(), field.value());
            }));
    });
}

net::awaitable< void >
variant_websocket::client_handshake(beast::websocket::response_type &response,
                                    std::string const &              host,
                                    std::string const &              target)
{
    return visit([&](auto &ws) {
        return ws.async_handshake(response, host, target, net::use_awaitable);
    });
}

net::any_io_executor
variant_websocket::get_executor()
{
    return visit([](auto &ws) { return ws.get_executor(); });
}

net::awaitable< void >
variant_websocket::send_close(beast::websocket::close_reason cr)
{
    assert(get_executor() == co_await net::this_coro::executor);

    co_await visit([&](auto &ws) {
        error_code ec;
        return ws.async_close(cr, net::redirect_error(net::use_awaitable, ec));
    });
}

net::awaitable< void >
variant_websocket::drop()
{
    co_await send_close();
}

void
variant_websocket::text()
{
    visit([](auto &ws) { ws.text(); });
}
void
variant_websocket::binary()
{
    visit([](auto &ws) { ws.binary(); });
}

tls_layer *
variant_websocket::query_tls()
{
    if (holds_alternative< wss_layer >(var_))
        return std::addressof(get< wss_layer >(var_).next_layer());
    else
        return nullptr;
}

void
variant_websocket::emplace_tls(net::any_io_executor                   exec,
                               ssl::context &                         sslctx,
                               beast::websocket::stream_base::timeout to)
{
    assert(holds_alternative< monostate >(var_));
    auto &wss = var_.emplace< wss_layer >(std::move(exec), sslctx);
    wss.set_option(to);
}

void
variant_websocket::emplace_tcp(net::any_io_executor                   exec,
                               beast::websocket::stream_base::timeout to)
{
    assert(holds_alternative< monostate >(var_));
    auto &ws = var_.emplace< ws_layer >(std::move(exec));
    ws.set_option(to);
}

beast::websocket::close_reason
variant_websocket::variant_websocket::reason() const
{
    return visit([](auto &ws) { return ws.reason(); });
}

net::awaitable< std::tuple< error_code, std::size_t > >
variant_websocket::read(beast::flat_buffer &buf)
{
    error_code ec;
    auto       size = co_await visit([&](auto &ws) {
        return ws.async_read(buf, net::redirect_error(net::use_awaitable, ec));
    });
    co_return std::make_tuple(ec, size);
}

bool
variant_websocket::is_binary() const
{
    return visit([](auto &ws) { return ws.binary(); });
}

}   // namespace websocket