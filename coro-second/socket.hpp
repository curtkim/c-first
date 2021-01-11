#pragma once

#include <cstdint>

/**
 * @struct addrinfo
 * @brief identify an Internet host and a service
 *
 * @link https://docs.microsoft.com/en-us/windows/win32/api/ws2def/
 * @see https://docs.microsoft.com/en-us/windows/win32/api/ws2def/
 */
struct addrinfo;

struct sockaddr_in;
struct sockaddr_in6;

/**
 * @brief initialize platform specific network module
 * @throw std::system_error
 */
void socket_setup() noexcept(false);

/**
 * @brief release platform specific network module
 */
void socket_teardown() noexcept;

/**
 * @brief network related error
 *
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_recent() noexcept;

/**
 * @brief check if given socket descriptor is valid
 *
 * @param sd
 * @return true     the sd is considered valid
 * @return false    negative or `INVALID_SOCKET`
 */
bool socket_is_valid(int64_t sd) noexcept;

/**
 * @brief create 1 socket
 * @param hint family, type, protocol
 * @param sd   reference to save the new socket
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_create(const addrinfo& hint, int64_t& sd) noexcept;

/**
 * @brief dispose the given socket
 *
 * @param sd
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_close(int64_t sd) noexcept;

/**
 * @brief bind the socket to given address
 *
 * @param sd    socket to bind
 * @param local IPv4 address
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_bind(int64_t sd, const sockaddr_in& local) noexcept;

/**
 * @brief bind the socket to given address
 *
 * @param sd    socket to bind
 * @param local IPv6 address
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_bind(int64_t sd, const sockaddr_in6& local) noexcept;

/**
 * @brief start listening with the socket
 *
 *  The `backlog` value is fixed to 7.
 *
 * @param sd    socket to start listen
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_listen(int64_t sd) noexcept;

/**
 * @brief try connect to given endpoint(IPv4)
 *
 * @param sd        socket to start connect
 * @param remote    IPv4 address
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_connect(int64_t sd, const sockaddr_in& remote) noexcept;

/**
 * @brief try connect to given endpoint(IPv6)
 *
 * @param sd        socket to start connect
 * @param remote    IPv6 address
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_connect(int64_t sd, const sockaddr_in6& remote) noexcept;

/**
 * @brief accept a connection request and return client socket
 *
 * @param ln    listener socket
 * @param sd    reference to descriptor to save the new connected socket
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_accept(int64_t ln, int64_t& sd) noexcept;

/**
 * @brief get the socket's address
 *
 * @param sd    socket to query the bound address
 * @param local object to receive IPv4 address
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_get_name(int64_t sd, sockaddr_in& local) noexcept;

/**
 * @brief get the socket's address
 *
 * @param sd    socket to query the bound address
 * @param local object to receive IPv6 address
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_get_name(int64_t sd, sockaddr_in6& local) noexcept;

/**
 * @brief   get connected peer's address
 *
 * @param sd    socket to query the peer address
 * @param remote    object to receive IPv4 address
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_get_peer(int64_t sd, sockaddr_in& remote) noexcept;

/**
 * @brief   get connected peer's address
 *
 * @param sd    socket to query the peer address
 * @param remote    object to receive IPv6 address
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_get_peer(int64_t sd, sockaddr_in6& remote) noexcept;

/**
 * @brief change the socket's option
 *
 * @param sd
 * @param level
 * @param option
 * @param value
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_set_option(int64_t sd, int64_t level, //
                           int64_t option, int64_t value) noexcept;

/**
 * @brief make socket to operate in non-blocking mode
 *
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_set_option_nonblock(int64_t sd) noexcept;

/**
 * @brief make socket to reuse address
 *
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_set_option_reuse_address(int64_t sd) noexcept;

/**
 * @brief make tcp send without delay
 *
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_set_option_nodelay(int64_t sd) noexcept;

/**
 * @brief set the socket's send timeout
 *
 * @param sd    socket to change timeout
 * @param us    microsecond fot the duration
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_set_option_send_timout(int64_t sd, //
                                       uint32_t us) noexcept;

/**
 * @brief set the socket's recv timeout
 *
 * @param sd    socket to change timeout
 * @param us    microsecond fot the duration
 * @return  error code from `errno` or `WSAGetLastError`
 */
uint32_t socket_set_option_recv_timout(int64_t sd, //
                                       uint32_t us) noexcept;

/**
 * @brief test if the error code is for non-blocking
 */
bool socket_would_block(uint32_t ec) noexcept;
