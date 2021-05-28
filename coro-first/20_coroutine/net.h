#pragma once
#ifndef COROUTINE_NET_IO_H
#define COROUTINE_NET_IO_H
#include <gsl/gsl>

#include <20_coroutine/return.h>

/**
 * @defgroup Network
 * Helper types to apply `co_await` for socket operations + Name resolution utilities
 */

#if __has_include(<WinSock2.h>) // use winsock
#include <WS2tcpip.h>
#include <WinSock2.h>
#include <ws2def.h>

/** @brief indicates current import: using Windows Socket API? */
/** @brief indicates current import: using <netinet/*.h>? */
static constexpr bool is_winsock = true;
static constexpr bool is_netinet = false;

using io_control_block = OVERLAPPED;

#elif __has_include(<netinet/in.h>) // use netinet
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

static constexpr bool is_winsock = false;
static constexpr bool is_netinet = true;

/**
 * @brief Follow the definition of Windows `OVERLAPPED`
 * @see https://docs.microsoft.com/en-us/windows/win32/sync/synchronization-and-overlapped-input-and-output
 * @see https://docs.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-overlapped
 * @ingroup Network
 */
struct io_control_block {
  uint64_t internal;      // uint32_t errc, int32_t flag
  uint64_t internal_high; // int64_t len, socklen_t addrlen
  union {
    struct {
      int32_t offset;
      int32_t offset_high;
    };
    void* ptr; // sockaddr* addr;
  };
  int64_t handle; // int64_t sd;
};

#endif // winsock || netinet

namespace coro {

/**
 * @brief This is simply a view to storage. Be aware that it doesn't have ownership
 * @ingroup Network
 */
  using io_buffer_t = gsl::span<std::byte>;
  static_assert(sizeof(io_buffer_t) <= sizeof(void*) * 2);

/**
 * @brief A struct to describe "1 I/O request" to system API.
 * @ingroup Network
 * When I/O request is submitted, an I/O task becomes 1 20_coroutine handle
 */
  class io_work_t : public io_control_block {
  public:
    coroutine_handle<void> task{};
    io_buffer_t buffer{};

  protected:
    /**
     * @see await_ready
     * @return true  The given socket can be use for non-blocking operations
     * @return false For Windows, the return is always `false`
     */
    bool ready() const noexcept;

  public:
    /**
     * @brief Multiple retrieving won't be a matter
     * @return uint32_t error code from the system
     */
    uint32_t error() const noexcept;
  };
  static_assert(sizeof(io_work_t) <= 56);

/**
 * @brief Awaitable type to perform `sendto` I/O request
 * @see sendto
 * @see WSASendTo
 * @ingroup Network
 */
  class io_send_to final : public io_work_t {
  private:
    /**
     * @brief makes an I/O request with given context(`coroutine_handle<void>`)
     * @throw std::system_error
     */
    void suspend(coroutine_handle<void> t) noexcept(false);
    /**
     * @brief Fetch I/O result/error
     * @return int64_t return of `sendto`
     * 
     * This function must be used through `co_await`.
     * Multiple invoke of this will lead to malfunction.
     */
    int64_t resume() noexcept;

  public:
    bool await_ready() const noexcept {
      return this->ready();
    }
    /**
     * @throw std::system_error
     */
    void await_suspend(coroutine_handle<void> t) noexcept(false) {
      return this->suspend(t);
    }
    int64_t await_resume() noexcept {
      return this->resume();
    }
  };
  static_assert(sizeof(io_send_to) == sizeof(io_work_t));

/**
 * @brief Awaitable type to perform `recvfrom` I/O request
 * @see recvfrom
 * @see WSARecvFrom
 * @ingroup Network
 */
  class io_recv_from final : public io_work_t {
  private:
    /**
     * @brief makes an I/O request with given context(`coroutine_handle<void>`)
     * @throw std::system_error
     */
    void suspend(coroutine_handle<void> t) noexcept(false);
    /**
     * @brief Fetch I/O result/error
     * @return int64_t return of `recvfrom`
     * 
     * This function must be used through `co_await`.
     * Multiple invoke of this will lead to malfunction.
     */
    int64_t resume() noexcept;

  public:
    bool await_ready() const noexcept {
      return this->ready();
    }
    /**
     * @throw std::system_error
     */
    void await_suspend(coroutine_handle<void> t) noexcept(false) {
      return this->suspend(t);
    }
    int64_t await_resume() noexcept {
      return this->resume();
    }
  };
  static_assert(sizeof(io_recv_from) == sizeof(io_work_t));

/**
 * @brief Awaitable type to perform `send` I/O request
 * @see send
 * @see WSASend
 * @ingroup Network
 */
  class io_send final : public io_work_t {
  private:
    /**
     * @brief makes an I/O request with given context(`coroutine_handle<void>`)
     * @throw std::system_error
     */
    void suspend(coroutine_handle<void> t) noexcept(false);
    /**
     * @brief Fetch I/O result/error
     * @return int64_t return of `send`
     * 
     * This function must be used through `co_await`.
     * Multiple invoke of this will lead to malfunction.
     */
    int64_t resume() noexcept;

  public:
    bool await_ready() const noexcept {
      return this->ready();
    }
    void await_suspend(coroutine_handle<void> t) noexcept(false) {
      return this->suspend(t);
    }
    int64_t await_resume() noexcept {
      return this->resume();
    }
  };
  static_assert(sizeof(io_send) == sizeof(io_work_t));

/**
 * @brief Awaitable type to perform `recv` I/O request
 * @see recv
 * @see WSARecv
 * @ingroup Network
 */
  class io_recv final : public io_work_t {
  private:
    /**
     * @brief makes an I/O request with given context(`coroutine_handle<void>`)
     * @throw std::system_error
     */
    void suspend(coroutine_handle<void> t) noexcept(false);

    /**
     * @brief Fetch I/O result/error
     * @return int64_t return of `recv`
     * 
     * This function must be used through `co_await`.
     * Multiple invoke of this will lead to malfunction.
     */
    int64_t resume() noexcept;

  public:
    bool await_ready() const noexcept {
      return this->ready();
    }
    void await_suspend(coroutine_handle<void> t) noexcept(false) {
      return this->suspend(t);
    }
    int64_t await_resume() noexcept {
      return this->resume();
    }
  };
  static_assert(sizeof(io_recv) == sizeof(io_work_t));

/**
 * @brief Constructs `io_send_to` awaitable with the given parameters
 * @param sd 
 * @param remote 
 * @param buf 
 * @param work 
 * @return io_send_to& 
 * 
 * @ingroup Network
 */
  auto send_to(uint64_t sd, const sockaddr_in& remote, io_buffer_t buf,
               io_work_t& work) noexcept(false) -> io_send_to&;

/**
 * @brief Constructs `io_send_to` awaitable with the given parameters
 * @param sd 
 * @param remote 
 * @param buf 
 * @param work 
 * @return io_send_to& 
 * 
 * @ingroup Network
 */
  auto send_to(uint64_t sd, const sockaddr_in6& remote, io_buffer_t buf,
               io_work_t& work) noexcept(false) -> io_send_to&;

/**
 * @brief Constructs `io_recv_from` awaitable with the given parameters
 * @param sd 
 * @param remote 
 * @param buf 
 * @param work 
 * @return io_recv_from& 
 * 
 * @ingroup Network
 */
  auto recv_from(uint64_t sd, sockaddr_in& remote, io_buffer_t buf,
                 io_work_t& work) noexcept(false) -> io_recv_from&;

/**
 * @brief Constructs `io_recv_from` awaitable with the given parameters
 * @param sd 
 * @param remote 
 * @param buf 
 * @param work 
 * @return io_recv_from& 
 * 
 * @ingroup Network
 */
  auto recv_from(uint64_t sd, sockaddr_in6& remote, io_buffer_t buf,
                 io_work_t& work) noexcept(false) -> io_recv_from&;

/**
 * @brief Constructs `io_send` awaitable with the given parameters
 * @param sd 
 * @param buf 
 * @param flag 
 * @param work 
 * @return io_send&
 *  
 * @ingroup Network
 */
  auto send_stream(uint64_t sd, io_buffer_t buf, uint32_t flag,
                   io_work_t& work) noexcept(false) -> io_send&;

/**
 * @brief Constructs `io_recv` awaitable with the given parameters
 * @param sd 
 * @param buf 
 * @param flag 
 * @param work 
 * @return io_recv& 
 * 
 * @ingroup Network
 */
  auto recv_stream(uint64_t sd, io_buffer_t buf, uint32_t flag,
                   io_work_t& work) noexcept(false) -> io_recv&;

/**
 * @brief Poll internal I/O works and invoke user callback
 * @param nano timeout in nanoseconds 
 * @throw std::system_error
 * 
 * @ingroup Network
 */
  void poll_net_tasks(uint64_t nano) noexcept(false);

/**
 * @brief Thin wrapper of `getaddrinfo` for IPv4
 * 
 * @param hint 
 * @param host 
 * @param serv 
 * @param output
 * @return uint32_t Error code from the `getaddrinfo` that can be the argument of `gai_strerror`
 * @see getaddrinfo
 * @see gai_strerror
 * 
 * @ingroup Network
 */
  uint32_t get_address(const addrinfo& hint, //
                       gsl::czstring<> host, gsl::czstring<> serv,
                       gsl::span<sockaddr_in> output) noexcept;

/**
 * @brief Thin wrapper of `getaddrinfo` for IPv6
 * 
 * @param hint 
 * @param host 
 * @param serv 
 * @param output
 * @return uint32_t Error code from the `getaddrinfo` that can be the argument of `gai_strerror`
 * @see getaddrinfo
 * @see gai_strerror
 * 
 * @ingroup Network
 */
  uint32_t get_address(const addrinfo& hint, //
                       gsl::czstring<> host, gsl::czstring<> serv,
                       gsl::span<sockaddr_in6> output) noexcept;

/**
 * @brief Thin wrapper of `getnameinfo`
 * 
 * @param addr 
 * @param name 
 * @param serv can be `nullptr`
 * @param flags 
 * @return uint32_t EAI_AGAIN ...
 * @see getnameinfo
 * 
 * @ingroup Network
 */
  uint32_t get_name(const sockaddr_in& addr, //
                    gsl::zstring<NI_MAXHOST> name, gsl::zstring<NI_MAXSERV> serv,
                    int32_t flags = NI_NUMERICHOST | NI_NUMERICSERV) noexcept;

/**
 * @brief Thin wrapper of `getnameinfo`
 * @param addr 
 * @param name 
 * @param serv can be `nullptr`
 * @param flags 
 * @return uint32_t EAI_AGAIN ...
 * @see getnameinfo
 * 
 * @ingroup Network
 */
  uint32_t get_name(const sockaddr_in6& addr, //
                    gsl::zstring<NI_MAXHOST> name, gsl::zstring<NI_MAXSERV> serv,
                    int32_t flags = NI_NUMERICHOST | NI_NUMERICSERV) noexcept;

} // namespace coro

#endif // COROUTINE_NET_IO_H