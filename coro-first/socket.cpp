
#include <cerrno>
#include <chrono>
#include <system_error>

#include "socket.hpp"

#if __has_include(<WinSock2.h>) // use winsock
#include <WS2tcpip.h>
#include <WinSock2.h>
#include <ws2def.h>

#elif __has_include(<netinet/in.h>) // use netinet
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <fcntl.h>
#include <unistd.h>

#endif

using namespace std;

/**
 * @brief create a socket without error check
 */
int64_t socket_create(const addrinfo& hint) noexcept {
  return ::socket(hint.ai_family, //
                  hint.ai_socktype, hint.ai_protocol);
}

uint32_t socket_create(const addrinfo& hint, int64_t& sd) noexcept {
  sd = socket_create(hint);
  return sd < 0 ? socket_recent() : 0;
}

uint32_t socket_bind(int64_t sd, const sockaddr_in& local) noexcept {
  if (::bind(sd, (const sockaddr*)&local, sizeof(sockaddr_in)))
    return socket_recent();
  return 0;
}
uint32_t socket_bind(int64_t sd, const sockaddr_in6& local) noexcept {
  // bind socket and address
  if (::bind(sd, (const sockaddr*)&local, sizeof(sockaddr_in6)))
    return socket_recent();
  return 0;
}

uint32_t socket_listen(int64_t sd) noexcept {
  ::listen(sd, 7);
  return socket_recent();
}

uint32_t socket_connect(int64_t sd, const sockaddr_in& remote) noexcept {
  auto* ptr = reinterpret_cast<const sockaddr*>(&remote);
  if (::connect(sd, ptr, sizeof(sockaddr_in)) < 0)
    return socket_recent();
  return 0;
}

uint32_t socket_connect(int64_t sd, const sockaddr_in6& remote) noexcept {
  auto* ptr = reinterpret_cast<const sockaddr*>(&remote);
  if (::connect(sd, ptr, sizeof(sockaddr_in6)) < 0)
    return socket_recent();
  return 0;
}

uint32_t socket_accept(int64_t ln, int64_t& sd) noexcept {
  sd = ::accept(ln, nullptr, nullptr);
  return socket_recent();
}

uint32_t socket_get_name(int64_t sd, sockaddr_in& local) noexcept {
  socklen_t len = sizeof(sockaddr_in);
  if (::getsockname(sd, (sockaddr*)&local, &len))
    return socket_recent();
  return 0;
}

uint32_t socket_get_name(int64_t sd, sockaddr_in6& local) noexcept {
  socklen_t len = sizeof(sockaddr_in6);
  if (::getsockname(sd, (sockaddr*)&local, &len))
    return socket_recent();
  return 0;
}

uint32_t socket_get_peer(int64_t sd, sockaddr_in& local) noexcept {
  socklen_t len = sizeof(sockaddr_in);
  if (::getpeername(sd, (sockaddr*)&local, &len))
    return socket_recent();
  return 0;
}

uint32_t socket_get_peer(int64_t sd, sockaddr_in6& local) noexcept {
  socklen_t len = sizeof(sockaddr_in6);
  if (::getpeername(sd, (sockaddr*)&local, &len))
    return socket_recent();
  return 0;
}

uint32_t socket_set_option(int64_t sd, int64_t level, //
                           int64_t option, int64_t value) noexcept {
  if (auto ec = ::setsockopt(sd, level, option, (char*)&value, sizeof(value)))
    return socket_recent();
  return 0;
}

uint32_t socket_set_option_reuse_address(int64_t sd) noexcept {
  return socket_set_option(sd, SOL_SOCKET, SO_REUSEADDR, true);
}

uint32_t socket_set_option_nodelay(int64_t sd) noexcept {
  return socket_set_option(sd, IPPROTO_TCP, TCP_NODELAY, true);
}

uint32_t socket_set_option_timout(int64_t sd, chrono::microseconds us,
                                  int64_t option) noexcept {
  const auto s = chrono::duration_cast<chrono::seconds>(us);
  us -= s;
  timeval timeout{};
  timeout.tv_sec = s.count();
  timeout.tv_usec = us.count();

  if (::setsockopt(sd, SOL_SOCKET, option, //
                   (char*)&timeout, sizeof(timeval)) != 0) {
    return socket_recent();
  }
  return 0;
}

uint32_t socket_set_option_timout(int64_t sd, uint32_t us,
                                  int64_t option) noexcept {
  return socket_set_option_timout(sd, chrono::microseconds{us}, option);
}

uint32_t socket_set_option_send_timout(int64_t sd, //
                                       uint32_t us) noexcept {
  return socket_set_option_timout(sd, chrono::microseconds{us}, SO_SNDTIMEO);
}
uint32_t socket_set_option_recv_timout(int64_t sd, //
                                       uint32_t us) noexcept {
  return socket_set_option_timout(sd, chrono::microseconds{us}, SO_RCVTIMEO);
}

#if __has_include(<WinSock2.h>) // using winsock

bool socket_is_valid(int64_t h) noexcept {
    return h != INVALID_SOCKET;
}

WSADATA wsa_data{};

void socket_setup() noexcept(false) {
    if (wsa_data.wVersion) // already initialized
        return;

    // init version 2.2
    if (::WSAStartup(MAKEWORD(2, 2), &wsa_data)) {
        auto errc = WSAGetLastError();
        throw system_error{errc, system_category(), "WSAStartup"};
    }
}

void socket_teardown() noexcept {
    // not initialized or released
    if (wsa_data.wVersion == 0)
        return;
    ::WSACleanup();
    wsa_data.wVersion = 0;
}

uint32_t socket_recent() noexcept {
    return static_cast<uint32_t>(WSAGetLastError());
}

bool socket_would_block(int ec) noexcept {
    return ec == WSAEWOULDBLOCK || ec == EWOULDBLOCK || ec == EINPROGRESS ||
           ec == ERROR_IO_PENDING;
}

uint32_t socket_close(int64_t sd) noexcept {
    return ::shutdown(sd, SD_BOTH), ::closesocket(sd);
}

uint32_t socket_set_option_nonblock(int64_t sd) noexcept {
    u_long mode = TRUE;
    return ::ioctlsocket(sd, FIONBIO, &mode);
}

#elif __has_include(<netinet/in.h>) // using netinet

bool socket_is_valid(int64_t h) noexcept {
  return h > 0;
}

void socket_setup() noexcept(false) {
  // do nothing for posix system. network operation already available
}
void socket_teardown() noexcept {
}

uint32_t socket_recent() noexcept {
  return static_cast<uint32_t>(errno);
}

bool socket_would_block(int ec) noexcept {
  return ec == EINPROGRESS;
}

uint32_t socket_close(int64_t sd) noexcept {
  return shutdown(sd, SHUT_RDWR), close(sd), errno;
}

uint32_t socket_set_option_nonblock(int64_t sd) noexcept {
  // make non-block/async
  if (fcntl(sd, F_SETFL, O_NONBLOCK) != 0)
    return errno;
  return 0;
}

#endif
