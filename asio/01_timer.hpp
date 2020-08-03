#pragma once

#include <asio.hpp>

class TimerContext {
public:
  asio::system_timer timer;
  std::function<void()> func;
  std::function<void(const std::error_code &e)> func_wrapper;
  bool autofree = false;

  explicit TimerContext(asio::io_service &__io_svc) : timer(__io_svc) {
  }
};

void clearInterval(TimerContext *&ctx) {
  if (ctx) {
    ctx->timer.cancel();
    ctx = nullptr;
  }
}

void clearTimeout(TimerContext *&ctx) {
  clearInterval(ctx);
}

TimerContext* setInterval(asio::io_service &io_svc, const std::function<void()> &func, size_t interval) {
  auto *ctx = new TimerContext(io_svc);
  ctx->func = func;

  ctx->func_wrapper = [ctx, interval](const std::error_code &e) {
    if (e) {
      delete ctx;
      return;
    }

    ctx->timer.expires_from_now(std::chrono::milliseconds(interval));
    ctx->timer.async_wait(ctx->func_wrapper);

    ctx->func();
  };

  ctx->timer.expires_from_now(std::chrono::milliseconds(interval));
  ctx->timer.async_wait(ctx->func_wrapper);

  return ctx;
}

auto setTimeout(asio::io_service &io_svc, const std::function<void()> &func, size_t timeout) {
  auto *ctx = new TimerContext(io_svc);
  ctx->autofree = true;
  ctx->func = func;

  struct ret {
    operator TimerContext*() {
      std::cout << " ** " << std::endl;
      ctx->autofree = false;
      return ctx;
    }

    TimerContext *ctx;
  };

  ctx->func_wrapper = [ctx](const std::error_code &e) {
    if (e) {
      delete ctx;
      return;
    }

    ctx->func();

    if (ctx->autofree)
      delete ctx;
  };

  ctx->timer.expires_from_now(std::chrono::milliseconds(timeout));
  ctx->timer.async_wait(ctx->func_wrapper);

  return ret{ctx};
}
