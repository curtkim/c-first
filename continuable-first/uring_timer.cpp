// 동작하지 않음ㅠㅠ

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <string>
#include <thread>
#include <liburing.h>
#include <sys/time.h>

#include <system_error>


#if defined(CONTINUABLE_HAS_EXCEPTIONS)
#include <exception>
#endif

#include <continuable/continuable.hpp>

static void msec_to_ts(struct __kernel_timespec *ts, unsigned int msec)
{
  ts->tv_sec = msec / 1000;
  ts->tv_nsec = (msec % 1000) * 1000000;
}

static unsigned long long mtime_since(const struct timeval *s, const struct timeval *e)
{
  long long sec, usec;

  sec = e->tv_sec - s->tv_sec;
  usec = (e->tv_usec - s->tv_usec);
  if (sec > 0 && usec < 0) {
    sec--;
    usec += 1000000;
  }

  sec *= 1000;
  usec /= 1000;
  return sec + usec;
}

static unsigned long long mtime_since_now(struct timeval *tv)
{
  struct timeval end;

  gettimeofday(&end, NULL);
  return mtime_since(tv, &end);
}

template<typename Data>
struct Holder {
  __kernel_timespec ts;
  cti::promise<Data> promise;

  Holder(cti::promise<Data> promise) : promise(promise){
  }
};

auto set_timeout(struct io_uring *ring){
  return cti::make_continuable<std::string>(
    [&ring](auto&& promise) {

      Holder<std::string>* pHolder = new Holder<std::string>{promise};
      msec_to_ts(&(pHolder->ts), 1000);
      pHolder->promise = std::move(promise);

      struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
      io_uring_sqe_set_data(sqe, pHolder);
      io_uring_prep_timeout(sqe, &(pHolder->ts), 0, 0); // timespec, count, flag
    });
}

int main(int, char**) {

  fmt::print("{} main thread\n", std::this_thread::get_id());


  struct io_uring ring;
  int ret = io_uring_queue_init(8, &ring, 0);

  set_timeout(&ring)
  .then([](std::string v){
    fmt::print("{} {}\n", std::this_thread::get_id(), v);
  });


  struct io_uring_cqe *cqe;
  ret = io_uring_wait_cqe(&ring, &cqe);
  if (ret < 0) {
    fprintf(stderr, "%s: wait completion %d\n", __FUNCTION__, ret);
    std::exit(1);
  }
  ret = cqe->res;
  io_uring_cqe_seen(&ring, cqe);
  if (ret == -EINVAL) {
    fprintf(stdout, "%s: Timeout not supported, ignored\n", __FUNCTION__);
    return 0;
  } else if (ret != -ETIME) {
    fprintf(stderr, "%s: Timeout: %s\n", __FUNCTION__, strerror(-ret));
    std::exit(1);
  }

  Holder<std::string>* pHolder = (Holder<std::string>*)io_uring_cqe_get_data(cqe);
  pHolder->promise.set_value("done");
  // Or promise.set_exception(...);
  delete pHolder;


  return 0;
}