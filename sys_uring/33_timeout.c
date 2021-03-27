#include <liburing.h>
#include <stdio.h>

#include <errno.h>
#include <string.h>
#include <sys/time.h>

#define TIMEOUT_MSEC	200


// gettimeofday으로 system의 현재시간을 timeval에 채운다.
// io_uring_prep_timeout를 호출하기 위해서는 timespec을 채워야 한다.


// struct_timeval.h (in system)
//struct timeval
//{
//  __time_t tv_sec;		/* Seconds.  */
//  __suseconds_t tv_usec;	/* Microseconds.  */
//};

// in liburing
//struct __kernel_timespec {
//  int64_t		tv_sec;
//  long long	tv_nsec;
//};


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

static int test_single_timeout(struct io_uring *ring)
{
  struct io_uring_cqe *cqe;
  struct io_uring_sqe *sqe;

  struct __kernel_timespec ts;
  struct timeval tv;
  int ret;

  sqe = io_uring_get_sqe(ring);
  if (!sqe) {
    fprintf(stderr, "%s: get sqe failed\n", __FUNCTION__);
    goto err;
  }

  msec_to_ts(&ts, TIMEOUT_MSEC);
  printf("sec= %lld nsec= %lld\n", ts.tv_sec, ts.tv_nsec);
  io_uring_prep_timeout(sqe, &ts, 0, 0); // timespec, count, flag

  ret = io_uring_submit(ring);
  if (ret <= 0) {
    fprintf(stderr, "%s: sqe submit failed: %d\n", __FUNCTION__, ret);
    goto err;
  }

  gettimeofday(&tv, NULL);
  ret = io_uring_wait_cqe(ring, &cqe);
  if (ret < 0) {
    fprintf(stderr, "%s: wait completion %d\n", __FUNCTION__, ret);
    goto err;
  }
  ret = cqe->res;
  io_uring_cqe_seen(ring, cqe);
  if (ret == -EINVAL) {
    fprintf(stdout, "%s: Timeout not supported, ignored\n", __FUNCTION__);
    return 0;
  } else if (ret != -ETIME) {
    fprintf(stderr, "%s: Timeout: %s\n", __FUNCTION__, strerror(-ret));
    goto err;
  }

  unsigned long long exp = mtime_since_now(&tv);
  printf("%lld", exp);

  if (exp >= TIMEOUT_MSEC / 2 && exp <= (TIMEOUT_MSEC * 3) / 2)
    return 0;
  fprintf(stderr, "%s: Timeout seems wonky (got %llu)\n", __FUNCTION__, exp);
  err:
  return 1;
}

int main(int argc, char *argv[]) {
  printf("sizeof(long long) = %d\n", sizeof(long long));

  struct io_uring ring;
  int ret;

  if (argc > 1)
    return 0;

  ret = io_uring_queue_init(8, &ring, 0);
  if (ret) {
    fprintf(stderr, "ring setup failed\n");
    return 1;
  }

  ret = test_single_timeout(&ring);
  if (ret) {
    fprintf(stderr, "test_single_timeout failed\n");
    return ret;
  }
  return 0;
}
