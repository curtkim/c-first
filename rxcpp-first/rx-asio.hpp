#include "rxcpp/rx-scheduler.hpp"
#define WITHOUT_BOOST ON

// TODO: C++17 Networking TS
#ifdef WITHOUT_BOOST
  // Standalone ASIO
  #include <asio.hpp>
  namespace asio_ns=::asio;
  namespace system_ns=::std;
#else
  // Boost.ASIO
  #include <boost/asio.hpp>

  namespace asio_ns=::boost::asio;
  namespace system_ns=::boost::system;
#endif

namespace rxcpp {
namespace schedulers {
class asio : public scheduler_interface
{
  typedef asio this_type;
  asio_ns::io_service& io_service;

  asio(const this_type&) = delete;

  struct asio_worker : public worker_interface
  {
  private:
    typedef asio_worker this_type;

  public:
    explicit asio_worker(composite_subscription cs, asio_ns::io_service& ios_)
        : lifetime(cs), ios(ios_)
    {
      printf("worker %p created\n", this);
    }

    virtual ~asio_worker()
    {
      printf("worker %p destroyed\n", this);
      lifetime.unsubscribe();
    }

    virtual clock_type::time_point now() const override { return clock_type::now(); }

    virtual void schedule(const schedulable& scbl) const override
    {
      if (scbl.is_subscribed()) {
        auto keep_alive = shared_from_this();
        ios.post([=]() {
          (void)(keep_alive);
          //std::cout << "schedule\n";
          // allow recursion
          scbl(recursion(true).get_recurse());
        });
      }
    }

    virtual void schedule(clock_type::time_point when, const schedulable& scbl) const override
    {
      if (scbl.is_subscribed()) {

        //std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(when - now());
        //std::cout << "schedule --> when " << duration.count() << " sec\n";

        //std::time_t now_c = std::chrono::steady_clock::to_time_t(when);
        //printf("scheduled on %p with timeout\n", this);
        auto keep_alive = shared_from_this();
        auto timer = std::make_shared<asio_ns::basic_waitable_timer<clock_type>>(ios, when);
        timer->async_wait([=](const system_ns::error_code&) {
          (void)(keep_alive);
          (void)(timer);
          // allow recursion
          scbl(recursion(true).get_recurse());
        });
      }
    }

    composite_subscription lifetime;
    asio_ns::io_service& ios;
  };

public:
  asio(asio_ns::io_service& ios)
      : io_service(ios) { }

  virtual ~asio() { }

  virtual clock_type::time_point now() const { return clock_type::now(); }

  virtual worker create_worker(composite_subscription cs) const
  {
    return worker(cs, std::make_shared<asio_worker>(cs, io_service));
  }
};

inline scheduler make_asio(asio_ns::io_service& ios)
{
  return make_scheduler<asio>(ios);
}
}   // End of namespace schedulers

inline observe_on_one_worker observe_on_asio(asio_ns::io_service& io_service)
{
  return observe_on_one_worker(rxsc::make_asio(io_service));
}

inline synchronize_in_one_worker synchronize_in_asio(asio_ns::io_service& io_service)
{
  return synchronize_in_one_worker(rxsc::make_asio(io_service));
}

inline identity_one_worker identitiy_asio(asio_ns::io_service& io_service)
{
  return identity_one_worker(rxsc::make_asio(io_service));
}

inline serialize_one_worker serialize_asio(asio_ns::io_service& io_service)
{
  return serialize_one_worker(rxsc::make_asio(io_service));
}
}   // End of namespace rxcpp