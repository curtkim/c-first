// https://github.com/ReactiveX/RxCpp/issues/528

#include "rxcpp/rx.hpp"
#include "rx-asio.hpp"

struct IoServiceContext
{
  static auto GetContext()
  {
    static auto TheContext = std::make_shared<IoServiceContext>();
    return TheContext;
  };

  asio::io_service m_ios;
  std::vector<std::thread> m_threads;
  asio::io_service::work m_work;

  IoServiceContext() : m_work(m_ios) {
    unsigned int thread_pool_size = std::thread::hardware_concurrency() * 2;
    if (thread_pool_size == 0)
      thread_pool_size = 2;

    for (unsigned int i = 0; i < thread_pool_size; i++)
    {
      auto th = std::thread([this]() {
        try {
          m_ios.run();
        }
        catch (const std::exception& e)
        {
          auto w = e.what();
        }
        catch (...)
        {
        }
      });
      th.detach();
      m_threads.push_back(std::move(th));
    }
  };

  //~IoServiceContext(); // default dtor() good enough

};

//#define DEFAULT
#define ASIO
//#define NEW_THREAD

int main(int argc, const char *const argv[]) {

  // avoid subjects and prefer operators to subscribe
  //static rxcpp::subjects::subject<int> m_picture_processed, m_picture;

  std::cout << std::this_thread::get_id() << " main thread" << "\n";

  auto start = std::chrono::system_clock::now();
  IoServiceContext::GetContext();
  std::this_thread::sleep_for(std::chrono::milliseconds (500));

  auto asio_coordination = rxcpp::synchronize_in_asio(IoServiceContext::GetContext()->m_ios);

  rxcpp::observable<>::range(1,5)
      .observe_on(asio_coordination)
      .flat_map([&asio_coordination](int i) {
        return rxcpp::observable<>::just(i)
#ifdef DEFAULT
            .timeout(std::chrono::seconds(1));
#endif
#ifdef ASIO
        .timeout(std::chrono::seconds(1), asio_coordination);
#endif
#ifdef NEW_THREAD
        .timeout(std::chrono::seconds(1), rxcpp::synchronize_new_thread());
#endif

      })
      .tap([](int i) {
        std::cout << std::this_thread::get_id() << " " << i << "\n";
      })
      .as_blocking()
      .count();

  return 0;
}
