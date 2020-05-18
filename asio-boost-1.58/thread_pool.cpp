#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <boost/asio.hpp>
#include <boost/thread.hpp>


struct thread_pool {
    typedef std::unique_ptr<boost::asio::io_service::work> asio_worker;

    thread_pool(int threads) :service(), service_worker(new boost::asio::io_service::work(service)) {
        for (int i = 0; i < threads; ++i) {
            auto worker = [this] { return service.run(); };
            grp.add_thread(new boost::thread(worker));
        }
    }

    template<class F>
    void enqueue(F f) {
        service.post(f);
    }

    ~thread_pool() {
        service_worker.reset();
        grp.join_all();
        service.stop();
    }

private:
    boost::asio::io_service service;
    asio_worker service_worker;
    boost::thread_group grp;
};

int main() {
    thread_pool pool(2);

    pool.enqueue([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "Hello from Task 1 " << std::this_thread::get_id() << std::endl;
    });

    pool.enqueue([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "Hello from Task 2 " << std::this_thread::get_id() << std::endl;
    });
}