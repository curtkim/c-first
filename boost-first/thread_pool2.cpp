#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/move/move.hpp>
#include <iostream>
#include <unistd.h>

int sleep_print(int seconds) {
    std::cout << "goint to sleep (" << seconds << ")" << std::endl;
    sleep(seconds);
    std::cout << "wake up (" << seconds << ")" << std::endl;
    return 0;
}

typedef boost::packaged_task<int> task_t;
typedef boost::shared_ptr<task_t> ptask_t;


void push_job(int seconds, boost::asio::io_service& io_service, std::vector<boost::shared_future<int> >& pending_data) {
    ptask_t task = boost::make_shared<task_t>(boost::bind(&sleep_print, seconds));
    boost::shared_future<int> fut(task->get_future());
    pending_data.push_back(fut);
    io_service.post(boost::bind(&task_t::operator(), task));
}


int main() {

    boost::asio::io_service io_service;
    boost::thread_group threads;
    boost::asio::io_service::work work(io_service);

    for (int i = 0; i < boost::thread::hardware_concurrency(); ++i)
    {
        threads.create_thread(boost::bind(&boost::asio::io_service::run,
                                          &io_service));
    }
    std::vector<boost::shared_future<int> > pending_data; // vector of futures

    sleep_print(2);

    push_job(3, io_service, pending_data);
    push_job(4, io_service, pending_data);
    boost::wait_for_all(pending_data.begin(), pending_data.end());

    push_job(3, io_service, pending_data);
    push_job(4, io_service, pending_data);
    push_job(5, io_service, pending_data);
    boost::wait_for_all(pending_data.begin(), pending_data.end());

    return 0;
}