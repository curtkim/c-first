#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <iostream>

boost::asio::io_service io_service;

void WorkerThread()
{
    std::cout << "Thread Start\n";
    // yield되는 것 같다. io_serivce.stop()이 호출될때 까지 기다린다.
    io_service.run();
    std::cout << "Thread Finish\n";
}

int main( int argc, char * argv[] )
{
    // 아래 부분이 없으면 io_service.run()에서 멈춰지지 않는다.
    boost::shared_ptr< boost::asio::io_service::work > work(
            new boost::asio::io_service::work( io_service )
    );

    std::cout << "Press [return] to exit." << std::endl;

    boost::thread_group worker_threads;
    for( int x = 0; x < 4; ++x )
    {
        worker_threads.create_thread( WorkerThread );
    }

    std::cin.get();
    io_service.stop();
    worker_threads.join_all();

    return 0;
}
