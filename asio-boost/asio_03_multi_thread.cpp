#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <iostream>


void WorkerThread( boost::shared_ptr< boost::asio::io_service > io_service )
{
    std::cout << "Thread Start\n";
    io_service->run();
    std::cout << "Thread Finish\n";
}

int main( int argc, char * argv[] )
{
    boost::shared_ptr< boost::asio::io_service > io_service(
            new boost::asio::io_service
    );
    boost::shared_ptr< boost::asio::io_service::work > work(
            new boost::asio::io_service::work( *io_service )
    );

    std::cout << "Press [return] to exit." << std::endl;

    boost::thread_group worker_threads;
    for( int x = 0; x < 4; ++x )
    {
        worker_threads.create_thread( boost::bind( &WorkerThread, io_service ) );
    }

    std::cin.get();
    io_service->stop();
    worker_threads.join_all();

    return 0;
}