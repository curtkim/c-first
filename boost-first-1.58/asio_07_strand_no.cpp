#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <iostream>


boost::mutex global_stream_lock;

void WorkerThread( boost::shared_ptr< boost::asio::io_service > io_service )
{
    global_stream_lock.lock();
    std::cout << "[" << boost::this_thread::get_id()
              << "] Thread Start" << std::endl;
    global_stream_lock.unlock();

    io_service->run();

    global_stream_lock.lock();
    std::cout << "[" << boost::this_thread::get_id()
              << "] Thread Finish" << std::endl;
    global_stream_lock.unlock();
}

void PrintNum( int x )
{
    std::cout << "[" << boost::this_thread::get_id()
              << "] x: " << x << std::endl;
}

int main( int argc, char * argv[] )
{
    boost::shared_ptr< boost::asio::io_service > io_service(
            new boost::asio::io_service
    );
    boost::shared_ptr< boost::asio::io_service::work > work(
            new boost::asio::io_service::work( *io_service )
    );
    boost::asio::io_service::strand strand( *io_service );

    global_stream_lock.lock();
    std::cout << "[" << boost::this_thread::get_id()
              << "] The program will exit when all work has finished." << std::endl;
    global_stream_lock.unlock();

    boost::thread_group worker_threads;
    for( int x = 0; x < 2; ++x )
    {
        worker_threads.create_thread( boost::bind( &WorkerThread, io_service ) );
    }

    boost::this_thread::sleep( boost::posix_time::milliseconds( 1000 ) );

    //strand.post( boost::bind( &PrintNum, 1 ) );
    //strand.post( boost::bind( &PrintNum, 2 ) );
    //strand.post( boost::bind( &PrintNum, 3 ) );
    //strand.post( boost::bind( &PrintNum, 4 ) );
    //strand.post( boost::bind( &PrintNum, 5 ) );

    io_service->post( boost::bind( &PrintNum, 1 ) );
    io_service->post( boost::bind( &PrintNum, 2 ) );
    io_service->post( boost::bind( &PrintNum, 3 ) );
    io_service->post( boost::bind( &PrintNum, 4 ) );
    io_service->post( boost::bind( &PrintNum, 5 ) );

    work.reset();

    worker_threads.join_all();

    return 0;
}