#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <iostream>

// dispatch는 바로 실행되고
// post는 dispatch된 handler가 끝나고 실행된다.
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

void Dispatch( int x )
{
    global_stream_lock.lock();
    std::cout << "[" << boost::this_thread::get_id() << "] "
              << __FUNCTION__ << " x = " << x << std::endl;
    global_stream_lock.unlock();
}

void Post( int x )
{
    global_stream_lock.lock();
    std::cout << "[" << boost::this_thread::get_id() << "] "
              << __FUNCTION__ << " x = " << x << std::endl;
    global_stream_lock.unlock();
}

void Run3( boost::shared_ptr< boost::asio::io_service > io_service )
{
    for( int x = 0; x < 3; ++x )
    {
        // Dispatched events can execute from the current worker thread
        // even if there are other pending events queued up
        io_service->dispatch( boost::bind( &Dispatch, x * 2 ) );

        // The posted events have to wait until the handler completes
        io_service->post( boost::bind( &Post, x * 2 + 1 ) );

        boost::this_thread::sleep( boost::posix_time::milliseconds( 1000 ) );
    }
}

int main( int argc, char * argv[] )
{
    boost::shared_ptr< boost::asio::io_service > io_service(
            new boost::asio::io_service
    );
    boost::shared_ptr< boost::asio::io_service::work > work(
            new boost::asio::io_service::work( *io_service )
    );

    global_stream_lock.lock();
    std::cout << "[" << boost::this_thread::get_id()
              << "] The program will exit when all work has finished." << std::endl;
    global_stream_lock.unlock();

    boost::thread_group worker_threads;
    for( int x = 0; x < 1; ++x )
    {
        worker_threads.create_thread( boost::bind( &WorkerThread, io_service ) );
    }

    io_service->post( boost::bind( &Run3, io_service ) );
    work.reset();
    worker_threads.join_all();

    return 0;
}