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

    while( true )
    {
        try
        {
            boost::system::error_code ec;
            io_service->run( ec );
            if( ec )
            {
                global_stream_lock.lock();
                std::cout << "[" << boost::this_thread::get_id()
                          << "] Error: " << ec << std::endl;
                global_stream_lock.unlock();
            }
            break;
        }
        catch( std::exception & ex )
        {
            global_stream_lock.lock();
            std::cout << "[" << boost::this_thread::get_id()
                      << "] Exception: " << ex.what() << std::endl;
            global_stream_lock.unlock();
        }
    }

    global_stream_lock.lock();
    std::cout << "[" << boost::this_thread::get_id()
              << "] Thread Finish" << std::endl;
    global_stream_lock.unlock();
}

void TimerHandler(
    const boost::system::error_code & error,
    boost::shared_ptr< boost::asio::deadline_timer > timer,
    boost::shared_ptr< boost::asio::io_service::strand > strand
)
{
    if( error )
    {
        global_stream_lock.lock();
        std::cout << "[" << boost::this_thread::get_id()
                  << "] Error: " << error << std::endl;
        global_stream_lock.unlock();
    }
    else
    {
        std::cout << "[" << boost::this_thread::get_id()
                  << "] TimerHandler " << std::endl;

        timer->expires_from_now( boost::posix_time::seconds( 1 ) );
        timer->async_wait(strand->wrap( boost::bind( &TimerHandler, _1, timer, strand ) )
        );
    }
}

void PrintNum( int x )
{
    std::cout << "[" << boost::this_thread::get_id()
              << "] x: " << x << std::endl;
    boost::this_thread::sleep( boost::posix_time::milliseconds( 1000 ) );
}

int main( int argc, char * argv[] )
{
    boost::shared_ptr< boost::asio::io_service > io_service(
        new boost::asio::io_service
    );
    boost::shared_ptr< boost::asio::io_service::work > work(
        new boost::asio::io_service::work( *io_service )
    );
    boost::shared_ptr< boost::asio::io_service::strand > strand(
        new boost::asio::io_service::strand( *io_service )
    );

    global_stream_lock.lock();
    std::cout << "[" << boost::this_thread::get_id()
              << "] Press [return] to exit." << std::endl;
    global_stream_lock.unlock();

    boost::thread_group worker_threads;
    for( int x = 0; x < 2; ++x )
    {
        worker_threads.create_thread( boost::bind( &WorkerThread, io_service ) );
    }

    // timer
    boost::shared_ptr< boost::asio::deadline_timer > timer(
        new boost::asio::deadline_timer( *io_service )
    );
    timer->expires_from_now( boost::posix_time::seconds( 1 ) );
    timer->async_wait(
        strand->wrap( boost::bind( &TimerHandler, _1, timer, strand ) )
    );

    // strand->post
    boost::this_thread::sleep( boost::posix_time::seconds( 5 ) );

    strand->post( boost::bind( &PrintNum, 1 ) );
    strand->post( boost::bind( &PrintNum, 2 ) );
    strand->post( boost::bind( &PrintNum, 3 ) );
    strand->post( boost::bind( &PrintNum, 4 ) );
    strand->post( boost::bind( &PrintNum, 5 ) );

    std::cin.get();
    io_service->stop();
    worker_threads.join_all();

    return 0;
}