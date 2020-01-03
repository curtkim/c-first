#include <boost/asio.hpp>
#include <iostream>

void ex1(){
    boost::asio::io_service io_service;

    io_service.run();

    std::cout << "Do you reckon this line displays?" << std::endl;
}

void ex2() {
    boost::asio::io_service io_service;
    boost::asio::io_service::work work( io_service );

    io_service.run();

    std::cout << "Do you reckon this line displays?" << std::endl;
}

void ex3() {
    boost::asio::io_service io_service;

    for( int x = 0; x < 42; ++x )
    {
        io_service.poll();
        std::cout << "Counter: " << x << std::endl;
    }
}

void ex4(){
    boost::asio::io_service io_service;
    boost::asio::io_service::work work( io_service );

    for( int x = 0; x < 42; ++x )
    {
        io_service.poll();
        std::cout << "Counter: " << x << std::endl;
    }
}

void ex5(){
    boost::asio::io_service io_service;
    boost::shared_ptr< boost::asio::io_service::work > work(
            new boost::asio::io_service::work( io_service )
    );

    work.reset();

    io_service.run();

    std::cout << "Do you reckon this line displays?" << std::endl;
}

int main() {
    ex5();
    return 0;
}
