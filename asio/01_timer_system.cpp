#include <iostream>
#include <asio.hpp>

int main() {
    asio::io_context io_context;

    asio::system_timer timer(io_context,std::chrono::system_clock::now() + std::chrono::seconds(2));
    timer.async_wait([](const std::error_code & /*error*/) { std::cout << "timeout\n"; });

    std::cout << "---before run" << std::endl;
    io_context.run();
    std::cout << "---after run" << std::endl;

    return 0;
}