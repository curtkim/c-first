#include <boost/asio/io_service.hpp>
#include <boost/asio/posix/stream_descriptor.hpp>
#include <boost/asio/write.hpp>
#include <boost/system/error_code.hpp>
#include <iostream>
#include <unistd.h>

using namespace boost::asio;

int main()
{
    io_service ioservice;

    posix::stream_descriptor stream{ioservice, STDOUT_FILENO};
    auto handler = [](const boost::system::error_code&, std::size_t size) {
        std::cout << ", world!\n";
        std::cout << size << " printed" << std::endl;
    };
    async_write(stream, buffer("Hello"), handler);
    async_write(stream, buffer("Hi"), handler);

    ioservice.run();
}