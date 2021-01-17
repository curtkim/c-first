#include "main_window.hpp"
#include "qt_execution_context.hpp"
#include <QApplication>
#include <QTextEdit>
#include <iostream>

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    qt_execution_context exec_context;

    net::io_context ioctx;

    // like the old asio work_guard
    auto io_exec = net::any_io_executor(net::require(
        ioctx.get_executor(), net::execution::outstanding_work.tracked));

    std::thread t([&ioctx] {
      ioctx.run();
    });

    main_window w(ioctx.get_executor());
    w.show();

    app.exec();
    io_exec = net::any_io_executor();
    if (t.joinable())
      t.join();

    return 0;
}
