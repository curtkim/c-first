//
// Created by rhodges on 09/11/2020.
//

#include "test_widget.hpp"
#include "message_service.hpp"
#include "qt_executor.hpp"
#include <QString>
#include <string>

void test_widget::showEvent(QShowEvent *event)
{
    stopped_ = false;

    // start our coroutine
    net::co_spawn(get_executor(), [this] { return run_demo(); }, net::detached);

    QTextEdit::showEvent(event);
}

void test_widget::hideEvent(QHideEvent *event)
{
    // stop all coroutines
    stop_all();
    QWidget::hideEvent(event);
}

net::awaitable<void> test_widget::run_demo()
{
    using namespace std::literals;

    auto service = message_service(ioexec_);
    auto conn = co_await service.connect();

    auto done = false;

    listen_for_stop([&] {
        done = true;
        conn.disconnect();
        service.reset();
    });

    while (!done)
    {
        auto message = co_await conn.consume();
        this->setText(QString::fromStdString(message));
    }
    co_return;
}

void test_widget::listen_for_stop(std::function<void()> slot)
{
    if (stopped_)
        return slot();

    stop_signals_.push_back(std::move(slot));
}

void test_widget::stop_all()
{
    stopped_ = true;
    auto copy = std::exchange(stop_signals_, {});
    for (auto &slot : copy) slot();
}

void test_widget::closeEvent(QCloseEvent *event)
{
    stop_all();
    QWidget::closeEvent(event);
}

test_widget::test_widget(const boost::asio::io_context::executor_type &ioexec,
                         QWidget *parent)
    : QTextEdit(parent)
    , ioexec_(ioexec)
{}
