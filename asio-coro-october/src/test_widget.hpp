#pragma once
#include "config.hpp"
#include "has_guarded_executor.hpp"
#include <QTextEdit>

class test_widget
    : public QTextEdit
    , public has_guarded_executor
{
    Q_OBJECT
public:
    test_widget(net::io_context::executor_type const &ioexec,
                QWidget *parent = nullptr);

private:
    void showEvent(QShowEvent *event) override;

    void hideEvent(QHideEvent *event) override;

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    net::awaitable<void> run_demo();

    void listen_for_stop(std::function<void()> slot);

    void stop_all();

    net::io_context::executor_type ioexec_;
    std::vector<std::function<void()>> stop_signals_;
    bool stopped_ = false;
};
