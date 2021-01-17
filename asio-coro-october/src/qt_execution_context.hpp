#pragma once

#include "config.hpp"
#include "qt_work_event.hpp"
#include <QApplication>
#include <boost/noncopyable.hpp>
#include <cassert>

struct qt_execution_context
    : net::execution_context
    , boost::noncopyable
{
    qt_execution_context(QObject *target = qApp);

    ~qt_execution_context();

    template<class F>
    void
    post(F f);

    static qt_execution_context &
    singleton();

    struct filter : QObject
    {
        filter();
        auto
        eventFilter(QObject *, QEvent *event) -> bool override;
    };

private:
    static qt_execution_context *instance_;
    QObject *target_;
    filter filter_;
};

// impl

template<class F>
void
qt_execution_context::post(F f)
{
    // c++20 auto template deduction
    auto event = new basic_qt_work_event(std::move(f));
    QApplication::postEvent(target_, event);
}
