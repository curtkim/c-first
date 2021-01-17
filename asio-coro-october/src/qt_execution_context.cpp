#include "qt_execution_context.hpp"

qt_execution_context* qt_execution_context::instance_ = nullptr;

qt_execution_context::qt_execution_context(QObject *target)
    : target_(target)
    , filter_()
{
    target_->installEventFilter(&filter_);
    instance_ = this;
}

qt_execution_context::~qt_execution_context()
{
    target_->removeEventFilter(&filter_);
}

qt_execution_context & qt_execution_context::singleton()
{
    assert(instance_);
    return *instance_;
}

qt_execution_context::filter::filter() : QObject(nullptr) {}

auto qt_execution_context::filter::eventFilter(QObject *, QEvent *event) ->bool
{
    if (event->type() == qt_work_event_base::generated_type())
    {
        auto p = static_cast<qt_work_event_base*>(event);
        p->accept();
        p->invoke();
        return true;
    }
    else
        return false;
}

