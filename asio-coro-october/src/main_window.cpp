#include "main_window.hpp"
#include "test_widget.hpp"
#include <QMdiArea>
#include <QMenuBar>

main_window::main_window(net::io_context::executor_type const &io_exec,
                         QWidget *parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
    , io_exec_(io_exec)
{
    mdi_area = new QMdiArea();
    mdi_area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    mdi_area->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setCentralWidget(mdi_area);

    widgets_menu = menuBar()->addMenu(tr("&Widgets"));
    auto action = widgets_menu->addAction(tr("&Another"));
    connect(action, &QAction::triggered, this, &main_window::make_new_widget);
}

void main_window::make_new_widget()
{
    auto view = new test_widget(io_exec_);
    view->setAttribute(Qt::WA_DeleteOnClose);
    mdi_area->addSubWindow(view);
    view->show();
}