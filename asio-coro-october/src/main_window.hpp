#include "config.hpp"
#include <QMainWindow>

class QMdiArea;

class main_window : public QMainWindow
{
    Q_OBJECT

public:
    main_window(net::io_context::executor_type const &io_exec,
                QWidget *parent = nullptr,
                Qt::WindowFlags flags = Qt::WindowFlags());

    void
    make_new_widget();

private:
    net::io_context::executor_type io_exec_;
    QMenu *widgets_menu = nullptr;
    QMdiArea *mdi_area = nullptr;
};