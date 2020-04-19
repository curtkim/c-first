cmake_minimum_required(VERSION 3.15)
project(main)

add_definitions("-std=c++17")

add_executable(main main.cpp)
target_link_libraries(main ncurses)

add_executable(win win.cpp)
target_link_libraries(win ncurses)

add_executable(win_border win_border.cpp)
target_link_libraries(win_border ncurses)

