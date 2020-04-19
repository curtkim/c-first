cmake_minimum_required(VERSION 3.15)
project(ncurse)

foreach(APP hello win win_border tt simple_color simple_key printw_example mouse_menu key_code)
  add_executable(${APP} ${APP}.cpp)
  target_compile_features(${APP} PRIVATE cxx_std_11)
  target_link_libraries(${APP} ncurses)
endforeach()

foreach(APP form_simple form_win panel_simple menu_simple)
  add_executable(${APP} ${APP}.cpp)
  target_compile_features(${APP} PRIVATE cxx_std_11)
  target_link_libraries(${APP} ncurses form panel menu)
endforeach()
