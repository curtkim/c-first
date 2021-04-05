cmake_minimum_required(VERSION 3.16)
project(lager-first)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
#find_package(SDL2_image REQUIRED)
find_package(SDL2_ttf REQUIRED)
message("${SDL2_LIBRARY}")
message("${SDL2_TTF_LIBRARY}")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

include_directories(.)

add_executable(counter_std counter/std/main.cpp)
target_link_libraries(counter_std ${CONAN_LIBS})
target_compile_features(counter_std PUBLIC cxx_std_17)

#add_executable(counter_ncurses counter/ncurses/main.cpp counter/ncurses/terminal.cpp)
#target_link_libraries(counter_ncurses ${CONAN_LIBS})
#target_compile_features(counter_ncurses PUBLIC cxx_std_17)

add_executable(counter_sdl2 counter/sdl2/main.cpp)
target_link_libraries(counter_sdl2 ${CONAN_LIBS} ${SDL2_LIBRARY} ${SDL2_TTF_LIBRARY})
target_compile_features(counter_sdl2 PUBLIC cxx_std_17)

add_executable(todo_imgui todo/imgui/main.cpp todo/todo.cpp)
target_link_libraries(todo_imgui ${CONAN_LIBS} ${SDL2_LIBRARY} ${SDL2_TTF_LIBRARY})
target_compile_features(todo_imgui PUBLIC cxx_std_17)

#add_executable(autopong autopong/sdl2/main.cpp autopong/autopong.cpp)
#target_link_libraries(autopong ${CONAN_LIBS})
#target_compile_features(autopong PUBLIC cxx_std_17)
