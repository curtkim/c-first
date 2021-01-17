cmake_minimum_required(VERSION 3.15)
project(asio-coro-october)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -v")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
find_package(Threads)

add_executable(main src/main.cpp
        src/main_window.cpp
        src/message_service.cpp
        src/qt_execution_context.cpp
        src/test_widget.cpp)
target_compile_options(main PRIVATE -fcoroutines)
target_compile_definitions(main PUBLIC ASIO_HAS_CO_AWAIT)
target_compile_definitions(main PUBLIC ASIO_HAS_STD_COROUTINE)
target_link_libraries(main ${CONAN_LIBS} pthread Qt5::Widgets Qt5::Core Threads::Threads)
target_include_directories(main PUBLIC src)
target_compile_features(main PUBLIC cxx_std_20)
