cmake_minimum_required(VERSION 3.16)
project(ipopt-first)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(nlp-example nlp-example.cpp MyNLP.cpp)
target_link_libraries(nlp-example ${CONAN_LIBS}) #ipopt

add_executable(hs071_main hs071_main.cpp hs071_nlp.cpp)
target_link_libraries(hs071_main ${CONAN_LIBS}) #ipopt


message(status "IPOPT_CFLAGS: " ${IPOPT_CFLAGS})
message(status "IPOPT_CFLAGS_OTHER: " ${IPOPT_CFLAGS_OTHER})
set(CMAKE_CXX_FLAGS "-DHAVE_CSTDDEF -DHAVE_MPI_INITIALIZED")

add_executable(example example.cpp)
target_link_libraries(example ${CONAN_LIBS}) #ipopt
