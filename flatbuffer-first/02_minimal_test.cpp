#include <cstdlib>
#include <iostream>

#include "flatbuffers/flatbuffers.h"

#ifndef FLATBUFFERS_HEADER_ONLY
  #include "flatbuffers/util.h"
#endif
// Test to validate Conan package generated

int main(int /*argc*/, const char * /*argv*/ []) {

  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<flatbuffers::String> offset = builder.CreateString("test");
  if (!offset.IsNull()) {
    std::cout << "Offset is " << offset.o << ".\n";
  } else {
    std::cout << "Offset is null.\n";
    return EXIT_FAILURE;
  }

#ifndef FLATBUFFERS_HEADER_ONLY
  const std::string filename("../conanbuildinfo.cmake");
  if (flatbuffers::FileExists(filename.c_str())) {
    std::cout << "File " << filename << " exists.\n";
  } else {
    std::cout << "File " << filename << " does not exist.\n";
    return EXIT_FAILURE;
  }
#endif

  return EXIT_SUCCESS;
}