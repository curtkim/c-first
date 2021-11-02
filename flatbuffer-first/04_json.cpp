#include <iostream>
#include <string>

#include "flatbuffers/idl.h"

const std::string input_json_data = R"(
{
  firstName: "somename",
  lastName: "someothername",
  age: 21
}
)";


int main()
{
  std::string schemafile;
  bool ok = flatbuffers::LoadFile("../../sample.fbs", false, &schemafile);
  if (!ok) {
    std::cout << "load file failed!" << std::endl;
    return -1;
  }

  flatbuffers::Parser parser;
  parser.Parse(schemafile.c_str());
  if (!parser.Parse(input_json_data.c_str())) {
    std::cout << "flatbuffers parser failed with error : " << parser.error_ << std::endl;
    return -1;
  }

  std::string jsongen;
  if (!GenerateText(parser, parser.builder_.GetBufferPointer(), &jsongen)) {
    std::cout << "Couldn't serialize parsed data to JSON!" << std::endl;
    return -1;
  }

  std::cout << "intput json"
    << input_json_data
    << "output json\n"
    << jsongen;

  return 0;
}
