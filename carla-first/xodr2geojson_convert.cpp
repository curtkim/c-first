#include <fstream>
#include <iostream>

#include "xodr2geojson.hpp"


int main() {

  std::ifstream t("Town01.xodr");
  std::stringstream buffer;
  buffer << t.rdbuf();
  std::string xodr = buffer.str();
  //std::cout << xodr << std::endl;

  XodrGeojsonConverter converter;
  auto geojson = converter.Convert(xodr);
  std::cout << geojson << std::endl;

  return 0;
}

