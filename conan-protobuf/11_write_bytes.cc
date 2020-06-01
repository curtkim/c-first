#include <iostream>
#include <fstream>
#include "info.pb.h"

int main() {
    Info info;
    char * data = "1234567890";
    std::cout << strlen(data) << std::endl;

    info.set_data(data, 10);
    info.PrintDebugString();

    std::cout << info.mutable_unknown_fields()->field_count() << std::endl;
    std::cout << info.ByteSize() << std::endl;

    std::ofstream ofs("info.data");
    info.SerializeToOstream(&ofs);
    return 0;
}