#include <iostream>
#include <fstream>
#include "info.pb.h"

int main() {
    Info info;
    char * data = "1234567890";
    std::cout << strlen(data) << std::endl;

    info.set_bytes(data, 10);
    info.PrintDebugString();

    std::cout << info.ByteSize() << std::endl;

    std::ofstream ofs("info.data");
    info.SerializeToOstream(&ofs);

    std::cout << "===================" << std::endl;

    char buffer[100];
    info.SerializeToArray(buffer, info.ByteSizeLong());
    Info info2;
    info2.ParseFromArray(buffer, info.ByteSizeLong());
    info2.PrintDebugString();

    return 0;
}