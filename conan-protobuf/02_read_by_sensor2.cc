#include <iostream>
#include <fstream>
#include "sensor2.pb.h"

int main() {
    Sensor2 sensor;
    std::ifstream ifs("sensor.data");
    if( !sensor.ParseFromIstream(&ifs)) {
        std::cout << "failed" << std::endl;
    }

    std::cout << sensor.name() << std::endl;

    return 0;
}