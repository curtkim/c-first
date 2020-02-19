#include <iostream>
#include <fstream>
#include "proto/sensor.pb.h"    // 'proto' 경로를 넣어주어야 한다.

int main() {
    Sensor sensor;
    sensor.set_name("Laboratory");
    sensor.set_temperature(23.4);
    sensor.set_humidity(68);
    sensor.set_door(Sensor_SwitchLevel_OPEN);

    std::cout << "Serialize " << sensor.name() << " to sensor.pb\n";
    std::ofstream ofs("sensor.pb");
    sensor.SerializeToOstream(&ofs);

    return 0;
}