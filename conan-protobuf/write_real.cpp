#include <iostream>

#include <sys/stat.h>
#include <fcntl.h>

#include <unistd.h>
#include <string>

#include "sensor.pb.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>


bool SetProtoToASCIIFile(const google::protobuf::Message &message,
                         int file_descriptor) {
    using google::protobuf::TextFormat;
    using google::protobuf::io::FileOutputStream;
    using google::protobuf::io::ZeroCopyOutputStream;
    if (file_descriptor < 0) {
        std::cout << "Invalid file descriptor.";
        return false;
    }
    ZeroCopyOutputStream *output = new FileOutputStream(file_descriptor);
    bool success = TextFormat::Print(message, output);
    delete output;
    close(file_descriptor);
    return success;
}

bool SetProtoToASCIIFile(const google::protobuf::Message &message,
                         const std::string &file_name) {
    int fd = open(file_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
    if (fd < 0) {
        std::cout << "Unable to open file " << file_name << " to write.";
        return false;
    }
    return SetProtoToASCIIFile(message, fd);
}

bool GetProtoFromASCIIFile(const std::string &file_name,
                           google::protobuf::Message *message) {

    using google::protobuf::TextFormat;
    using google::protobuf::io::FileInputStream;
    using google::protobuf::io::ZeroCopyInputStream;

    int file_descriptor = open(file_name.c_str(), O_RDONLY);
    if (file_descriptor < 0) {
        std::cout << "Failed to open file " << file_name << " in text mode.";
        // Failed to open;
        return false;
    }

    ZeroCopyInputStream *input = new FileInputStream(file_descriptor);
    bool success = TextFormat::Parse(input, message);
    if (!success) {
        std::cout << "Failed to parse file " << file_name << " as text proto.";
    }
    delete input;
    close(file_descriptor);
    return success;
}

int main() {
    Sensor sensor;
    sensor.set_name("Laboratory");
    sensor.set_temperature(23.4);
    sensor.set_humidity(68);
    sensor.set_door(Sensor_SwitchLevel_OPEN);

    std::string filename = "sensor.pb.txt";
    SetProtoToASCIIFile(sensor, filename);
    Sensor sensor2;
    GetProtoFromASCIIFile(filename, &sensor2);

    std::cout << sensor.name() << std::endl;

    return 0;
}