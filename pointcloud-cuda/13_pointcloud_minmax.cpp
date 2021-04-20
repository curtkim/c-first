#include <iostream>

#include <tinyply.h>
#include "11_example_utils.hpp"

/*
int new_count = 0;
int new_bytes = 0;

void * operator new(size_t size)
{
    new_count++;
    new_bytes += size;
    std::cout << "New operator size=" << size << std::endl;
    void * p = malloc(size);
    return p;
}

void operator delete(void * p)
{
    std::cout << "Delete operator " << std::endl;
    free(p);
}
*/

using namespace tinyply;

struct float4 { float x, y, z, i; };

std::vector<float4> read_ply_file(const std::string & filepath)
{
    std::ifstream file_stream(filepath, std::ios::binary);
    if (!file_stream || file_stream.fail())
        throw std::runtime_error("file_stream failed to open " + filepath);

    file_stream.seekg(0, std::ios::end);
    const float size_mb = file_stream.tellg() * float(1e-6);
    file_stream.seekg(0, std::ios::beg);

    PlyFile file;
    file.parse_header(file_stream);

    std::shared_ptr<PlyData> data;

    try {
        data = file.request_properties_from_element("vertex", { "x", "y", "z", "I" });
    }
    catch (const std::exception & e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    file.read(file_stream);

    float4* first = reinterpret_cast<float4 *>(data->buffer.get());
    float4* last = first + data->count;
    return std::vector<float4>(first, last);
}


int main() {
    std::vector<float4> data = read_ply_file("../../pc_000077.ply");
    std::cout << data.size() << " total vertices "<< std::endl;

//    for(auto& pt : data)
//        std::cout << pt.x << "\n";
}