#include <iostream>
#include <sstream>

#include <msgpack.hpp>

// hex_dump is not a part of msgpack-c. 
inline std::ostream& hex_dump(std::ostream& o, std::string const& v) {
    std::ios::fmtflags f(o.flags());
    o << std::hex;
    for (auto c : v) {
        o << "0x" << std::setw(2) << std::setfill('0') << (static_cast<int>(c) & 0xff) << ' ';
    }
    o.flags(f);
    return o;
}

void pack() {
    std::stringstream ss;
    msgpack::pack(ss, "compact");
    hex_dump(std::cout, ss.str()) << std::endl;
}

void pack_vector() {
    std::stringstream ss;
    std::vector<int> v { 1, 5, 8, 2, 6 };
    msgpack::pack(ss, v);
    hex_dump(std::cout, ss.str()) << std::endl;    
}

void pack_map() {
    std::stringstream ss;
    std::map<std::string, int> v { { "ABC", 5 }, { "DEFG", 2 } };
    msgpack::pack(ss, v);
    hex_dump(std::cout, ss.str()) << std::endl;    
}

// 잘 하고 있는 건지 잘 모르겠음
void pack_unpack() {
    std::stringstream ss;
    msgpack::pack(ss, "compact");
    hex_dump(std::cout, ss.str()) << std::endl;
    
    ss.seekg(0, ss.end);
    int size = ss.tellg();

    msgpack::object_handle result;
    unpack(result, ss.str().c_str(), size);

    std::string a = result.get().convert();
    std::cout << a << std::endl;
}

int main() {
    std::cout << MSGPACK_VERSION << std::endl;

    pack_unpack();

    /*
    pack();
    pack_vector();
    pack_map();
    */
}