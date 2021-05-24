#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"
#include <iostream>

struct my_type {
    int i;

    template<typename OStream>
    friend OStream &operator<<(OStream &os, const my_type &c) {
        return os << "[my_type i=" << c.i << "]";
    }
};

void user_defined_example() {
    auto my_value = my_type{14};
    spdlog::info("Some info message with arg: {}", my_value);

    std::cout << my_value << std::endl;
}

int main() {
    user_defined_example();
}