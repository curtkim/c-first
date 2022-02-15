#pragma once

#include <klein/klein.hpp>
#include <fmt/format.h>

void print(const std::string_view name, const kln::point& p){
    fmt::print("{} = ({}, {}, {})\n", name, p.x(), p.y(), p.z());
}

void print(const kln::mat4x4& mat){
    fmt::print("{} {} {} {}\n", mat.data[0], mat.data[4], mat.data[8], mat.data[12]);
    fmt::print("{} {} {} {}\n", mat.data[1], mat.data[5], mat.data[9], mat.data[13]);
    fmt::print("{} {} {} {}\n", mat.data[2], mat.data[6], mat.data[10], mat.data[14]);
    fmt::print("{} {} {} {}\n", mat.data[3], mat.data[7], mat.data[11], mat.data[15]);
}
void print(const kln::mat3x4& mat){
    fmt::print("{} {} {} {}\n", mat.data[0], mat.data[4], mat.data[8], mat.data[12]);
    fmt::print("{} {} {} {}\n", mat.data[1], mat.data[5], mat.data[9], mat.data[13]);
    fmt::print("{} {} {} {}\n", mat.data[2], mat.data[6], mat.data[10], mat.data[14]);
}


float rad2deg(float rad){
    return rad * (180.0/M_PI);
}
