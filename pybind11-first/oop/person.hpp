#pragma once

#include <string>

namespace oop {

struct Person {
    std::string name;
    int age;
};
std::string formatPerson(const Person& person);

}

