#ifndef MODERN_CPP_TESTOBJECT_H
#define MODERN_CPP_TESTOBJECT_H

#include <string>
#include <iostream>

using namespace std;

class TestObject {
public:

    int n1 = 100;
    std::string s1 = "test";

    void print() {
        std::cout << n1 << ", " << s1 << std::endl;
    }
};


#endif //MODERN_CPP_TESTOBJECT_H
