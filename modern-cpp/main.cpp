#include <iostream>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>

#include "TestObject.h"

using namespace std;

void test_range_loop() {
    int NumberList[5] = { 1, 2, 3, 4, 5 };
    for( auto i : NumberList ) {
        cout << i << endl;
    }
}

void test_vector_range_loop() {
    int size = 5;
    std::vector<int> array(size);
    for(int i=0; i<size; ++i){
        array[i] = i;
    }

    for( auto i : array ) {
        std::cout << i << std::endl;
    }
}

void test_unordered_map() {
    unordered_map<string, int> umap;

    // inserting values by using [] operator
    umap["GeeksforGeeks"] = 10;
    umap["Practice"] = 20;
    umap["Contribute"] = 30;

    // Traversing an unordered map
    for (auto x : umap)
        cout << x.first << " " << x.second << endl;
}

enum ITEMTYPE : short {
    WEAPON,
    EQUIPMENT,
    GEM = 10,
    DEFENCE,
};

void test_enum() {
    short ItemType1 = WEAPON;
    short ItemType2 = ITEMTYPE::DEFENCE;

    cout << ItemType1 << " " << ItemType2 << endl;
}

// https://redforce01.tistory.com/78?category=695802
void test_field_init() {
    TestObject a;
    a.print();

    TestObject b {1, "Test"};
    b.print();
}

void test_initializer_list() {
    std::vector<int> v { 1, 2, 3 };
    for(auto i : v)
        cout << i << ' ';
    cout << endl;
}

constexpr double pow( double x, size_t y) {
    return y != 1 ? x * pow(x, y - 1) : x;
}


int main()
{
    test_range_loop();
    test_vector_range_loop();
    test_unordered_map();
    test_enum();
    test_field_init();
    test_initializer_list();

    cout << pow(2.0, 2) << " " << pow(3.0, 3) << endl;
    return 0;
}
