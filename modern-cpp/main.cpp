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

void test_lambda() {
    [] { std::cout << "Hello World!" << std::endl; } ();

    auto func = [] { std::cout << "Hello World!" << std::endl; };
    func();

    auto func2 = [] ( int n ) { std::cout << "I have " << n << " girl friends" << std::endl; };
    func2 ( 3 );
}

void test_lambda_return() {
    auto func1 = [] { return 3.14; };
    auto func2 = [] (float f) { return f; };
    auto func3 = [] () -> float { return 3.14; };

    float f1 = func1();
    float f2 = func2( 3.14f );
    float f3 = func3();
}

void test_lambda_each() {
    std::vector<int> v1;
    v1.emplace_back( 10 );
    v1.emplace_back( 20 );
    v1.emplace_back( 30 );

    std::for_each ( v1.begin(), v1.end(),
            [] ( int n ) { std::cout << n << std::endl; }
            );
}

std::function < void() > funcLambda() {
    std::string str("This is Lambda!");
    return [=] { std::cout << "What!?" << str << std::endl; };
}

void test_lambda_capture() {
    int x = 100;
    [&]() {std::cout << x << std::endl; x = 200; } ();
    std::cout << x << std::endl; // print x = 200
}

void test_lambda_capture2() {
    int x = 100;
    [=]() { std::cout << x << std::endl; } (); // print x = 100
    [=]() mutable { std::cout << x << std::endl; x = 200; }(); // x = 200
    std::cout << x << std::endl; // print x = 100
}

void test_generic_lambda() {
    // gerneric lambda
    auto sum = [](auto a, decltype(a) b) { return a + b; };

    int i = sum( 3, 4 );
    double d = sum ( 3.14, 2.77 );

    cout << i << " " << d << endl;
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

    test_lambda();
    test_lambda_return();
    test_lambda_each();

    auto func = funcLambda();
    func();
    funcLambda()();

    test_lambda_capture();
    test_lambda_capture2();
    test_generic_lambda();

    return 0;
}
