#include <iostream>
#include <memory>

#include <vector>

using namespace std;


void add(std::shared_ptr<int> p){
    // reference가 증가한다.
    cout << "p use_count: " << p.use_count() << endl;
    cout << p.get() + 1 << endl;
}

void test_shared_ptr() {
    cout << "=== " << __FUNCTION__ << endl;
    std::shared_ptr<int> p1{new int{1}};
    std::cout << *p1 << '\n';

    add(p1);
    cout << "p1 use_count: " << p1.use_count() << endl;

    std::shared_ptr<int> p2{p1};
    p1.reset(new int{2});
    std::cout << "p1:" << *p1.get() << " use_count:" << p1.use_count() << '\n';
    std::cout << "p2:" << *p2.get() << '\n';

    p1.reset();
    std::cout << "p1 use_count:" << p1.use_count() << '\n';

    std::cout << std::boolalpha << static_cast<bool>(p1) << '\n';
    std::cout << std::boolalpha << static_cast<bool>(p2) << '\n';
}

void typeinfo()
{
    cout << "=== " << __FUNCTION__ << endl;
    auto p1 = std::make_shared<int>(1);
    int a = 1;
    float b = 2.0;
    std::cout << "typeid" <<  typeid(p1).name() << '\n';
    std::cout << "typeid" <<  typeid(a).name() << '\n';
    std::cout << "typeid" <<  typeid(b).name() << '\n';
}

int main() {

    test_shared_ptr();
    typeinfo();

    return 0;
}