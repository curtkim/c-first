#include <boost/thread.hpp>
#include <iostream>

void F1()
{
    std::cout << __FUNCTION__ << std::endl;
}

void F2( int i, float f )
{
    std::cout << "i: " << i << std::endl;
    std::cout << "f: " << f << std::endl;
}

class MyClass
{
public:
    void F3( int i, float f )
    {
        std::cout << "i: " << i << std::endl;
        std::cout << "f: " << f << std::endl;
    }
};

int main( int argc, char * argv[] )
{
    boost::bind( &F1 );
    boost::bind( &F1 )();
    boost::bind( &F2, 42, 3.14f )();

    MyClass c;
    boost::bind( &MyClass::F3, &c, 42, 3.14f )();

    return 0;
}
