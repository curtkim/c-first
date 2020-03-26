#include <iostream>

void load(int a){
    if( a % 2 == 0)
        return;
    else
        throw std::invalid_argument("argument="+std::to_string(a));
}

int main()
{
    load(0);

    try {
        load(1);
    }
    catch (std::exception& e){
        std::cout << "exception catch " << e.what() << std::endl;
    }

    load(3);
    
    return 0;
}
