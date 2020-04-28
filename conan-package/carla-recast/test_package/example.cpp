#include <iostream>

#include "Recast.h"

int main()
{

    int one = 1;
    int two = 2;
    rcSwap(one, two);

    std::cout << one << " " << two << std::endl;
    
    return 0;
}