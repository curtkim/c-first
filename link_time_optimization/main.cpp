#include <vector>
#include <iostream>

#include "lib1.h"
#include "lib2.h"

int main(){
    auto const v = std::vector<int> {1,2,7,4,1,7};

    std::cout << sum(v) << std::endl;
    std::cout << count(v, 7) << std::endl;
    
    return 0;
}