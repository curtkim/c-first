#include <array>
#include <iostream>

int main()
{
    // Creating and initializing an array of size 10.
    std::array<int , 10> arr = {1,2,3,4,5,6,7,8,9,10};

    // Random access operator [] to fetch any element from array
    int x = arr[2];

    std::cout<<"x(by []) = "<< x <<std::endl;

    // Accessing out of range elements using [] leads to undefined behaviour
    // i.e. it can give garbage value or crash
    x = arr[12];

    std::cout<<"x = "<< x <<std::endl;

    // Accessing element using at() function
    x = arr.at(2);

    std::cout<<"x(by at) = "<< x <<std::endl;

    // Accessing out of range elements using at() will throw exception
    try
    {
        x = arr.at(12);
    }
    catch (const std::out_of_range& exp)
    {
        std::cerr << "---";
        std::cerr << exp.what() << std::endl;
    }

    // Accessing elements from std::array object using std::get<>()
    x = std::get<2>(arr);

    // Accessing out of range elements using std::get<>() will throw error at compile time
    //x = std::get<12>(arr);

    std::cout<<"x(by tuple) = "<< x <<std::endl;

    return 0;
}