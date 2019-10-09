#include <tuple>
#include <iostream>
#include <string>
#include <stdexcept>
#include <map>

using namespace std;

std::tuple<double, char, std::string> get_student(int id)
{
    if (id == 0) return std::make_tuple(3.8, 'A', "Lisa Simpson");
    if (id == 1) return std::make_tuple(2.9, 'C', "Milhouse Van Houten");
    if (id == 2) return std::make_tuple(1.7, 'D', "Ralph Wiggum");
    throw std::invalid_argument("id");
}

void test_structured_binding() {
    auto student0 = get_student(0);
    std::cout << "ID: 0, "
              << "GPA: " << std::get<0>(student0) << ", "
              << "grade: " << std::get<1>(student0) << ", "
              << "name: " << std::get<2>(student0) << '\n';

    double gpa1;
    char grade1;
    std::string name1;
    std::tie(gpa1, grade1, name1) = get_student(1);
    std::cout << "ID: 1, "
              << "GPA: " << gpa1 << ", "
              << "grade: " << grade1 << ", "
              << "name: " << name1 << '\n';

    // C++17 structured binding:
    auto [ gpa2, grade2, name2 ] = get_student(2);
    std::cout << "ID: 2, "
              << "GPA: " << gpa2 << ", "
              << "grade: " << grade2 << ", "
              << "name: " << name2 << '\n';
}

void test_map() {
    std::map<std::string, int> mapOfWords;
    // Inserting data in std::map
    mapOfWords.insert(std::make_pair("earth", 1));
    mapOfWords.insert(std::make_pair("moon", 2));
    mapOfWords["sun"] = 3;
    // Will replace the value of already added key i.e. earth
    mapOfWords["earth"] = 4;

    // iterator
    std::map<std::string, int>::iterator it = mapOfWords.begin();
    while(it != mapOfWords.end())
    {
        std::cout<<it->first<<" :: "<<it->second<<std::endl;
        it++;
    }
    // Check if insertion is successful or not
    if(mapOfWords.insert(std::make_pair("earth", 1)).second == false)
    {
        std::cout<<"Element with key 'earth' not inserted because already existed"<<std::endl;
    }

    // Searching element in std::map by key.
    if(mapOfWords.find("sun") != mapOfWords.end())
        std::cout<<"word 'sun' found"<<std::endl;
    if(mapOfWords.find("mars") == mapOfWords.end())
        std::cout<<"word 'mars' not found"<<std::endl;
}

void test_map2() {
    const std::map<std::string, std::string> capitals {
            { "Poland", "Warsaw"},
            { "USA", "Washington"},
            { "France", "Paris"},
            { "UK", "London"},
            { "Germany", "Berlin"}
    };

    for (const auto & [k,v] : capitals)
    {
        cout << k << " " << v << endl;
    }
}

void test_init_if() {
    const std::string myString = "My Hello World Wow";

    if (const auto it = myString.find("Hello"); it != std::string::npos)
        std::cout << it << " Hello\n";

    if (const auto it = myString.find("World"); it != std::string::npos)
        std::cout << it << " World\n";
}

struct MyClass
{
    inline static const int sValue = 777;
};

template<typename ...Args> auto sum2(Args ...args)
{
    return (args + ...);
}


int main() {

    test_structured_binding();
    test_map();
    test_map2();

    test_init_if();

    cout << MyClass::sValue << endl;
    cout << sum2(1, 2, 3, 4, 5) << "\n";

    return 0;
}