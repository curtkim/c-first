#include <cassert>
#include <iostream>
#include <string>

#include <string_view>

// Overloaded the global operator
void* operator new(std::size_t count){
    std::cout << "   " << count << " bytes" << std::endl;
    return malloc(count);
}

void getString(const std::string& str){}

void getStringView(std::string_view strView){}

int main() {

    std::cout << "1) std::string" << std::endl;
    std::string large = "0123456789-123456789-123456789-123456789";
    std::string substr = large.substr(10);

    std::cout << "2) std::string_view" << std::endl;
    std::string_view largeStringView{large.c_str(), large.size()};
    largeStringView.remove_prefix(10);
    assert(substr == largeStringView);

    std::cout << "3) getString" << std::endl;
    getString(large);
    getString("0123456789-123456789-123456789-123456789");
    const char message []= "0123456789-123456789-123456789-123456789";
    getString(message);

    std::cout << "4) getStringView" << std::endl;
    getStringView(large);
    getStringView("0123456789-123456789-123456789-123456789");
    getStringView(message);
}