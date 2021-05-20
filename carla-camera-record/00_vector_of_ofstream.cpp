#include <iostream>
#include <fstream>
#include <vector>


int main() {
    using namespace std;

    vector<ofstream> outs(2);
    for(int i = 0; i < 2; i++){
        outs.emplace_back(std::to_string(i) + ".txt", std::ios::out);
    }

    for(auto& out: outs) {
        out.write("abc\n", 4);
        out.write("123\n", 4);
    }

    for(auto& out: outs)
        out.close();
}