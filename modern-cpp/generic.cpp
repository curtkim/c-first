#include <iostream>

template <typename T1, typename T2>
auto compose(T1 a, T2 b) -> decltype(a + b){
    return a+b;
}

template <typename T1, typename T2>
auto compose2(T1 a, T2 b) { // "-> decltype(a + b)" 필요없다.
    return a+b;
}

using namespace std;

int main() {
    cout << compose(1, 2) << endl;
    cout << compose2(1, 2) << endl;
    return 0;
}
