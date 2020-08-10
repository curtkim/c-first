#include <iostream>
#include <memory>

#include <vector>
#include <algorithm>

using namespace std;

void vector_iterate() {
    cout << "=== " << __FUNCTION__ << endl;
    vector<int> ar = { 1, 2, 3, 4, 5 };

    // Declaring iterator to a vector
    vector<int>::iterator ptr;

    // Displaying vector elements using begin() and end()
    cout << "The vector elements are : ";
    for (ptr = ar.begin(); ptr < ar.end(); ptr++)
        cout << *ptr << " ";
    cout << endl;
}

void vector_contains() {
    cout << "=== " << __FUNCTION__ << endl;
    vector<string> v = { "1", "2", "3" };

    bool contains = std::find(v.begin(), v.end(), "3") != v.end();
    cout << (contains ? "contains" : "no contains") << endl;
}

void vector_distance() {
    cout << "=== " << __FUNCTION__ << endl;
    vector<string> v = { "1", "2", "3" };

    std::vector<string>::iterator it = std::find(v.begin(), v.end(), "3");
    int index = std::distance(v.begin(), it);
    cout << "index=" << index << endl;
    cout << v[index] << endl;
}

int main() {
    vector_iterate();
    vector_contains();
    vector_distance();
    return 0;
}