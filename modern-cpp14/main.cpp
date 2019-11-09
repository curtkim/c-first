#include <iostream>
#include <memory>
#include <typeinfo>

#include <iterator>
#include <vector>
#include <algorithm>

using namespace std;


void add(std::shared_ptr<int> p){
  // reference가 증가한다.
  cout << "p use_count: " << p.use_count() << endl;
  cout << p.get() + 1 << endl;
}

void test_shared_ptr() {
  cout << "=== " << __FUNCTION__ << endl;
  std::shared_ptr<int> p1{new int{1}};
  std::cout << *p1 << '\n';

  add(p1);
  cout << "p1 use_count: " << p1.use_count() << endl;

  std::shared_ptr<int> p2{p1};
  p1.reset(new int{2});
  std::cout << "p1:" << *p1.get() << " use_count:" << p1.use_count() << '\n';
  std::cout << "p2:" << *p2.get() << '\n';

  p1.reset();
  std::cout << "p1 use_count:" << p1.use_count() << '\n';

  std::cout << std::boolalpha << static_cast<bool>(p1) << '\n';
  std::cout << std::boolalpha << static_cast<bool>(p2) << '\n';
}

void typeinfo()
{
  cout << "=== " << __FUNCTION__ << endl;
  auto p1 = std::make_shared<int>(1);
  int a = 1;
  float b = 2.0;
  std::cout << "typeid" <<  typeid(p1).name() << '\n';
  std::cout << "typeid" <<  typeid(a).name() << '\n';
  std::cout << "typeid" <<  typeid(b).name() << '\n';
}

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

  test_shared_ptr();
  typeinfo();
  vector_iterate();
  vector_contains();
  vector_distance();
  return 0;
}
