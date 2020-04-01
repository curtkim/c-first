#include <iostream>
#include <range/v3/all.hpp> // get everything
#include <string>
using std::cout;

auto is_six = [](int i) { return i == 6; };


void for_loop()
{
  std::string s{"hello"};
  ranges::for_each(s, [](char c) { cout << c << ' '; });
  cout << '\n';
}

void any_all_none() {

  std::vector<int> v{6, 2, 3, 4, 5, 6};

  cout << "vector any_of is_six: " << ranges::any_of(v, is_six) << '\n';
  cout << "vector all_of is_six: " << ranges::all_of(v, is_six) << '\n';
  cout << "vector none_of is_six: " << ranges::none_of(v, is_six) << '\n';
}

void find() {
  std::vector<int> v{6, 2, 6, 4, 6, 1};
  {
    auto i = ranges::find(v, 6); // 1 2 3 4 5 6
    cout << "*i: " << *i << '\n';
  }

  {
    auto i = ranges::find(v, 10); // 1 2 3 4 5 6
    if (i == ranges::end(v)) {
      cout << "didn't find 10\n";
    }
  }

  {
    auto i = ranges::find_if(v, is_six);
    if(i != ranges::end(v))
    {
      cout << "*i: " << *i << '\n';
    }
  }
}

int main(int argc, char** argv)
{
  find();
  any_all_none();
  for_loop();

  return 0;
}
