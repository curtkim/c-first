#include <iostream>
#include <list>

using namespace std;

//function for printing the elements in a list
void showlist(list <int> g)
{
  list <int> :: iterator it;
  for(it = g.begin(); it != g.end(); ++it)
    cout << '\t' << *it;
  cout << '\n';
}

int main() {

  const size_t FIXED_LIST_SIZE(10);
  list<int> my_list;

  for(int i = 0; i < 100; i++){
    if( my_list.size() >= 10)
      my_list.pop_front();
    my_list.push_back(i);

    showlist(my_list);
  }

  return 0;
}