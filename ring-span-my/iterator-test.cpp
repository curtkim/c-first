#include <assert.h>
#include <iostream>
#include <vector>

using namespace std;

void by_pointer() {
  int ari[]={1,2,3,4,5};
  int *it;
  for (it=&ari[0];it!=&ari[5];it++)
    printf("%d ",*it);
  printf("\n");
}

void by_iterator() {
  vector<int> vi = {1,2,3,4,5};
  vector<int>::iterator it;
  for (it=vi.begin();it!=vi.end();it++)
    printf("%d ",*it);
  printf("\n");
}

int main() {
  by_pointer();
  by_iterator();
}
