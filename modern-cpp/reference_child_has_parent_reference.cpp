#include <assert.h>

struct Parent {
  long parent_value;
};

struct Child {
  const Parent& parent;
  long child_value;
};

int main(){
  Parent parent = {1};
  Child child1 = {parent, 2};
  Child child2 = {parent, 3};

  void* parent_addr = &parent;
  void* child1_parent_addr = (void*)&(child1.parent);
  void* child2_parent_addr = (void*)&(child2.parent);

  assert(parent_addr == child1_parent_addr);
  assert(parent_addr == child2_parent_addr);

}
