#include <atomic>
#include <iostream>

struct Node {
  int data;
  Node* next;
};


class Stack {

std::atomic<Node*> head;

public:
  Stack() {
    std::cout << "Stack is " <<  (head.is_lock_free() ? "" : "not ")  << "lock-free" << std::endl;
  }

  void Push(int data) {
    Node* new_node = new Node{data, nullptr};
    new_node->next = head.load();
    while(!std::atomic_compare_exchange_weak(  &head,  &new_node->next,  new_node));
  }
};

int main() {
  Stack s;
  s.Push(1);
}
