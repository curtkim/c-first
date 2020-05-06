#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch;

void create() {
  // eye
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
  std::cout << tensor.type() << " " << tensor.size(0) << " " << tensor.size(1) << std::endl;

  // rand
  std::cout << torch::rand({3,2}) << std::endl;

  // zeros
  std::cout << torch::zeros({3,2}) << std::endl;

  // direct
  std::cout << torch::tensor({5.5f, 3.f}) << std::endl;
}

void op() {
  auto a = torch::eye(3);
  auto b = torch::zeros({3,3});
  auto c = torch::tensor({1, 1, 1});

  std::cout << a+b << std::endl;
  std::cout << torch::add(a,b) << std::endl;

  a.add_(c);
  std::cout << a << std::endl;
}

void by_index() {
  auto a = torch::eye(3);

  cout << a[0] << endl;
  cout << a[0][0] << endl;
  cout << a[0][0].item() << endl;
  cout << a.view({9}) << endl;
  cout << a.view({-1, 9}) << endl;
  cout << a.reshape({-1, 9}) << endl;
}

void from_blob() {

  int blob[] = {1,2,3,4,5,6,7,8,9};
  auto a = torch::from_blob(&blob, {3,3}, at::kInt);
  cout << a << endl;

  auto b = at::zeros({2,2}, at::device(at::kCPU).dtype(at::kLong));
  cout << b << endl;

}

int main() {

  //create();
  //op();
  by_index();
  from_blob();

  return 0;
}
