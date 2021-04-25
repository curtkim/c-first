//#include <torch/script.h>
#include "precompile.hpp"
#include <iostream>

using namespace std;
using namespace torch;

Tensor my_sum(Tensor& tensor) {
    return tensor.sum();
}


void function_param(){
    cout << "\n================= "<< __FUNCTION__ << endl;
    Tensor tensor = at::eye(2, at::kLong);
    Tensor tensor2 = my_sum(tensor);
    cout << tensor2 << " " << "\n";

    //assert(tensor.equal(torch::tensor({2}, at::kLong)));
}

void create() {
  cout << "\n================= "<< __FUNCTION__ << endl;
  // eye
  Tensor tensor = at::eye(2);
  std::cout << tensor << std::endl;
  std::cout << tensor.options().dtype() << " " << tensor.size(0) << " " << tensor.size(1) << std::endl;

  assert(tensor.equal(torch::tensor({
    {1, 0},
    {0, 1}
  }, at::kFloat)));

  // rand
  std::cout << torch::rand({3,2}) << std::endl;

  // zeros
  std::cout << torch::zeros({3,2}) << std::endl;

  // direct
  std::cout << torch::tensor({5.5f, 3.f}) << std::endl;
}

void op() {
  cout << "\n================= op" << endl;
  auto a = torch::eye(3);
  auto b = torch::zeros({3,3});
  auto c = tensor({1, 1, 1});
  auto d = tensor({1, 2, 3}, at::kDouble);


  std::cout << a+b << std::endl;
  std::cout << add(a,b) << std::endl;

  a.add_(c);
  std::cout << a << std::endl;

  std::cout << c / d << std::endl;
}

void by_index() {
  cout << "\n================= by_index" << endl;

  auto a = torch::eye(3);

  cout << "a[0]=" << a[0] << endl;
  cout << "a[0][0]=" << a[0][0] << endl;
  cout << "a[0][0].item()=" << a[0][0].item() << endl;
  cout << "a.view({9})=" << a.view({9}) << endl;
  cout << "a.view({-1, 9})=" << a.view({-1, 9}) << endl;
  cout << "a.reshape({-1, 9})=" << a.reshape({-1, 9}) << endl;
}

void from_blob() {
  cout << "\n================= from_blob" << endl;

  int blob[] = {1,2,3,4,5,6,7,8,9};
  auto a = torch::from_blob(&blob, {3,3}, at::kInt);
  cout << a << endl;
  cout << a.sum() << endl;

  auto b = at::zeros({2,2}, at::device(at::kCPU).dtype(at::kLong));
  cout << b << endl;

  auto c = at::zeros({2,2}, at::device(at::kCUDA).dtype(at::kLong));
  cout << c << endl;
}

void broadcast() {
  cout << "\n================= broadcast" << endl;
  auto sample = torch::randn({3, 2}, at::kFloat) * torch::tensor({1.0, 3.5}) + torch::tensor({2.0, 20.0});
  auto mu = sample.mean({0});

  cout << sample << endl;
  cout << mu << endl;
  cout << sample - mu << endl;

  auto z_score = (sample - mu) / sample.std({0});
  cout << z_score << endl;
}

int main() {

  function_param();
  create();
  op();
  by_index();
  from_blob();
  broadcast();

  return 0;
}
