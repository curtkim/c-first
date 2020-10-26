#include <ATen/ATen.h>
#include <iostream>
#include <assert.h>
#include <torch/torch.h>

using namespace std;

int main() {

  assert(torch::cuda::is_available());
  if( torch::cuda::is_available()){
    cout << "cuda available" << endl;
  }
  else {
    cout << "cuda unavailable" << endl;
    return 1;
  }


  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2}); // float type
  auto c = a + b.to(at::kInt);

  cout << a << endl;
  cout << b << endl;
  cout << c << endl;

  cout << a.sum(at::kInt) << endl;

  return 0;
}

