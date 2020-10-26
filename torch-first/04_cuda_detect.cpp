#include <iostream>
#include <torch/torch.h>

using namespace std;

int main() {

  cout << "CUDA: " << (torch::cuda::is_available() ? "Yes" : "No") << endl;

  cout << "cuDNN: "
       << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << endl;

  auto a = at::zeros({2,2}, at::device(at::kCUDA).dtype(at::kLong));
  cout << a << endl;

}