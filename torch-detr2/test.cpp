#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include "lib/image_io.hpp"

using image_io::load_image;
using image_io::save_image;

using namespace std;

int main() {

  std::cout << "CUDA: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
  std::cout << "cuDNN: " << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << std::endl;

  torch::DeviceType device_type = torch::kCUDA;

  std::vector<torch::Tensor> tensorVec;
  torch::load(tensorVec, "output.tensor");
  torch::Tensor probas = tensorVec[0];
  torch::Tensor boxes = tensorVec[1];
  cout << probas.device() << endl;

  std::cout << "probas.softmax(-1).shape " << probas.softmax(-1).sizes() << std::endl;
  // [0, : , :-1]
  at::Tensor probas2 = probas.softmax(-1).index({
    0,
    at::indexing::Ellipsis,
    at::indexing::Slice(at::indexing::None, -1)
  });
  std::cout << "probas2.shape " << probas2.sizes() << std::endl;

  const auto [values, indices] = probas2.max(-1);
  std::cout << "values.shape " << values.sizes() << std::endl;
  at::Tensor keep = values.gt(0.9);
  std::cout << "keep.shape " << keep.sizes() << std::endl;

  std::cout << boxes.index({0, keep}) << std::endl;
  auto columns = boxes.index({0, keep}).unbind(1);
  torch::Tensor x_c = columns[0];
  torch::Tensor y_c = columns[1];
  torch::Tensor w = columns[2];
  torch::Tensor h = columns[3];

  std::vector<torch::Tensor> b;
  b.push_back(x_c - 0.5 * w);
  b.push_back(y_c - 0.5 * h);
  b.push_back(x_c + 0.5 * w);
  b.push_back(y_c + 0.5 * h);
  auto xyxy = torch::stack(b, 1);
  cout << "xyxy" << endl;
  cout << xyxy << endl;

  torch::Tensor whwh = torch::tensor({1066., 800., 1066., 800.}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto xyxy_rescale = xyxy.multiply(whwh);
  cout << xyxy_rescale << endl;

  cout << "probas2.index({keep}).shape" << endl;
  cout << probas2.index({keep}).sizes() << endl;

  cout << "probas2.index({keep}).argmax(1)" << endl;
  cout << probas2.index({keep}).argmax(1) << endl;
  std::cout << "ok\n";
}