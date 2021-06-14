#include <iostream>
// 아래 include가 없으면 다음 compile error가 발생한다.
// error: ‘is_available’ is not a member of ‘at::cuda’ r
#include <torch/torch.h>
#include <torchvision/models/resnet.h>

int main()
{
  auto model = vision::models::ResNet18();
  model->eval();

  // Create a random input tensor and run it through the model.
  auto in = torch::rand({1, 3, 10, 10});
  auto out = model->forward(in);

  std::cout << out.sizes();

  if (torch::cuda::is_available()) {
    // Move model and inputs to GPU
    model->to(torch::kCUDA);
    auto gpu_in = in.to(torch::kCUDA);
    auto gpu_out = model->forward(gpu_in);

    std::cout << gpu_out.sizes();
  }
}