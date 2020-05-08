#include <torch/script.h> // One-stop header.

//#include "torchvision/DeformConv.h"
#include "torchvision/PSROIAlign.h"
#include "torchvision/PSROIPool.h"
#include "torchvision/ROIAlign.h"
#include "torchvision/ROIPool.h"
#include "torchvision/empty_tensor_op.h"
#include "torchvision/nms.h"

#include <iostream>
#include <memory>

// https://github.com/pytorch/vision/pull/1407#issuecomment-562096191
static auto registry =
  torch::RegisterOperators()
    .op("torchvision::nms", &nms)
    .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor",
        &roi_align)
    .op("torchvision::roi_pool", &roi_pool)
    .op("torchvision::_new_empty_tensor_op", &new_empty_tensor)
    .op("torchvision::ps_roi_align", &ps_roi_align)
    .op("torchvision::ps_roi_pool", &ps_roi_pool);
//                .op("torchvision::deform_conv2d", &deform_conv2d)
//                .op("torchvision::_cuda_version", &_cuda_version)


int main(int argc, const char* argv[]) {

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("../../fasterrcnn_resnet50_fpn.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << e.msg() << std::endl;
        std::cerr << "error loading the model\n";
        return -1;
    }

    //std::cout << module << std::endl;

    /*
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(1, 0, 5) << '\n';
    */

    std::cout << "ok\n";
    return 0;
}