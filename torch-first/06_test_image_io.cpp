//#include <torch/script.h>
#include "precompile.hpp"
#include <assert.h>
#include <iostream>

#include "utils/image_io.h"


using namespace std;
using namespace torch;

int main() {
    namespace F = torch::nn::functional;

    auto origin_image = image_io::load_image("resources/parrots.png"); // [3, 384, 512]
    std::cout << "origin_image.sizes() = " << origin_image.sizes() << std::endl; //

    auto option = F::InterpolateFuncOptions()
            .scale_factor(std::vector<double>({1.5, 1.5}))
            .mode(torch::kNearest)
            .recompute_scale_factor(true);

    // mini-batch x channels x [optional depth] x [optional height] x width
    // (https://pytorch.org/docs/stable/nn.functional.html)
    // dimensions 4를 만들어 주기 위해 unsqueeze를 호출한다.
    auto resized_image = F::interpolate(origin_image.unsqueeze(0), option);

    //auto resized_image = F::interpolate(origin_image, F::InterpolateFuncOptions().size(std::vector<int64_t>({3, 576, 768})).mode(torch::kNearest));
    std::cout << "resized_image.sizes() = " << resized_image.sizes() << std::endl; //
    image_io::save_image(resized_image, "parrots_resized.png");
    //auto torch.nn.functional.interpolate(origin_image);


    auto image = image_io::load_image("resources/parrots.png", {(int)(384*1.5), (int)(512*1.5)});
    std::cout << "image.sizes() = " << image.sizes() << std::endl; // [3, 576, 768]

    image_io::save_image(image, "parrots_1.5x.png");

    //3*500*768 crop
    auto cropped_image = image.index({at::indexing::Slice(0,3), at::indexing::Slice(0, 300), at::indexing::Ellipsis});
    std::cout << "cropped_image.sizes() = " << cropped_image.sizes() << std::endl; // [3, 300, 768]
    //index({at::indexing::Slice(0, 2), 1}).equal(
    image_io::save_image(cropped_image, "parrots_cropped.png");

    return 0;
}
