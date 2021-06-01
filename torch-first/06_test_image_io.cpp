//#include <torch/script.h>
#include "precompile.hpp"
#include <assert.h>
#include <iostream>

#include "utils/image_io.h"


using namespace std;
using namespace torch;

int main() {

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
