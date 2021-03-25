#include <iostream>
#include <tl/expected.hpp>

struct Image{
  long data;
};

using fail_reason = int;

tl::expected<Image,fail_reason> crop_to_cat(const Image& img){
  return Image{img.data / 10};
}

tl::expected<Image,fail_reason> add_bow_tie(const Image& img){
  return Image{img.data * 2};
}

Image make_smaller(const Image& img){
  return Image{img.data - 1};
}
Image add_rainbow(const Image& img){
  return Image{img.data + 1};
}


tl::expected<Image,fail_reason> get_cute_cat (const Image& img) {
  return crop_to_cat(img)
    .and_then(add_bow_tie)
    .map(make_smaller)
    .map(add_rainbow);
}

int main(int argc, char** argv)
{
  auto img = get_cute_cat(Image{100});
  if(img)
    std::cout << img.value().data << "\n";
  else
    std::cout << "unexpected " << img.error() << "\n";

  return 0;
}

