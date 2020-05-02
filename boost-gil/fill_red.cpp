#include <boost/gil.hpp>
#include <boost/gil/extension/io/jpeg.hpp>
#include <iostream>

using namespace boost::gil;

int main()
{
  namespace gil = boost::gil;
  gil::rgb8_image_t img(256, 256);
  gil::rgb8_pixel_t red1(255,0,0);     // make a red RGB pixel

  fill_pixels(view(img), red1);

  uint8_t first_channel = view(img)[0][0];
  uint8_t second_channel = view(img)[0][1];
  uint8_t third_channel = view(img)[0][2];
  assert(first_channel == 255);
  assert(second_channel == 0);
  assert(third_channel == 0);
  std::cout << first_channel << " " << second_channel << " "<< third_channel << std::endl;

  std::cout << img._view.num_dimensions << std::endl;
  write_view("red.jpg", const_view(img), jpeg_tag{});

  return 0;
}
