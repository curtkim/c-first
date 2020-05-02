// Always assume
#include <vector>
#include <boost/gil.hpp>
#include <iostream>

using namespace std;
using namespace boost;
using namespace boost::gil;

int main() {
  int Width = 640;
  int Height = 480;

  // create a rgb float buffer
  float* src_buffer = new float[ 3 * Width * Height ];

  // create a gil view of the src buffer
  rgb32f_view_t v = interleaved_view(
      Width
      , Height
      , (rgb32f_pixel_t*) src_buffer
      , 3 * 4 * Width // row length in bytes
  );

  // set pixel values
  fill_pixels(v, rgb32f_pixel_t( 1.f, 1.f, 0.f ));

  // let's create a rgb8 view
  typedef color_converted_view_type<rgb32f_view_t, rgb8_pixel_t>::type ccv_t;
  ccv_t dst = color_converted_view<rgb8_pixel_t>(v);

  // all channels should be 255 which is gil's default behavior
  rgb8_pixel_t p = *dst.xy_at(0,0);
  auto r = get_color(p, red_t());
  auto g = get_color(p, green_t());
  auto b = get_color(p, blue_t());

  cout << (int)r << endl;
  cout << (int)g << endl;
  cout << (int)b << endl;

  return 0;
}
