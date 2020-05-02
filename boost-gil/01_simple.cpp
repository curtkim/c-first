// Always assume
#include <vector>
#include <boost/gil.hpp>

using namespace std;
using namespace boost;
using namespace boost::gil;

/*
auto get_red(pixel_t p)
{
  return get_color(p, red_t());
}
*/

int main() {

  rgba8_image_t src(1,1);

  // get the first pixel of an image
  auto first_pixel = view(src)[0];

  // get first channel of first pixel of an image (should be uint8_t in this case)
  uint8_t first_channel_of_first_pixel = view(src)[0][0];



  argb8_image_t img( 640, 480 );
  fill_pixels(view(img), argb8_pixel_t(0, 255, 0, 0 ));

  // get a view to the first channel
  auto v = nth_channel_view( view(img), 1 );

  // convert the first channel view to gray8
  auto c = color_converted_view<gray8_pixel_t>( v );
}
