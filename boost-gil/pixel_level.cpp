#include <boost/gil.hpp>

int main() {
  namespace gil = boost::gil;

  gil::rgb8_pixel_t p1(255,0,0);     // make a red RGB pixel
  gil::bgr8_pixel_t p2 = p1;         // RGB and BGR are compatible and the channels will be properly mapped.

  assert(p1==p2);               // p2 will also be red.
  assert(p2[0]!=p1[0]);         // operator[] gives physical channel order (as laid down in memory)
  assert(gil::semantic_at_c<0>(p1) == gil::semantic_at_c<0>(p2)); // this is how to compare the two red channels
  get_color(p1, gil::green_t()) = get_color(p2, gil::blue_t());  // channels can also be accessed by name

  const unsigned char* r;
  const unsigned char* g;
  const unsigned char* b;
  gil::rgb8c_planar_ptr_t ptr(r,g,b); // constructing const planar pointer from const pointers to each plane
  gil::rgb8c_planar_ref_t ref=*ptr;   // just like built-in reference, dereferencing a planar pointer returns a planar reference

  return 0;
}

