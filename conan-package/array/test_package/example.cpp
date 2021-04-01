#include "array/array.h"
#include "array/ein_reduce.h"

#include <functional>
#include <iostream>
#include <random>

using namespace nda;

int main()
{
  using my_3d_shape_type = shape<dim<>, dim<>, dim<>>;
  constexpr int width = 16;
  constexpr int height = 10;
  constexpr int depth = 3;
  my_3d_shape_type my_3d_shape(width, height, depth);

  array<int, my_3d_shape_type> my_array(my_3d_shape);

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // Variadic verion:
        my_array(x, y, z) = 5;
        // Or the index_type versions:
        my_array({x, y, z}) = 5;
        my_array[{x, y, z}] = 5;
      }
    }
  }

  my_array.for_each_value([](int& value) {
    value = 5;
  });

    /*
  for_all_indices(my_3d_shape, [&](int x, int y, int z) {
    my_array(x, y, z) = 5;
  });
  for_each_index(my_3d_shape, [&](my_3d_shape_type::index_type i) {
    my_array[i] = 5;
  });
    */
  for_all_indices<2, 0, 1>(my_3d_shape, [](int x, int y, int z) {
    std::cout << x << ", " << y << ", " << z << std::endl;
  });

    return 0;
}
