#include <openvdb/openvdb.h>
#include <iostream>


int main(){
  openvdb::initialize();
  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();

  std::cout << "Testing random access:\n";
  openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

  // Define a coordinate with large signed indies.
  openvdb::Coord xyz(1000, -20000000, 30000000);
  accessor.setValue(xyz, 1.0);

  std::cout << "Grid" << xyz << "=" << accessor.getValue(xyz) << "\n";
  xyz.reset(1000, 20000000, -30000000);
  // Verify that the voxel value at (1000, 20000000, -30000000) is the background value 0

  std::cout << "Grid" << xyz << "=" << accessor.getValue(xyz) << "\n";
  accessor.setValue(xyz, 2.0);
  // Set the voxels at the two extremes of the available coordinate space.
  // For 32-bit signed coordinates these are (-2147483648, -2147483648, -2147483648)
  // and (2147483647, 2147483647, 2147483647).
  accessor.setValue(openvdb::Coord::min(), 3.0f);
  accessor.setValue(openvdb::Coord::max(), 4.0f);

  std::cout << "Testing sequential access:\n";
  for( openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter){
    std::cout << "Grid" << iter.getCoord() << " = " << *iter << "\n";
  }
}
