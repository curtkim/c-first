#include <geos/geom/LineString.h>
#include <geos/geom/Coordinate.h>
#include <geos/geom/CoordinateArraySequence.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/PrecisionModel.h>
#include <geos/linearref/LengthIndexedLine.h>

#include <iostream>

using namespace std;
using namespace geos::geom;

// replacement of a minimal set of functions:
void* operator new(std::size_t sz) {
  std::printf("global op new called, size = %zu\n",sz);
  return std::malloc(sz);
}
void operator delete(void* ptr) noexcept
{
  std::puts("global op delete called");
  std::free(ptr);
}

int main(int argc, char** argv)
{
  auto fac = GeometryFactory::create();

  vector<Coordinate> coords = {
    Coordinate(0, 0),
    Coordinate(5, 5),
    Coordinate(10, 10)
  };
  CoordinateArraySequence seq(std::move(coords), 0);

  /*
  std::unique_ptr<CoordinateArraySequence> pSeq = make_unique<CoordinateArraySequence>(std::move(seq));
  std::unique_ptr<LineString> line = fac->createLineString(std::move(pSeq));
  line->getLength()
  */

  LineString* pLineString = fac->createLineString(seq);
  cout << "length=" << pLineString->getLength() << endl;

  Point* pPoint = fac->createPoint(Coordinate(5, 0));
  cout << "distance=" << pLineString->distance((Geometry*)pPoint) << endl;

  
  geos::linearref::LengthIndexedLine indexedLine(pLineString);

  double index = indexedLine.project(Coordinate(5, 0, 0));
  cout << "index= " << index << endl;

  Coordinate coord = indexedLine.extractPoint(index);
  cout << "coord= " << coord << endl;

  delete pLineString;
  delete pPoint;

  return 0;
}

