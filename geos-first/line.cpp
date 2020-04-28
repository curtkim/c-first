#include <geos/geom/LineString.h>
#include <geos/geom/Coordinate.h>
#include <geos/geom/CoordinateArraySequence.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/PrecisionModel.h>

#include <iostream>

using namespace std;


int main(int argc, char** argv)
{
  auto fac = geos::geom::GeometryFactory::create();
  auto pseq = new geos::geom::CoordinateArraySequence();

  using geos::geom::Coordinate;

  pseq->add(Coordinate(0, 0, 0));
  pseq->add(Coordinate(5, 5, 5));
  pseq->add(Coordinate(10, 10, 10));

  geos::geom::LineString* pLineString = fac->createLineString(pseq);
  cout << "length=" << pLineString->getLength() << endl;

  geos::geom::Point* pPoint = fac->createPoint(Coordinate(5, 0));
  cout << "distance=" << pLineString->distance((geos::geom::Geometry*)pPoint) << endl;

  //delete pLineString;
  //delete pPoint;

  return 0;
}

