#define ACCEPT_USE_OF_DEPRECATED_PROJ_API_H
#include <iostream>
#include <proj_api.h>

using namespace std;

int main()
{
  double x = 200000;
  double y = 500000;

  char *srid5181 = "+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=500000 +ellps=GRS80 +units=m +no_defs";
  char *srid4326 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs";

  projPJ source = pj_init_plus(srid5181);
  projPJ target = pj_init_plus(srid4326);

  if(source==NULL || target==NULL)
    return false;

//  x *= DEG_TO_RAD;
//  y *= DEG_TO_RAD;

  int success = pj_transform(source, target, 1, 1, &x, &y, NULL );

  x *= RAD_TO_DEG;
  y *= RAD_TO_DEG;

  cout << success << endl;
  cout << x << ", " << y << endl;

  return 0;
}