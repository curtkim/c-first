#include "common.h"
#include <array>
#include <cstring>
#include <algorithm>

using namespace std;

struct GPS {
  double lng;
  double lat;
  int accuracy;

  GPS() {
    std::cout << "created" << std::endl;
  }

  GPS(const GPS &t)
  {
    cout<<"Copy constructor called "<<endl;
    std::memcpy(this, &t, sizeof(GPS));
  }

  GPS& operator=(GPS& p) {
    cout<<"Assign constructor called "<<endl;
    std::memcpy(this, &p, sizeof(GPS));
  }

  ~GPS() {
    std::cout << "destroyed" << std::endl;
  }

};

void print_span(const nonstd::ring_span<GPS>& span) {
  cout << "=== " << __FUNCTION__ << endl;

  std::cout << span[0].lng << std::endl;
  std::cout << span[4].lng << std::endl;
  std::cout << span.front().lng << std::endl;
  std::cout << span.back().lng << std::endl;

  cout << "=== iterate begin ~ end" << endl;
  for(auto it = span.begin(); it < span.end(); it++){
    std::cout << (*it).lng << std::endl;
  }

  cout << "=== iterate for each" << endl;
  // copy constructor가 호출된다.
  for(const auto gps : span){
    std::cout << gps.lng << std::endl;
  }

  std::for_each(span.begin(), span.end(), [](const auto& gps){
    cout << gps.lng << endl;
  });

  cout << "=== " << __FUNCTION__ << endl;
}

int main()
{
  std::array<GPS, 5> gps_array;

  gps_array[0].lng = 0;
  gps_array[1].lng = 1;
  gps_array[2].lng = 2;
  gps_array[3].lng = 3;
  gps_array[4].lng = 4;

  GPS *start = gps_array.begin();
  GPS *end = start + 4;

  for (GPS *it = start; it < end; it++) {
    std::cout << it->lng << std::endl;
  }

  nonstd::ring_span<GPS> span(gps_array.data(), gps_array.data() + gps_array.size(), gps_array.data(), 5);
  print_span(span);
}