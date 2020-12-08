#include "common.h"

#include <memory>
#include <vector>

struct Widget{};

int main()
{
  std::vector< std::unique_ptr<Widget>> v( 10 );
  nonstd::ring_span< std::unique_ptr<Widget> > r( v.begin(), v.end() );

  r.push_back( std::make_unique<Widget>() );
  r.push_back( std::make_unique<Widget>() );

  r = nonstd::ring_span< std::unique_ptr<Widget> >( v.begin(), v.end() );
}