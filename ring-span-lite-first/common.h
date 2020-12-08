//
// Created by curt on 20. 12. 8..
//

#ifndef RING_SPAN_LITE_0_4_0_COMMON_H
#define RING_SPAN_LITE_0_4_0_COMMON_H

#include "nonstd/ring_span.hpp"
#include <iostream>
#include <numeric>
#include <vector>

template< typename T, size_t N >
inline size_t dim( T (&arr)[N] ) { return N; }

template< typename T, class Popper>
inline std::ostream & operator<<( std::ostream & os, ::nonstd::ring_span<T, Popper> const & rs )
{
  os << "[ring_span: ";
  std::copy( rs.begin(), rs.end(), std::ostream_iterator<T>(os, ", ") );
  return os << "]";
}

template<typename T>
inline std::ostream & operator<<( std::ostream & os, std::vector<T> const & v )
{
  os << "[vector: ";
  std::copy( v.begin(), v.end(), std::ostream_iterator<T>(os, ", ") );
  return os << "]";
}

#endif //RING_SPAN_LITE_0_4_0_COMMON_H
