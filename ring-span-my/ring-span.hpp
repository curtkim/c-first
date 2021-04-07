#pragma once

#include <cstddef>


template<class T>
class ring_span {
public:
  typedef T   value_type;
  typedef T * pointer;
  typedef T & reference;
  typedef T const & const_reference;
  typedef std::size_t size_type;


private:
  pointer   m_data;
  size_type m_capacity;
  size_type m_size;
  size_type m_front_idx;

public:

  template< class ContiguousIterator >
  ring_span(
    ContiguousIterator   begin
    , ContiguousIterator end
    , ContiguousIterator first
    , size_type          size
  ) noexcept
    : m_data     ( &* begin )
  , m_size     ( size     )
  , m_capacity ( static_cast<size_type>( end   - begin ) )
  , m_front_idx( static_cast<size_type>( first - begin ) )
  {
    assert( m_size <= m_capacity );
  }

  // observers:

  bool empty() const noexcept
  {
    return m_size == 0;
  }

  bool full() const noexcept
  {
    return m_size == m_capacity;
  }

  size_type size() const noexcept
  {
    return m_size;
  }

  size_type capacity() const noexcept
  {
    return m_capacity;
  }


  size_type normalize_( size_type const idx ) const noexcept
  {
    return idx % m_capacity;
  }

  reference at_( size_type idx ) noexcept
  {
    return m_data[ normalize_(m_front_idx + idx) ];
  }

  const_reference at_( size_type idx ) const noexcept
  {
    return m_data[ normalize_(m_front_idx + idx) ];
  }

  reference operator[]( size_type idx ) noexcept
  {
    assert( idx < m_size ); return at_(idx);
  }

  const_reference operator[]( size_type idx ) const noexcept
  {
    assert( idx < m_size ); return at_(idx);
  }



};