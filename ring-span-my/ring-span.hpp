#pragma once

#include <cstddef>


template< class RS, bool is_const >
class ring_iterator
{
public:
  typedef ring_iterator<RS, is_const> type;

  typedef std::ptrdiff_t difference_type;
  typedef typename RS::value_type value_type;

  typedef typename std::conditional<is_const, const value_type, value_type>::type * pointer;
  typedef typename std::conditional<is_const, const value_type, value_type>::type & reference;
  typedef std::random_access_iterator_tag iterator_category;

  ring_iterator() = default;

#if nsrs_RING_SPAN_LITE_EXTENSION
  // conversion to const iterator:

    operator ring_iterator<RS, true>() const nsrs_noexcept
    {
        return ring_iterator<RS, true>( m_idx, m_rs );
    }
#endif

  // access content:

  reference operator*() const noexcept
  {
    return m_rs->at_( m_idx );
  }

  // see issue #21:

  pointer operator->() const noexcept
  {
    return & m_rs->at_( m_idx );
  }

  // advance iterator:

  type & operator++() noexcept
  {
    ++m_idx; return *this;
  }

  type operator++( int ) noexcept
  {
    type r(*this); ++*this; return r;
  }

  type & operator--() noexcept
  {
    --m_idx; return *this;
  }

  type operator--( int ) noexcept
  {
    type r(*this); --*this; return r;
  }

#if defined(__clang__) || defined(__GNUC__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

  type & operator+=( int i ) noexcept
  {
    m_idx += i; return *this;
  }

  type & operator-=( int i ) noexcept
  {
    m_idx -= i; return *this;
  }

#if defined(__clang__) || defined(__GNUC__)
# pragma GCC diagnostic pop
#endif

#if nsrs_RING_SPAN_LITE_EXTENSION

  template< bool C >
    difference_type operator-( ring_iterator<RS,C> const & rhs ) const nsrs_noexcept
    {
        return static_cast<difference_type>( this->m_idx ) - static_cast<difference_type>( rhs.m_idx );
    }
#endif

  // comparison:

  template< bool C >
  bool operator<( ring_iterator<RS,C> const & rhs ) const noexcept
  {
    assert( this->m_rs == rhs.m_rs ); return ( this->m_idx < rhs.m_idx );
  }

  template< bool C >
  bool operator==( ring_iterator<RS,C> const & rhs ) const noexcept
  {
    assert( this->m_rs == rhs.m_rs ); return ( this->m_idx == rhs.m_idx );
  }

  // other comparisons expressed in <, ==:

  template< bool C >
  inline bool operator!=( ring_iterator<RS,C> const & rhs ) const noexcept
  {
    return ! ( *this == rhs );
  }

  template< bool C >
  inline bool operator<=( ring_iterator<RS,C> const & rhs ) const noexcept
  {
    return ! ( rhs < *this );
  }

  template< bool C >
  inline bool operator>( ring_iterator<RS,C> const & rhs ) const noexcept
  {
    return rhs < *this;
  }

  template< bool C >
  inline bool operator>=( ring_iterator<RS,C> const & rhs ) const noexcept
  {
    return ! ( *this < rhs );
  }

private:
  friend RS;  // clang: non-class friend type 'RS' is a C++11 extension [-Wc++11-extensions]
  friend class ring_iterator<RS, ! is_const>;

  typedef typename RS::size_type size_type;
  typedef typename std::conditional<is_const, const RS, RS>::type ring_type;

  ring_iterator( size_type idx, typename std::conditional<is_const, const RS, RS>::type * rs ) noexcept
    : m_idx( idx ), m_rs ( rs  )
  {}

private:
  size_type   m_idx;
  ring_type * m_rs;
};

// advanced iterator:

template< class RS, bool C >
inline ring_iterator<RS,C> operator+( ring_iterator<RS,C> it, int i ) noexcept
{
  it += i; return it;
}

template< class RS, bool C >
inline ring_iterator<RS,C> operator-( ring_iterator<RS,C> it, int i ) noexcept
{
  it -= i; return it;
}


template<class T>
class ring_span {
public:
  typedef T   value_type;
  typedef T * pointer;
  typedef T & reference;
  typedef T const & const_reference;
  typedef std::size_t size_type;

  typedef ring_iterator< ring_span<T>, false  > iterator;
  typedef ring_iterator< ring_span<T>, true   > const_iterator;


private:
  pointer   m_data;
  size_type m_capacity;
  size_type m_front_idx;
  size_type m_size;

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


  reference operator[]( size_type idx ) noexcept
  {
    assert( idx < m_size ); return at_(idx);
  }

  const_reference operator[]( size_type idx ) const noexcept
  {
    assert( idx < m_size ); return at_(idx);
  }


  reference front() noexcept
  {
    return *begin();
  }

  const_reference front() const noexcept
  {
    return *begin();
  }

  reference back() noexcept
  {
    return *(--end());
  }

  const_reference back() const noexcept
  {
    return *(--end());
  }

  // iteration:

  iterator begin() noexcept
  {
    return iterator( 0, this );
  }

  const_iterator begin() const noexcept
  {
    return cbegin();
  }

  const_iterator cbegin() const noexcept
  {
    return const_iterator( 0, this );
  }

  iterator end() noexcept
  {
    return iterator( size(), this );
  }

  const_iterator end() const noexcept
  {
    return cend();
  }

  const_iterator cend() const noexcept
  {
    return const_iterator( size(), this );
  }

private:
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

};