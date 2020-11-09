#include <list>
#include <iterator>
#include <iostream>
#include <iterator>
#include <vector>

namespace my {
  namespace details {
    template<class RAIter, class Distance>
    void advance(RAIter &it, Distance n, std::random_access_iterator_tag) {
      it += n;
    }

    template<class BidirIter, class Distance>
    void advance(BidirIter &it, Distance n, std::bidirectional_iterator_tag) {
      if (n > 0) {
        while (n--) ++it;
      } else {
        while (n++) --it;
      }
    }

    template<class InputIter, class Distance>
    void advance(InputIter &it, Distance n, std::input_iterator_tag) {
      while (n--) {
        ++it;
      }
    }
  }

  template<class Iter, class Distance>
  void advance(Iter &it, Distance n) {
    details::advance(it, n,
                     typename std::iterator_traits<Iter>::iterator_category{});
  }
}

namespace newstd {
// implementation via constexpr if, available in C++17
  template<class It, class Distance>
  constexpr void advance(It& it, Distance n)
  {
    using category = typename std::iterator_traits<It>::iterator_category;
    static_assert(std::is_base_of_v<std::input_iterator_tag, category>);

    auto dist = typename std::iterator_traits<It>::difference_type(n);
    if constexpr (std::is_base_of_v<std::random_access_iterator_tag, category>)
      it += dist;
    else {
      while (dist > 0) {
        --dist;
        ++it;
      }
      if constexpr (std::is_base_of_v<std::bidirectional_iterator_tag, category>) {
        while (dist < 0) {
          ++dist;
          --it;
        }
      }
    }
  }
}

int main() {
  {
    std::list<int> region2;

    region2.push_back(1);
    region2.push_back(2);
    region2.push_back(3);

    auto iter = region2.begin();
    my::advance(iter, 2);
    std::cout << *iter << '\n';
  }

  {
    std::vector<int> v{1, 2, 3};
    auto vi = v.begin();
    my::advance(vi, 2);
    std::cout << *vi << '\n';
  }
}