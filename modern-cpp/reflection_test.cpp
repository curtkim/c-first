// from https://lordsof.tech/programming/reflection-in-c14/
#include <iostream>
#include <utility>

// Forward declares for ADL
template<typename T, int N> struct ObjectGetter {
  friend void processMember(ObjectGetter<T, N>, T*, size_t);
  friend constexpr int memberSize(ObjectGetter<T, N>);
};

// The class that adds implementations according to its parametres
template<typename T, int N, typename Stored>
struct ObjectDataStorage {
  friend void processMember(ObjectGetter<T, N>, T* instance, size_t offset)
  {
    std::cout << N << ": " <<
              *reinterpret_cast<Stored*>(reinterpret_cast<uint8_t*>(instance)
                                         + offset) << std::endl;
  };
  friend constexpr int memberSize(ObjectGetter<T, N>) {
    return sizeof(Stored);
  }
};
// The class whose conversions cause instantiations of ObjectDataStorage
template<typename T, int N>
struct ObjectInspector {
  template <typename Inspected,
    std::enable_if_t<sizeof(ObjectDataStorage<T, N, Inspected>) !=
                     -1>* = nullptr>
  operator Inspected() {
    return Inspected{};
  }
};
// Partial template specialisation for recursively finding member count
template <typename T, typename sfinae, size_t... indexes>
struct MemberCounter {
  constexpr static size_t get() {
    return sizeof...(indexes) - 1;
  }
};
template <typename T, size_t... indexes>
struct MemberCounter<T,
  decltype( T {ObjectInspector<T, indexes>()...} )*, indexes...> {
  constexpr static size_t get() {
    return MemberCounter<T, T*, indexes..., sizeof...(indexes)>::get();
  }
};
// Calculation of padding, assuming all composite types contain word-sized
// members. True for std::string, smart pointers and all STL containers (NOT
// std::array). Should use something specialised for all supported types in
// the final version
template <size_t previous, size_t size>
constexpr size_t padded() {
  constexpr int wordSize = sizeof(void*);
  return (size==1 || size==2 || size==4 || (size==8 && wordSize==8)) ?
         ((previous + size) % size == 0 ? previous : previous + size - (previous
                                                                        + size) % size) :
         ((previous + size) % wordSize == 0 ? previous : previous
                                                         + wordSize - (previous + size) % wordSize);
}
// Iteration through all elements, the first overload stops the recursion
template <typename T, size_t offset>
void goThroughElements(T* instance, std::index_sequence<>) { }

template <typename T, size_t offset, size_t index, size_t... otherIndexes>
void goThroughElements(T* instance, std::index_sequence<index, otherIndexes...>) {
  constexpr size_t size = memberSize(ObjectGetter<T, index>{});
  constexpr size_t paddedOffset = padded<offset, size>();
  processMember(ObjectGetter<T, index>{}, instance, paddedOffset);
  goThroughElements<T, paddedOffset + size>(instance,
                                            std::index_sequence<otherIndexes...>{});
}
template <typename T>
void iterateObject(T& instance) {
  goThroughElements<T, 0>(&instance,
                          std::make_index_sequence<MemberCounter<T, T*>::get()>());
}

// Usage
#include <string>
struct Mystery {
  int a = 3;
  std::string b = "We don't need C++17!";
  float c = 4.5;
  bool d = true;
  void* e = nullptr;
  short int f = 13;
  double g = 14.34;
};

int main() {
  std::cout << "Members: ";
  std::cout << MemberCounter<Mystery, Mystery*>::get() << std::endl;
  Mystery investigated;
  iterateObject(investigated);
}