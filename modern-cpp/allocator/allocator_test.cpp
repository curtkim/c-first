#include <cstdint>
#include <iterator>
#include <vector>
#include <iostream>

template <typename T>
class PreAllocator
{
private:
  T* memory_ptr;
  std::size_t memory_size;

public:
  typedef std::size_t     size_type;
  typedef T*              pointer;
  typedef T               value_type;

  PreAllocator(T* memory_ptr, std::size_t memory_size) : memory_ptr(memory_ptr), memory_size(memory_size) {}

  PreAllocator(const PreAllocator& other) throw() : memory_ptr(other.memory_ptr), memory_size(other.memory_size) {};

  template<typename U>
  PreAllocator(const PreAllocator<U>& other) throw() : memory_ptr(other.memory_ptr), memory_size(other.memory_size) {};

  template<typename U>
  PreAllocator& operator = (const PreAllocator<U>& other) { return *this; }
  PreAllocator<T>& operator = (const PreAllocator& other) { return *this; }
  ~PreAllocator() {}


  pointer allocate(size_type n, const void* hint = 0) {
    std::cout << "\tallocate n=" << n << "\n";
    return memory_ptr;
  }
  void deallocate(T* ptr, size_type n) {}

  size_type max_size() const {return memory_size;}
};

int main()
{
  const int SIZE = 100;
  {
    int my_arr[SIZE] = {9};
    std::cout << "My_Arr[0]: " << my_arr[0] << "\n";

    std::cout << &my_arr << std::endl;
    std::vector<int, PreAllocator<int>> my_vec(0, PreAllocator<int>(&my_arr[0], SIZE));
    std::cout << "My_Vec.size(): " << my_vec.size()
      << "My_Vec.capacity(): " << my_vec.capacity()
      << std::endl;

    // 101개를 push_back하면 에러가 발생한다.
    for(int i = 0; i < SIZE; i++)
      my_vec.push_back(1024);

    std::cout << "My_Vec[0]: " << my_vec[0] << "\n";
    std::cout << "My_Arr[0]: " << my_arr[0] << "\n";
    std::cout << "My_Vec.size(): " << my_vec.size()
              << "My_Vec.capacity(): " << my_vec.capacity()
              << std::endl;


  }

  {
    int *my_heap_ptr = new int[SIZE]();
    std::vector<int, PreAllocator<int>> my_heap_vec(0, PreAllocator<int>(&my_heap_ptr[0], SIZE));
    my_heap_vec.push_back(1024);
    std::cout << "My_Heap_Vec[0]: " << my_heap_vec[0] << "\n";
    std::cout << "My_Heap_Ptr[0]: " << my_heap_ptr[0] << "\n";

    std::cout << "My_Vec.capacity(): " << my_heap_vec.capacity() << std::endl;

    delete[] my_heap_ptr;
    my_heap_ptr = nullptr;
  }
}