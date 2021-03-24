#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>
#include <assert.h>
#include <tuple>
#include <string>
#include <array>

template<typename T>
bool isEqual(std::vector<T> const &v1, std::vector<T> const &v2)
{
  return (v1.size() == v2.size() &&
          std::equal(v1.begin(), v1.end(), v2.begin()));
}

struct MyRecord
{
  uint8_t x, y;
  float z;
  std::string name;
  std::tuple<uint64_t, double> pair;
  std::vector<int32_t> vector;
  std::array<std::string, 2> array;

  template <class Archive>
  void serialize( Archive & ar )
  {
    ar( x, y, z, name, std::get<0>(pair), std::get<1>(pair), vector, array);
  }
};

int main(int argc, char** argv)
{
  std::ofstream os("out.cereal", std::ios::binary);
  cereal::BinaryOutputArchive archive( os );

  MyRecord record {1, 2, 3.0, "abcde", std::make_tuple(4, 5.0), {6,7,8,9}, {"a", "b"}};
  archive( record );
  os.close(); // 꼭 필요함.


  std::ifstream is("out.cereal", std::ios::binary);
  cereal::BinaryInputArchive inputArchive(is);

  MyRecord rec2;
  inputArchive(rec2);
  is.close();

  assert(1 == rec2.x );
  assert(2 == rec2.y );
  assert(3.0 == rec2.z );
  assert(0 == rec2.name.compare("abcde"));
  assert(4 == std::get<0>(rec2.pair) );
  assert(5.0 == std::get<1>(rec2.pair) );
  assert(isEqual(rec2.vector, {6,7,8,9}));
  assert(isEqual(rec2.vector, {6,7,8,9}));
  assert(0 == rec2.array[0].compare("a") );
  assert(0 == rec2.array[1].compare("b") );

  return 0;
}

