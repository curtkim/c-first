#include <iostream>

#include <tao/json/value.hpp>

int main( int /*unused*/, char** /*unused*/ )
{
  tao::json::value v = {
    { "a", 20 },
    { "b", 30 }
  };
  std::cout << v.is_object() << std::endl;
  std::cout << v.is_string() << std::endl;

  std::cout << v.at( "a" ).as<int>() << std::endl;
  std::cout << v.at( "b" ).as<int>() << std::endl;

  auto c = v.optional< int >( "c" );  // b is empty
  std::cout << c.has_value() << std::endl;

  return 0;
}