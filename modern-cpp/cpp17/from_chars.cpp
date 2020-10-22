#include <charconv> // from_char, to_char
#include <iostream>
#include <string>

void doit(const std::string str) {
  int value = 0;
  const auto res = std::from_chars(str.data(),
                                   str.data() + str.size(),
                                   value);

  if (res.ec == std::errc())
    std::cout << "value: " << value << ", distance: " << res.ptr - str.data() << '\n';
  else if (res.ec == std::errc::invalid_argument)
    std::cout << "invalid argument!\n";
  else if (res.ec == std::errc::result_out_of_range)
    std::cout << "out of range! res.ptr distance: " << res.ptr - str.data() << '\n';
}

int main() {
  const std::string str{ "1234" };
  doit(str);

  const std::string str2{ "1234567890" };
  doit(str2);

  const std::string str3{ "12345678901234" };
  doit(str3);

}