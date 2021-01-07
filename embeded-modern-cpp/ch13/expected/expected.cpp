#include <iostream>
#include <system_error>
#include <variant>
#include <unistd.h>
#include <sys/fcntl.h>


template <typename T>
class Expected {

std::variant<T, std::error_code> v;

public:
  Expected(T val) : v(val) {}
  Expected(std::error_code e) : v(e) {}

  bool valid() const {
    return std::holds_alternative<T>(v);
  }
  const T& value() const {
    return std::get<T>(v);
  }
  const std::error_code& error() const {
    return std::get<std::error_code>(v);
  }
};

Expected<int> OpenForRead(const std::string& name) {
  int fd = ::open(name.c_str(), O_RDONLY);
  if (fd < 0) {
    return Expected<int>(std::error_code(errno, std::system_category()));
  }
  return Expected<int>(fd);
}

int main() {

  std::cout << "sizeof(std::error_code) = " << sizeof(std::error_code) << std::endl;

  auto result = OpenForRead("nonexistent.txt");
  if (result.valid()) {
    std::cout << "File descriptor" << result.value() << std::endl;
  } else {
    std::cout << "Open failed: "<< result.error().message() << " " << result.error().value() << " ENOENT=" << ENOENT << std::endl;
    std::cout << result.error().category().name() << std::endl;
  }
  return 0;
}