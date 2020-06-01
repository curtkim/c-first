#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <pybind11/embed.h> // everything needed for embedding
namespace py = pybind11;
using namespace py::literals;

void test1() {
  py::exec(R"(
        kwargs = dict(name="World", number=42)
        message = "Hello, {name}! The answer is {number}".format(**kwargs)
        print(message)
    )");
}

void test2() {
  auto kwargs = py::dict("name"_a="World", "number"_a=42);
  auto message = "Hello, {name}! The answer is {number}"_s.format(**kwargs);
  py::print(message);
}

void test3() {
  auto locals = py::dict("name"_a="World", "number"_a=42);
  py::exec(R"(
        message = "Hello, {name}! The answer is {number}".format(**locals())
    )", py::globals(), locals);

  auto message = locals["message"].cast<std::string>();
  std::cout << message;
}

int main() {
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive

  py::print("Hello, World!"); // use the Python API
  test1();
  test2();
  test3();
}