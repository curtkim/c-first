#pragma once

#include <pybind11/pybind11.h>

struct Value {
    int a;
};

namespace py = pybind11;

py::class_<Value>(m, "Value")
  .def_readwrite("a", &Value::a)
