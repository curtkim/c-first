#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "vector_inner.h"

namespace py = pybind11;

PYBIND11_MODULE(vector,m)
{
    m.doc() = "pybind11 vector plugin";

    m.def("modify", &modify, "Multiply all entries of a list by 2.0");
}