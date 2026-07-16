#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_tensor(py::module_& m);
void bind_layers(py::module_& m);
