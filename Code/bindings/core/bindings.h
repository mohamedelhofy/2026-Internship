#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_tensor(py::module_& m);
void bind_layers(py::module_& m);
void bind_losses(py::module_& m);
void bind_data(py::module_& m);
void bind_transforms(py::module_& m);
void bind_dataloader(py::module_& m);
void bind_optimizers(py::module_& m);
void bind_training(py::module_& m);
