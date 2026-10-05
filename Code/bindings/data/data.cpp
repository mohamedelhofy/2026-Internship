#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "data/dataset.h"
#include "data/tensor_dataset.h"

namespace py = pybind11;

void bind_data(py::module_& m) {
    py::class_<Dataset, std::shared_ptr<Dataset>>(m, "Dataset");

    py::class_<TensorDataset, Dataset, std::shared_ptr<TensorDataset>>(m, "TensorDataset")
        .def(py::init<const std::vector<Tensor>&>(), py::arg("tensors"))
        .def("__len__", &TensorDataset::size)
        .def("__getitem__", &TensorDataset::get_item);
}
