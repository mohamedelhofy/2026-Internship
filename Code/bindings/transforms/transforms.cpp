#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "data/transforms.h"

namespace py = pybind11;

void bind_transforms(py::module_& m) {
    py::class_<Transform, std::shared_ptr<Transform>>(m, "Transform")
        .def("__call__", [](Transform& self, const Tensor& img) { return self.forward(img); });

    py::class_<RandomHorizontalFlip, Transform, std::shared_ptr<RandomHorizontalFlip>>(
        m, "RandomHorizontalFlip")
        .def(py::init<Tensor::value_type, std::optional<unsigned int>>(), py::arg("p") = 0.5f,
             py::arg("seed") = py::none());

    py::class_<RandomCrop, Transform, std::shared_ptr<RandomCrop>>(m, "RandomCrop")
        .def(py::init<std::pair<Tensor::size_type, Tensor::size_type>, Tensor::size_type,
                      std::optional<unsigned int>>(),
             py::arg("size"), py::arg("padding") = 4, py::arg("seed") = py::none());

    py::class_<Compose, Transform, std::shared_ptr<Compose>>(m, "Compose")
        .def(py::init<const std::vector<std::shared_ptr<Transform>>&>(), py::arg("transforms"));
}
