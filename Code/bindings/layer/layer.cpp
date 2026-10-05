#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Essential for std::vector, std::optional, std::shared_ptr

#include "activations/activations.h" // <-- Include the activations header
#include "layer/dropout.h"
#include "layer/layer.h"
#include "layer/linear.h"
#include "layer/sequential.h"

namespace py = pybind11;

// Function to bind all layer classes to the provided python module
void bind_layers(py::module_& m) {
    // ================= Base Layer =================
    py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
        .def("forward", &Layer::forward, py::arg("x"), py::arg("training") = true)
        .def(
            "__call__",
            [](Layer& self, const Tensor& x, bool training) { return self.forward(x, training); },
            py::arg("x"), py::arg("training") = true)
        .def("parameters", &Layer::parameters, py::return_value_policy::reference);

    // ================= Linear Layer =================
    py::class_<Linear, Layer, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<Tensor::size_type, Tensor::size_type, bool, std::optional<unsigned int>>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true,
             py::arg("seed") = py::none())
        .def_property_readonly("in_features", &Linear::in_features)
        .def_property_readonly("out_features", &Linear::out_features)
        .def_property_readonly("has_bias", &Linear::has_bias);

    // ================= Dropout Layer =================
    py::class_<Dropout, Layer, std::shared_ptr<Dropout>>(m, "Dropout")
        .def(py::init<Tensor::value_type, std::optional<unsigned int>>(), py::arg("p") = 0.5f,
             py::arg("seed") = py::none())
        .def_property_readonly("p", &Dropout::p);

    // ================= Activations =================
    py::class_<ReLU, Layer, std::shared_ptr<ReLU>>(m, "ReLU").def(py::init<>());

    py::class_<Sigmoid, Layer, std::shared_ptr<Sigmoid>>(m, "Sigmoid").def(py::init<>());

    py::class_<Tanh, Layer, std::shared_ptr<Tanh>>(m, "Tanh").def(py::init<>());

    py::class_<GELU, Layer, std::shared_ptr<GELU>>(m, "GELU").def(py::init<>());

    py::class_<Softmax, Layer, std::shared_ptr<Softmax>>(m, "Softmax")
        .def(py::init<int>(), py::arg("dim") = -1);

    // ================= Sequential Container =================
    py::class_<Sequential, Layer, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(py::init<>())
        // Allow dynamic *args initialization: Sequential(Linear(), Dropout())
        .def(py::init([](const py::args& args) {
            auto seq = std::make_shared<Sequential>();
            for (const auto& item : args) {
                seq->add(item.cast<std::shared_ptr<Layer>>());
            }
            return seq;
        }))
        .def("add", &Sequential::add, py::arg("layer"))
        .def_property_readonly("empty", &Sequential::empty)
        .def("__len__", &Sequential::size);
}
