#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "optimizers/adam.h"
#include "optimizers/adamw.h"
#include "optimizers/optimizer.h"
#include "optimizers/sgd.h"

namespace py = pybind11;

void bind_optimizers(py::module_& m) {
    // Register the `tinytorch.optimizers` submodule.
    py::module_ opt = m.def_submodule("optimizers", "Optimization algorithms");

    // ================= Base Optimizer =================
    py::class_<Optimizer>(opt, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad)
        .def_property_readonly("step_count", &Optimizer::step_count);

    // ================= SGD =================
    py::class_<SGD, Optimizer>(opt, "SGD")
        .def(py::init<const Optimizer::ParamList&, float, float, float>(), py::arg("params"),
             py::arg("lr") = 0.01f, py::arg("momentum") = 0.0f, py::arg("weight_decay") = 0.0f)
        .def_property_readonly("lr", &SGD::lr)
        .def_property_readonly("momentum", &SGD::momentum)
        .def_property_readonly("weight_decay", &SGD::weight_decay);

    // ================= Adam =================
    py::class_<Adam, Optimizer>(opt, "Adam")
        .def(py::init<const Optimizer::ParamList&, float, float, float, float, float>(),
             py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def_property_readonly("lr", &Adam::lr)
        .def_property_readonly("beta1", &Adam::beta1)
        .def_property_readonly("beta2", &Adam::beta2)
        .def_property_readonly("eps", &Adam::eps)
        .def_property_readonly("weight_decay", &Adam::weight_decay);

    // ================= AdamW =================
    py::class_<AdamW, Adam>(opt, "AdamW")
        .def(py::init<const Optimizer::ParamList&, float, float, float, float, float>(),
             py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.01f)
        .def_property_readonly("weight_decay", &AdamW::weight_decay);
}
