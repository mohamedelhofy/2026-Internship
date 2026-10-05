#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Essential for std::vector, std::optional, std::shared_ptr

#include "losses/binary_cross_entropy_loss.h"
#include "losses/cross_entropy_loss.h"
#include "losses/loss.h"
#include "losses/mse_loss.h"

namespace py = pybind11;

// Function to bind all loss classes to the provided python module
void bind_losses(py::module_& m) {
    // ================= Reduction Mode =================
    // Upper-case names because "None" is a Python keyword and could not be
    // spelled as an attribute.
    py::enum_<Reduction>(m, "Reduction")
        .value("MEAN", Reduction::Mean)
        .value("SUM", Reduction::Sum)
        .value("NONE", Reduction::None);

    // ================= Base Loss =================
    py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss")
        .def("forward", &Loss::forward, py::arg("prediction"), py::arg("target"))
        .def(
            "__call__",
            [](const Loss& self, const Tensor& prediction, const Tensor& target) {
                return self.forward(prediction, target);
            },
            py::arg("prediction"), py::arg("target"))
        .def_property_readonly("reduction", &Loss::reduction);

    // ================= Mean Squared Error =================
    py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init<Reduction>(), py::arg("reduction") = Reduction::Mean);

    // ================= Cross Entropy =================
    py::class_<CrossEntropyLoss, Loss, std::shared_ptr<CrossEntropyLoss>>(m, "CrossEntropyLoss")
        .def(py::init<Reduction>(), py::arg("reduction") = Reduction::Mean)
        .def(
            "forward",
            [](const CrossEntropyLoss& self, const Tensor& logits, const Tensor& targets) {
                return self.forward(logits, targets);
            },
            py::arg("logits"), py::arg("targets"))
        .def(
            "forward",
            [](const CrossEntropyLoss& self, const Tensor& logits,
               const CrossEntropyLoss::IndexList& targets) {
                return self.forward(logits, targets);
            },
            py::arg("logits"), py::arg("targets"))
        .def(
            "__call__",
            [](const CrossEntropyLoss& self, const Tensor& logits, const Tensor& targets) {
                return self.forward(logits, targets);
            },
            py::arg("logits"), py::arg("targets"))
        .def(
            "__call__",
            [](const CrossEntropyLoss& self, const Tensor& logits,
               const CrossEntropyLoss::IndexList& targets) {
                return self.forward(logits, targets);
            },
            py::arg("logits"), py::arg("targets"));

    // ================= Binary Cross Entropy =================
    py::class_<BinaryCrossEntropyLoss, Loss, std::shared_ptr<BinaryCrossEntropyLoss>>(
        m, "BinaryCrossEntropyLoss")
        .def(py::init<Reduction, Tensor::value_type>(), py::arg("reduction") = Reduction::Mean,
             py::arg("epsilon") = BinaryCrossEntropyLoss::default_epsilon)
        .def_property_readonly("epsilon", &BinaryCrossEntropyLoss::epsilon);
}
