#include <sstream>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "activations/activations.h"
#include "tensor/tensor.h"

namespace py = pybind11;

// Convert Python tuple -> Tensor::Shape
static Tensor::Shape tuple_to_shape(const py::tuple& t) {
    Tensor::Shape indices;
    indices.reserve(t.size());

    for (const py::handle& item : t) {
        indices.push_back(item.cast<Tensor::size_type>());
    }

    return indices;
}

void bind_tensor(py::module_& m) {
    m.doc() = "TinyTorch Tensor Library";

    py::class_<Tensor>(m, "Tensor")

        // ================= Constructors =================

        .def(py::init<>())

        .def(py::init<const Tensor::Storage&, const Tensor::Shape&>(), py::arg("data"),
             py::arg("shape"))

        // ================= Properties =================

        .def_property_readonly("shape", &Tensor::shape)

        .def_property_readonly(
            "data", static_cast<const Tensor::Storage& (Tensor::*)() const>(&Tensor::data))

        .def_property_readonly("size", &Tensor::size)

        .def_property_readonly("ndim", &Tensor::ndim)

        .def_property_readonly("dtype", &Tensor::dtype)

        .def_property_readonly("empty", &Tensor::empty)

        // ================= Matrix Multiplication =================

        .def("matmul", &Tensor::matmul)

        .def("__matmul__", &Tensor::matmul)

        // ================= Element Access =================

        .def("__getitem__", [](const Tensor& tensor,
                               Tensor::size_type index) { return tensor(Tensor::Shape{index}); })

        .def("__getitem__", [](const Tensor& tensor,
                               const py::tuple& index) { return tensor(tuple_to_shape(index)); })

        .def("__setitem__", [](Tensor& tensor, Tensor::size_type index,
                               Tensor::value_type value) { tensor(Tensor::Shape{index}) = value; })

        .def("__setitem__", [](Tensor& tensor, const py::tuple& index,
                               Tensor::value_type value) { tensor(tuple_to_shape(index)) = value; })

        // ================= Utilities =================

        .def("__len__",
             [](const Tensor& tensor) { return tensor.shape().empty() ? 0 : tensor.shape()[0]; })

        .def("__bool__", [](const Tensor& tensor) { return !tensor.empty(); })

        .def("copy", [](const Tensor& tensor) { return Tensor(tensor); })

        .def("tolist", [](const Tensor& tensor) { return tensor.data(); })
        .def("reshape", &Tensor::reShape)
        .def("transpose", &Tensor::Transpose)
        .def("sum", [](const Tensor& tensor) { return tensor.sum(); })
        .def(
            "sum", [](const Tensor& tensor, Tensor::size_type axis) { return tensor.sum(axis); },
            py::arg("axis"))
        .def("mean", [](const Tensor& tensor) { return tensor.mean(); })
        .def(
            "mean", [](const Tensor& tensor, Tensor::size_type axis) { return tensor.mean(axis); },
            py::arg("axis"))
        .def("max", &Tensor::max)
        .def("min", &Tensor::min)

        .def("relu", &Tensor::relu)
        .def("sigmoid", &Tensor::sigmoid)
        .def("tanh", &Tensor::tanh)
        .def("gelu", &Tensor::gelu)
        .def("softmax", &Tensor::softmax, py::arg("dim") = -1)
        .def("log_softmax", &Tensor::log_softmax, py::arg("dim") = -1)
        .def("log", &Tensor::log)
        .def("clamp", &Tensor::clamp, py::arg("min"), py::arg("max"))

        // ================= Comparison =================

        .def(py::self == py::self)

        // ================= Arithmetic =================

        .def(py::self + py::self)
        .def(py::self + Tensor::value_type())
        .def(Tensor::value_type() + py::self)

        .def(py::self - py::self)
        .def(py::self - Tensor::value_type())
        .def(Tensor::value_type() - py::self)

        .def(py::self * py::self)
        .def(py::self * Tensor::value_type())
        .def(Tensor::value_type() * py::self)

        .def(py::self / py::self)
        .def(py::self / Tensor::value_type())
        .def(Tensor::value_type() / py::self)

        // ================= Printing =================

        .def("__repr__",
             [](const Tensor& tensor) {
                 std::ostringstream oss;
                 oss << tensor;
                 return oss.str();
             })

        .def("__str__",
             [](const Tensor& tensor) {
                 std::ostringstream oss;
                 oss << tensor;
                 return oss.str();
             })

        // Property to get/set requires_grad
        .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)

        // Read-only property to get the gradient tensor (returns None if no grad)
        .def_property_readonly("grad",
                               [](const Tensor& tensor) -> std::optional<Tensor> {
                                   if (tensor.grad_) {
                                       return *tensor.grad_;
                                   }
                                   return std::nullopt;
                               })

        // Backward method
        .def(
            "backward",
            [](Tensor& tensor, std::optional<Tensor> gradient) { tensor.backward(gradient); },
            py::arg("gradient") = py::none())

        // Zero grad
        .def("zero_grad", &Tensor::zero_grad);
}
