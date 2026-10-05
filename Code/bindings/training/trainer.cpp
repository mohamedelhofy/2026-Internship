#include "training/trainer.h"
#include "training/clip_grad.h"
#include "training/scheduler.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// هنا غيرنا الاسم لـ bind_training عشان يطابق الـ bindings.h والـ module.cpp
void bind_training(py::module_& m) {
    py::class_<CosineSchedule, std::shared_ptr<CosineSchedule>>(m, "CosineSchedule")
        .def(py::init<float, float, int>(), py::arg("max_lr"), py::arg("min_lr"),
             py::arg("total_epochs"))
        .def("get_lr", &CosineSchedule::get_lr, py::arg("epoch"));

    m.def("clip_grad_norm", &clip_grad_norm, py::arg("parameters"), py::arg("max_norm") = 1.0f);

    py::class_<Trainer>(m, "Trainer")
        .def(py::init<Sequential&, Optimizer&, MSELoss&, std::shared_ptr<CosineSchedule>, float>(),
             py::arg("model"), py::arg("optimizer"), py::arg("loss_fn"),
             py::arg("scheduler") = nullptr, py::arg("grad_clip_norm") = -1.0f)
        .def("train_epoch", &Trainer::train_epoch, py::arg("dataloader"),
             py::arg("accumulation_steps") = 1)
        .def("evaluate", &Trainer::evaluate, py::arg("dataloader"))
        .def("save_checkpoint", &Trainer::save_checkpoint, py::arg("path"))
        .def("load_checkpoint", &Trainer::load_checkpoint, py::arg("path"))
        .def_readwrite("epoch", &Trainer::epoch)
        .def_readwrite("step", &Trainer::step)
        .def_readwrite("history", &Trainer::history);
}
