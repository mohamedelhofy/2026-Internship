#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "data/dataloader.h"
#include "data/image_dataset.h"

namespace py = pybind11;

void bind_dataloader(py::module_& m) {
    py::class_<ImageDataset, Dataset, std::shared_ptr<ImageDataset>>(m, "ImageDataset")
        .def(py::init<const std::vector<std::string>&, const std::vector<Tensor>&,
                      std::shared_ptr<Transform>>(),
             py::arg("image_paths"), py::arg("labels"), py::arg("transform") = nullptr)
        .def("__len__", &ImageDataset::size)
        .def("__getitem__", &ImageDataset::get_item);

    py::class_<DataLoader, std::shared_ptr<DataLoader>>(m, "DataLoader")
        .def(py::init<const Dataset&, Tensor::size_type, bool, std::optional<unsigned int>>(),
             py::arg("dataset"), py::arg("batch_size"), py::arg("shuffle") = false,
             py::arg("seed") = py::none())
        .def("__len__", &DataLoader::size)
        .def(
            "__iter__",
            [](const DataLoader& loader) {
                return py::make_iterator(loader.begin(), loader.end());
            },
            py::keep_alive<0, 1>());
}
