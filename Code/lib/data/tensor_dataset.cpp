#include "data/tensor_dataset.h"
#include <algorithm>

TensorDataset::TensorDataset(const std::vector<Tensor>& tensors) : tensors_(tensors) {
    if (tensors_.empty()) {
        throw std::invalid_argument("TensorDataset requires at least one tensor.");
    }

    num_samples_ = tensors_[0].shape().empty() ? 0 : tensors_[0].shape()[0];

    for (const auto& tensor : tensors_) {
        if (tensor.shape().empty() || tensor.shape()[0] != num_samples_) {
            throw std::invalid_argument(
                "All tensors must have the same size in the first dimension.");
        }
    }
}

Tensor::size_type TensorDataset::size() const { return num_samples_; }

std::vector<Tensor> TensorDataset::get_item(Tensor::size_type idx) const {
    if (idx >= num_samples_) {
        throw std::out_of_range("Dataset index out of bounds.");
    }

    std::vector<Tensor> sample;
    sample.reserve(tensors_.size());

    for (const auto& tensor : tensors_) {
        Tensor::Shape sample_shape(tensor.shape().begin() + 1, tensor.shape().end());

        Tensor::size_type sample_size = 1;
        for (auto dim : sample_shape) {
            sample_size *= dim;
        }

        Tensor::Storage sample_data(sample_size);
        Tensor::size_type offset = idx * sample_size;

        std::copy(tensor.data().begin() + offset, tensor.data().begin() + offset + sample_size,
                  sample_data.begin());

        sample.emplace_back(sample_data, sample_shape);
    }

    return sample;
}
